import os
import tempfile
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document

# Document processing
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    PDFMinerLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector storage - Using ChromaDB
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_chroma import Chroma

# Cross-encoder for re-ranking
from sentence_transformers import CrossEncoder

# Ollama integration for LLM
import ollama
import time


class ImprovedRAG:
    def __init__(
        self,
        llm_model_name: str = "llama3",
        embedding_model_name: str = "nomic-embed-text",
        data_dir: str = "data",
        db_dir: str = "vectordb",
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        k_retrieval: int = 15,  # Increased from 10 for better coverage
        temperature: float = 0.1,  # Added temperature control for LLM
    ):
        """
        Initialize the ImprovedRAG system with advanced features.

        Args:
            llm_model_name: Name of the Ollama model to use
            embedding_model_name: Name of the embedding model
            data_dir: Directory containing documents to index
            db_dir: Directory to store the vector database
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            k_retrieval: Number of documents to retrieve initially
            temperature: Temperature setting for the LLM (lower = more focused)
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.temperature = temperature
        self.ollama_api_base = os.environ.get(
            "OLLAMA_API_BASE", "http://localhost:11434")

        # Check Ollama availability
        self._check_ollama_availability()

        # Initialize text splitter with improved settings for better context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            keep_separator=True  # Keep separators to maintain readability
        )

        # Initialize or load vector database
        self._init_vectordb()

        # System prompt for LLM - enhanced for better output quality
        self.system_prompt = """
        You are an AI assistant tasked with providing detailed answers based solely on the given context. 
        Your goal is to analyze the information provided and formulate a comprehensive, well-structured 
        response to the question.

        Context will be passed as "Context:"
        User question will be passed as "Question:"

        To answer the question:
        1. Thoroughly analyze the context, identifying key information relevant to the question.
        2. Organize your thoughts and plan your response to ensure a logical flow of information.
        3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
        4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
        5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
        6. If the question is about code or technical implementation, include relevant code snippets or technical details from the context.
        7. When responding to queries about blockchain or web3 technology, be precise with terminology and technical explanations.

        Format your response as follows:
        1. Use clear, concise language.
        2. Organize your answer into paragraphs for readability.
        3. Use bullet points or numbered lists where appropriate to break down complex information.
        4. If relevant, include any headings or subheadings to structure your response.
        5. Ensure proper grammar, punctuation, and spelling throughout your answer.
        6. When mentioning technical terms, briefly explain them if they appear to be domain-specific.

        Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
        """

        # Initialize cross-encoder for re-ranking with caching
        self.cross_encoder_cache = {}
        try:
            # Use a more recent cross-encoder model if available
            cross_encoder_models = [
                "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Try a better model first
                "cross-encoder/ms-marco-MiniLM-L-6-v2",   # Fall back to original
            ]

            for model in cross_encoder_models:
                try:
                    self.cross_encoder = CrossEncoder(model)
                    self.has_cross_encoder = True
                    st.sidebar.success(f"Using cross-encoder model: {model}")
                    break
                except Exception:
                    continue
            else:
                raise Exception(
                    "No cross-encoder models were successfully loaded")

        except Exception as e:
            st.warning(
                f"Could not initialize Cross-Encoder: {str(e)}. Continuing without re-ranking.")
            self.has_cross_encoder = False

    def _check_ollama_availability(self):
        """Check if Ollama server is available and running"""
        try:
            # Try to list models to check connection
            _ = ollama.list()
            st.sidebar.success("Connected to Ollama server")
        except Exception as e:
            st.sidebar.error(f"Error connecting to Ollama server: {str(e)}")
            st.sidebar.info(
                "Please make sure Ollama is running at: " + self.ollama_api_base)

            # Keep trying in background for a few seconds
            for i in range(3):
                try:
                    time.sleep(2)
                    _ = ollama.list()
                    st.sidebar.success(
                        "Successfully connected to Ollama server!")
                    return
                except:
                    pass

    def _init_vectordb(self):
        """Initialize or load the vector database with enhanced error handling and recovery."""
        # Ensure db_dir exists
        os.makedirs(self.db_dir, exist_ok=True)

        # Flag to track if we're using persistent or in-memory DB
        self.using_persistent_db = True

        # Create embedding function using Ollama
        try:
            # Improved embedding function with configurable URL
            embeddings_url = f"{self.ollama_api_base}/api/embeddings"
            self.ollama_ef = OllamaEmbeddingFunction(
                url=embeddings_url,
                model_name=self.embedding_model_name
            )
            st.sidebar.success(
                f"Using {self.embedding_model_name} for embeddings")
        except Exception as e:
            st.sidebar.error(
                f"Error initializing embedding function: {str(e)}")
            st.stop()  # Critical error - can't continue without embeddings

        # Three-stage fallback for database initialization
        # 1. Try persistent DB with advanced settings
        try:
            # Advanced ChromaDB client with custom settings
            chroma_client = chromadb.PersistentClient(
                path=self.db_dir,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )

            # Get or create collection with optimized parameters
            self.collection = chroma_client.get_or_create_collection(
                name="document_collection",
                embedding_function=self.ollama_ef,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 100,  # Higher accuracy during construction
                    "hnsw:search_ef": 50,         # Better search quality
                    "hnsw:M": 16                  # More connections per node
                }
            )

            # Create wrapper for LangChain compatibility with improved configuration
            self.vectordb = Chroma(
                client=chroma_client,
                collection_name="document_collection",
                embedding_function=self.ollama_ef
            )

            st.sidebar.success(
                "Connected to optimized persistent vector database.")

        # 2. Try basic persistent DB if advanced fails
        except Exception as e1:
            st.sidebar.warning(f"Advanced DB initialization failed: {str(e1)}")
            st.sidebar.info("Trying basic persistent database...")

            try:
                # Basic ChromaDB client without custom settings
                chroma_client = chromadb.PersistentClient(path=self.db_dir)

                # Basic collection
                self.collection = chroma_client.get_or_create_collection(
                    name="document_collection",
                    embedding_function=self.ollama_ef
                )

                # Basic Chroma wrapper
                self.vectordb = Chroma(
                    client=chroma_client,
                    collection_name="document_collection",
                    embedding_function=self.ollama_ef
                )

                st.sidebar.success(
                    "Connected to basic persistent vector database.")

            # 3. Fall back to in-memory DB as last resort
            except Exception as e2:
                st.sidebar.error(f"Persistent database failed: {str(e2)}")
                st.sidebar.warning(
                    "Falling back to in-memory database (changes won't persist).")

                try:
                    # In-memory client
                    client = chromadb.EphemeralClient()
                    self.collection = client.create_collection(
                        name="document_collection",
                        embedding_function=self.ollama_ef
                    )

                    # In-memory Chroma wrapper
                    self.vectordb = Chroma(
                        client=client,
                        collection_name="document_collection",
                        embedding_function=self.ollama_ef
                    )

                    self.using_persistent_db = False
                    st.sidebar.info(
                        "Using in-memory database for this session only.")

                except Exception as e3:
                    st.sidebar.error(
                        f"Fatal error: Could not initialize any database: {str(e3)}")
                    st.stop()  # Critical error - can't continue without vector DB

        # Create optimized retriever with advanced parameters
        self.retriever = self.vectordb.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": self.k_retrieval,
                "fetch_k": self.k_retrieval * 3,  # Fetch more candidates for MMR
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )

        # Show database stats
        try:
            count = self.collection.count()
            if count > 0:
                st.sidebar.info(f"Database contains {count} document chunks")

                # Show collection info if available
                try:
                    # Display some metadata about the collection
                    collection_info = {
                        "Documents": count,
                        "Database type": "Persistent" if self.using_persistent_db else "In-memory",
                        "Embedding model": self.embedding_model_name
                    }

                    with st.sidebar.expander("Database details"):
                        for key, value in collection_info.items():
                            st.write(f"**{key}:** {value}")
                except:
                    pass
            else:
                st.sidebar.info("Database is empty. Please ingest documents.")
        except Exception as e:
            st.sidebar.warning(f"Could not fetch document count: {str(e)}")
            pass

    def process_document(self, uploaded_file):
        """
        Process an uploaded document file by converting it to text chunks with robust error handling.

        Args:
            uploaded_file: A Streamlit UploadedFile object

        Returns:
            A list of Document objects containing the chunked text
        """
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_size = len(uploaded_file.getbuffer())

        # Check file size
        if file_size > 50 * 1024 * 1024:  # 50 MB limit
            st.warning(
                f"File {uploaded_file.name} is too large ({file_size/1024/1024:.1f} MB). Maximum size is 50 MB.")
            return []

        # Save uploaded file to a temporary location with correct extension
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension)
        temp_file.write(uploaded_file.getbuffer())
        temp_file.close()

        docs = []
        error_msg = None

        try:
            # Load document based on file extension with multiple fallback options
            if file_extension == ".pdf":
                # Try PyPDFLoader first
                try:
                    loader = PyPDFLoader(temp_file.name, extract_images=False)
                    docs = loader.load()
                    st.success(
                        f"Loaded PDF successfully using PyPDFLoader: {uploaded_file.name}")
                except Exception as pdf_error:
                    st.warning(f"Primary PDF loader failed: {str(pdf_error)}")

                    # Fallback to PDFMinerLoader
                    try:
                        loader = PDFMinerLoader(temp_file.name)
                        docs = loader.load()
                        st.success(
                            f"Loaded PDF using backup PDFMinerLoader: {uploaded_file.name}")
                    except Exception as backup_error:
                        error_msg = f"All PDF loaders failed: {str(backup_error)}"

            elif file_extension == ".txt":
                loader = TextLoader(temp_file.name, encoding="utf-8")
                docs = loader.load()
                st.success(f"Loaded text file: {uploaded_file.name}")

            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file.name)
                docs = loader.load()
                st.success(f"Loaded Word document: {uploaded_file.name}")

            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(temp_file.name)
                docs = loader.load()
                st.success(f"Loaded markdown file: {uploaded_file.name}")

            else:
                st.warning(f"Unsupported file type: {file_extension}")
                return []

            # If we got any docs, process them
            if docs:
                # Add file metadata to all documents
                for doc in docs:
                    if not hasattr(doc, "metadata"):
                        doc.metadata = {}
                    doc.metadata.update({
                        "source": uploaded_file.name,
                        "file_type": file_extension[1:],  # Remove the dot
                        "date_processed": time.strftime("%Y-%m-%d %H:%M:%S")
                    })

                # Split into chunks with progress indicator
                with st.spinner(f"Splitting {len(docs)} document pages into chunks..."):
                    text_splits = self.text_splitter.split_documents(docs)

                # Log statistics
                st.success(
                    f"Created {len(text_splits)} chunks from {len(docs)} document pages")

                # Debugging info
                with st.expander("Document Processing Details"):
                    st.write(f"**File:** {uploaded_file.name}")
                    st.write(f"**Size:** {file_size/1024:.1f} KB")
                    st.write(f"**Pages extracted:** {len(docs)}")
                    st.write(f"**Chunks created:** {len(text_splits)}")
                    if len(text_splits) > 0:
                        st.write(
                            f"**Average chunk size:** {sum(len(chunk.page_content) for chunk in text_splits)/len(text_splits):.1f} characters")

                return text_splits
            else:
                if error_msg:
                    st.error(error_msg)
                return []

        except Exception as e:
            st.error(
                f"Error processing document {uploaded_file.name}: {str(e)}")
            return []

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def ingest_documents(self, uploaded_files):
        """
        Ingest uploaded documents into the vector database with batching and progress tracking.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            Boolean indicating success or failure
        """
        if not uploaded_files:
            st.warning("No files were uploaded.")
            return False

        all_splits = []
        file_names = []

        # Process each uploaded file with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_files = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress_value = (i / total_files)
            progress_bar.progress(progress_value)
            status_text.text(
                f"Processing file {i+1}/{total_files}: {uploaded_file.name}")

            # Normalize filename for use as ID
            normalized_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            file_names.append(normalized_file_name)

            # Process the document
            splits = self.process_document(uploaded_file)
            if splits:
                all_splits.extend(splits)
                st.success(
                    f"✓ Processed {uploaded_file.name}: {len(splits)} chunks")

        # Update final progress
        progress_bar.progress(1.0)
        status_text.text("Processing complete. Adding to database...")

        if not all_splits:
            st.warning("No valid documents were processed.")
            return False

        # Add documents to the vector database with batching for better performance
        try:
            # Prepare data for ChromaDB
            documents, metadatas, ids = [], [], []

            for idx, split in enumerate(all_splits):
                file_idx = idx % len(file_names)
                doc_id = f"{file_names[file_idx]}_{idx}"

                documents.append(split.page_content)
                metadatas.append(split.metadata)
                ids.append(doc_id)

            # Add to collection with batch size limiting for large datasets
            BATCH_SIZE = 100  # Process in batches to avoid memory issues
            total_docs = len(documents)
            batch_count = (total_docs + BATCH_SIZE -
                           1) // BATCH_SIZE  # Ceiling division

            batch_progress = st.progress(0)
            batch_status = st.empty()

            for i in range(batch_count):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, total_docs)

                batch_status.text(
                    f"Adding batch {i+1}/{batch_count} to database...")
                batch_progress.progress((i + 0.5) / batch_count)

                # Add batch to collection
                self.collection.upsert(
                    documents=documents[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx],
                    ids=ids[start_idx:end_idx]
                )

                batch_progress.progress((i + 1) / batch_count)

            batch_status.text("Database update complete!")

            # Display summary
            st.success(
                f"✅ Added {len(all_splits)} document chunks to database from {len(uploaded_files)} files")

            # Show database info
            if self.using_persistent_db:
                st.info("Vector database persisted to disk successfully.")
            else:
                st.warning(
                    "Using in-memory database (changes won't persist between sessions).")

            return True

        except Exception as e:
            st.error(f"Error adding documents to database: {str(e)}")
            return False

    def re_rank_documents(self, query: str, documents: List[str], top_k: int = 3) -> Tuple[str, List[int]]:
        """
        Re-rank documents using the cross-encoder model with caching for better relevance.

        Args:
            query: The user's query
            documents: List of document strings to re-rank
            top_k: Number of top documents to return

        Returns:
            Tuple of (concatenated relevant text, list of relevant document indices)
        """
        if not documents:
            return "", []

        if not self.has_cross_encoder:
            # If cross-encoder is not available, just return top documents
            return "\n\n".join(documents[:top_k]), list(range(min(top_k, len(documents))))

        # Check cache first
        cache_key = f"{query}_{hash(str(documents))}"
        if cache_key in self.cross_encoder_cache:
            return self.cross_encoder_cache[cache_key]

        try:
            with st.spinner("Re-ranking documents for relevance..."):
                # Create pairs for re-ranking
                pairs = [(query, doc) for doc in documents]

                # Score all query-document pairs
                scores = self.cross_encoder.predict(pairs)

                # Create a list of (score, index) tuples and sort by score
                doc_score_pairs = sorted(
                    [(score, idx) for idx, score in enumerate(scores)],
                    key=lambda x: x[0],
                    reverse=True
                )

                # Extract indices of top_k documents
                relevant_text_ids = [idx for _, idx in doc_score_pairs[:top_k]]

                # Combine the relevant documents, sorted by relevance
                relevant_docs = [documents[idx] for idx in relevant_text_ids]
                relevant_text = "\n\n".join(relevant_docs)

                # Store in cache
                self.cross_encoder_cache[cache_key] = (
                    relevant_text, relevant_text_ids)

                return relevant_text, relevant_text_ids

        except Exception as e:
            st.warning(
                f"Error in re-ranking: {str(e)}. Using top documents without re-ranking.")
            # Fallback to returning top documents without re-ranking
            return "\n\n".join(documents[:top_k]), list(range(min(top_k, len(documents))))

    def query(self, query_text: str) -> str:
        """
        Process a query through the RAG system with enhanced error handling and logging.

        Args:
            query_text: The query to process

        Returns:
            String response from the LLM
        """
        try:
            # First check if we have documents
            doc_count = self.collection.count()
            if doc_count == 0:
                return "The knowledge base is empty. Please ingest documents first."

            with st.spinner("Searching knowledge base..."):
                # Query the collection with telemetry for performance analysis
                start_time = time.time()
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=self.k_retrieval,
                    include=["documents", "metadatas", "distances"]
                )
                query_time = time.time() - start_time

                # Get documents and their distances
                documents = results.get("documents")[0]
                distances = results.get("distances")[
                    0] if "distances" in results else None

                # Log statistics
                with st.expander("Search Statistics"):
                    st.write(f"**Query:** {query_text}")
                    st.write(f"**Documents retrieved:** {len(documents)}")
                    st.write(f"**Search time:** {query_time:.3f} seconds")
                    if distances:
                        st.write("**Top 3 document similarities:**")
                        for i in range(min(3, len(distances))):
                            # Convert cosine distance to similarity
                            score = 1 - (distances[i] / 2)
                            st.write(f"- Doc {i+1}: {score:.2%} similar")

                # Re-rank documents for better relevance
                relevant_text, _ = self.re_rank_documents(
                    query_text, documents)

                # Call LLM with the context and query
                response = self._call_llm(relevant_text, query_text)
                return response

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def query_with_sources(self, query_text: str) -> Dict[str, Any]:
        """
        Process a query and return the answer with source documents and enhanced metadata.

        Args:
            query_text: The query to process

        Returns:
            Dictionary with answer, sources and additional metadata
        """
        try:
            # First check if we have documents
            doc_count = self.collection.count()
            if doc_count == 0:
                return {
                    "answer": "The knowledge base is empty. Please ingest documents first.",
                    "sources": [],
                    "metadata": {"status": "empty_db"}
                }

            query_start_time = time.time()

            with st.spinner("Searching knowledge base..."):
                # Query the collection with all metadata
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=self.k_retrieval,
                    include=["documents", "metadatas",
                             "distances", "embeddings"]
                )

                # Get relevant data
                documents = results.get("documents")[0]
                metadatas = results.get("metadatas")[0]
                distances = results.get("distances")[
                    0] if "distances" in results else None

                # Stats for telemetry
                retrieval_time = time.time() - query_start_time

            with st.spinner("Re-ranking documents..."):
                # Re-rank documents
                rerank_start_time = time.time()
                relevant_text, relevant_ids = self.re_rank_documents(
                    query_text, documents, top_k=5)
                rerank_time = time.time() - rerank_start_time

                # Extract source information from the relevant documents with deduplication
                sources = []
                source_metadata = []
                seen_sources = set()

                for idx in relevant_ids:
                    if idx < len(metadatas):
                        metadata = metadatas[idx]
                        if "source" in metadata:
                            source = metadata["source"]
                            if source not in seen_sources:
                                seen_sources.add(source)
                                sources.append(source)

                                # Extract additional metadata for display
                                source_meta = {
                                    "source": source,
                                    "relevance": 1 - (distances[idx] / 2) if distances else None,
                                }
                                # Add other metadata if available
                                for key in ["page", "file_type", "date_processed"]:
                                    if key in metadata:
                                        source_meta[key] = metadata[key]

                                source_metadata.append(source_meta)

            with st.spinner("Generating response..."):
                # Call LLM with context and query
                llm_start_time = time.time()
                answer = self._call_llm(relevant_text, query_text)
                llm_time = time.time() - llm_start_time

            # Total processing time
            total_time = time.time() - query_start_time

            # Performance telemetry for debugging/monitoring
            telemetry = {
                "retrieval_time_ms": int(retrieval_time * 1000),
                "reranking_time_ms": int(rerank_time * 1000),
                "llm_time_ms": int(llm_time * 1000),
                "total_time_ms": int(total_time * 1000),
                "documents_retrieved": len(documents),
                "documents_used": len(relevant_ids),
                "sources_count": len(sources)
            }

            return {
                "answer": answer,
                "sources": sources,
                "source_metadata": source_metadata,
                "telemetry": telemetry
            }

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "metadata": {"status": "error", "error": str(e)}
            }

    def _call_llm(self, context: str, prompt: str) -> str:
        """
        Call the Ollama model with enhanced context handling and streaming support.

        Args:
            context: String containing relevant context
            prompt: String containing the user's question

        Returns:
            String with the generated response
        """
        try:
            # Check if context is too large
            if len(context) > 14000:  # Conservative limit for context
                st.warning(
                    f"Context is very large ({len(context)} chars). Trimming to avoid LLM limits.")
                # Trim context while keeping the structure
                paragraphs = context.split("\n\n")
                trimmed_context = ""
                current_length = 0
                max_context_length = 8000  # Safe limit for most models

                for para in paragraphs:
                    # +2 for "\n\n"
                    if current_length + len(para) + 2 <= max_context_length:
                        trimmed_context += para + "\n\n"
                        current_length += len(para) + 2
                    else:
                        break

                context = trimmed_context
                st.info(f"Trimmed context to {len(context)} chars.")

            # Call the model with enhanced parameters
            response = ollama.chat(
                model=self.llm_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {prompt}",
                    },
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": 0.8,  # Add top_p sampling for better quality
                    "num_predict": 2048,  # Allow longer responses
                }
            )

            return response['message']['content']

        except Exception as e:
            st.error(f"Error calling language model: {str(e)}")
            return f"Error calling language model: {str(e)}"

    def stream_llm_response(self, context: str, prompt: str):
        """
        Stream the LLM response for better user experience.

        Args:
            context: String containing relevant context
            prompt: String containing the user's question

        Yields:
            String chunks of the generated response
        """
        try:
            # Check if context is too large
            if len(context) > 14000:  # Conservative limit for context
                # Trim context to avoid LLM limits
                paragraphs = context.split("\n\n")
                trimmed_context = ""
                current_length = 0
                max_context_length = 8000  # Safe limit for most models

                for para in paragraphs:
                    if current_length + len(para) + 2 <= max_context_length:
                        trimmed_context += para + "\n\n"
                        current_length += len(para) + 2
                    else:
                        break

                context = trimmed_context

            # Call the model with streaming
            response = ollama.chat(
                model=self.llm_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {prompt}",
                    },
                ],
                stream=True,
                options={
                    "temperature": self.temperature,
                    "top_p": 0.8,
                    "num_predict": 2048,
                }
            )

            # Stream the response
            for chunk in response:
                if chunk.get("done") is False:
                    yield chunk["message"]["content"]

        except Exception as e:
            yield f"Error streaming response: {str(e)}"
