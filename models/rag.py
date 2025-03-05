import os
import tempfile
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple, Union

# Document processing
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    PDFMinerLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Vector storage - Using ChromaDB
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_chroma import Chroma

# Cross-encoder for re-ranking
from sentence_transformers import CrossEncoder

# Ollama integration for LLM
import ollama
import time


class StreamlinedRAG:
    def __init__(
        self,
        llm_model_name: str = "llama3",
        embedding_model_name: str = "nomic-embed-text",
        data_dir: str = "data",
        db_dir: str = "vectordb",
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        k_retrieval: int = 10,
        temperature: float = 0.1,
    ):
        """
        Initialize the StreamlinedRAG system.

        Args:
            llm_model_name: Name of the Ollama model to use
            embedding_model_name: Name of the embedding model
            data_dir: Directory containing documents to index
            db_dir: Directory to store the vector database
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            k_retrieval: Number of documents to retrieve initially
            temperature: Temperature setting for the LLM
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

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            keep_separator=True
        )

        # Initialize vector database
        self._init_vectordb()

        # System prompt
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

        Format your response as follows:
        1. Use clear, concise language.
        2. Organize your answer into paragraphs for readability.
        3. Use bullet points or numbered lists where appropriate to break down complex information.
        4. If relevant, include any headings or subheadings to structure your response.
        5. Ensure proper grammar, punctuation, and spelling throughout your answer.

        Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
        """

        # Initialize cross-encoder for re-ranking
        self._init_cross_encoder()

    def _check_ollama_availability(self):
        """Check if Ollama server is available and running"""
        try:
            # Try to list models to check connection
            _ = ollama.list()
            st.sidebar.success("Connected to Ollama server")
        except Exception as e:
            st.sidebar.error(f"Error connecting to Ollama server: {str(e)}")
            st.sidebar.info(
                "Please make sure Ollama is running at: " + self.ollama_api_base
            )

    def _init_vectordb(self):
        """Initialize or load the vector database"""
        # Ensure db_dir exists
        os.makedirs(self.db_dir, exist_ok=True)

        # Flag to track if we're using persistent or in-memory DB
        self.using_persistent_db = True

        try:
            # Create embedding function using Ollama
            embeddings_url = f"{self.ollama_api_base}/api/embeddings"
            self.ollama_ef = OllamaEmbeddingFunction(
                url=embeddings_url,
                model_name=self.embedding_model_name
            )
            st.sidebar.success(
                f"Using {self.embedding_model_name} for embeddings")

            # Initialize persistent ChromaDB client
            chroma_client = chromadb.PersistentClient(path=self.db_dir)

            # Get or create collection
            self.collection = chroma_client.get_or_create_collection(
                name="document_collection",
                embedding_function=self.ollama_ef,
                metadata={"hnsw:space": "cosine"}
            )

            # Create wrapper for LangChain compatibility
            self.vectordb = Chroma(
                client=chroma_client,
                collection_name="document_collection",
                embedding_function=self.ollama_ef
            )

            st.sidebar.success("Connected to vector database")

        except Exception as e:
            st.sidebar.error(f"Error initializing vector database: {str(e)}")
            # Fall back to in-memory DB as last resort
            try:
                client = chromadb.EphemeralClient()
                self.collection = client.create_collection(
                    name="document_collection",
                    embedding_function=self.ollama_ef
                )

                self.vectordb = Chroma(
                    client=client,
                    collection_name="document_collection",
                    embedding_function=self.ollama_ef
                )

                self.using_persistent_db = False
                st.sidebar.info(
                    "Using in-memory database for this session only")
            except Exception as e:
                st.sidebar.error(
                    f"Fatal error: Could not initialize any database: {str(e)}")
                st.stop()

        # Show database stats
        try:
            count = self.collection.count()
            if count > 0:
                st.sidebar.info(f"Database contains {count} document chunks")
            else:
                st.sidebar.info("Database is empty. Please ingest documents.")
        except Exception:
            pass

    def _init_cross_encoder(self):
        """Initialize the cross-encoder model for re-ranking"""
        try:
            # Try to load a more recent cross-encoder model
            cross_encoder_models = [
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
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

    def process_document(self, uploaded_file):
        """
        Process an uploaded document file by converting it to text chunks.

        Args:
            uploaded_file: A Streamlit UploadedFile object

        Returns:
            A list of Document objects containing the chunked text
        """
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # Save uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension)
        temp_file.write(uploaded_file.getbuffer())
        temp_file.close()

        docs = []

        try:
            # Load document based on file extension
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file.name)
                docs = loader.load()
            elif file_extension == ".txt":
                loader = TextLoader(temp_file.name, encoding="utf-8")
                docs = loader.load()
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file.name)
                docs = loader.load()
            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(temp_file.name)
                docs = loader.load()
            else:
                st.warning(f"Unsupported file type: {file_extension}")
                return []

            # Add file metadata
            for doc in docs:
                if not hasattr(doc, "metadata"):
                    doc.metadata = {}
                doc.metadata.update({
                    "source": uploaded_file.name,
                    "file_type": file_extension[1:],
                    "date_processed": time.strftime("%Y-%m-%d %H:%M:%S")
                })

            # Split into chunks
            text_splits = self.text_splitter.split_documents(docs)
            st.success(
                f"Created {len(text_splits)} chunks from {len(docs)} document pages")

            return text_splits

        except Exception as e:
            st.error(
                f"Error processing document {uploaded_file.name}: {str(e)}")
            return []

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def ingest_documents(self, uploaded_files):
        """
        Ingest uploaded documents into the vector database.

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

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Normalize filename for use as ID
            normalized_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            file_names.append(normalized_file_name)

            # Process document
            splits = self.process_document(uploaded_file)
            if splits:
                all_splits.extend(splits)
                st.success(
                    f"✓ Processed {uploaded_file.name}: {len(splits)} chunks")

        if not all_splits:
            st.warning("No valid documents were processed.")
            return False

        # Add documents to vector database
        try:
            # Prepare data for ChromaDB
            documents, metadatas, ids = [], [], []

            for idx, split in enumerate(all_splits):
                file_idx = idx % len(file_names)
                doc_id = f"{file_names[file_idx]}_{idx}"

                documents.append(split.page_content)
                metadatas.append(split.metadata)
                ids.append(doc_id)

            # Add to collection
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            st.success(
                f"✅ Added {len(all_splits)} document chunks to database")
            return True

        except Exception as e:
            st.error(f"Error adding documents to database: {str(e)}")
            return False

    def re_rank_documents(self, query: str, documents: List[str], top_k: int = 3) -> Tuple[str, List[int]]:
        """
        Re-rank documents using the cross-encoder model for better relevance.

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

        try:
            # Create pairs for re-ranking
            pairs = [(query, doc) for doc in documents]

            # Score all query-document pairs
            scores = self.cross_encoder.predict(pairs)

            # Sort by score
            doc_score_pairs = sorted(
                [(score, idx) for idx, score in enumerate(scores)],
                key=lambda x: x[0],
                reverse=True
            )

            # Extract indices of top_k documents
            relevant_text_ids = [idx for _, idx in doc_score_pairs[:top_k]]

            # Combine relevant documents
            relevant_docs = [documents[idx] for idx in relevant_text_ids]
            relevant_text = "\n\n".join(relevant_docs)

            return relevant_text, relevant_text_ids

        except Exception as e:
            st.warning(
                f"Error in re-ranking: {str(e)}. Using top documents without re-ranking.")
            # Fallback to top documents
            return "\n\n".join(documents[:top_k]), list(range(min(top_k, len(documents))))

    def query(self, query_text: str) -> str:
        """
        Process a query through the RAG system.

        Args:
            query_text: The query to process

        Returns:
            String response from the LLM
        """
        try:
            # Check if we have documents
            doc_count = self.collection.count()
            if doc_count == 0:
                return "The knowledge base is empty. Please ingest documents first."

            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=self.k_retrieval,
                include=["documents", "metadatas", "distances"]
            )

            # Get documents
            documents = results.get("documents")[0]

            # Re-rank documents
            relevant_text, _ = self.re_rank_documents(query_text, documents)

            # Call LLM with context and query
            response = self._call_llm(relevant_text, query_text)
            return response

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def query_with_sources(self, query_text: str) -> Dict[str, Any]:
        """
        Process a query and return the answer with source documents.

        Args:
            query_text: The query to process

        Returns:
            Dictionary with answer, sources and metadata
        """
        try:
            # Check if we have documents
            doc_count = self.collection.count()
            if doc_count == 0:
                return {
                    "answer": "The knowledge base is empty. Please ingest documents first.",
                    "sources": [],
                    "source_metadata": []
                }

            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=self.k_retrieval,
                include=["documents", "metadatas", "distances"]
            )

            # Get relevant data
            documents = results.get("documents")[0]
            metadatas = results.get("metadatas")[0]
            distances = results.get("distances")[
                0] if "distances" in results else None

            # Re-rank documents
            relevant_text, relevant_ids = self.re_rank_documents(
                query_text, documents)

            # Extract sources with deduplication
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

                            # Extract metadata for display
                            source_meta = {
                                "source": source,
                                "relevance": 1 - (distances[idx] / 2) if distances else None
                            }

                            # Add other metadata
                            for key in ["page", "file_type", "date_processed"]:
                                if key in metadata:
                                    source_meta[key] = metadata[key]

                            source_metadata.append(source_meta)

            # Call LLM
            answer = self._call_llm(relevant_text, query_text)

            return {
                "answer": answer,
                "sources": sources,
                "source_metadata": source_metadata
            }

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "source_metadata": []
            }

    def _call_llm(self, context: str, prompt: str) -> str:
        """
        Call the Ollama model with context and prompt.

        Args:
            context: String containing relevant context
            prompt: String containing the user's question

        Returns:
            String with the generated response
        """
        try:
            # Check if context is too large (limit to 8000 chars for safety)
            if len(context) > 8000:
                context = context[:8000]

            # Call the model
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
                    "top_p": 0.8,
                    "num_predict": 2048,
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
            if len(context) > 8000:
                context = context[:8000]

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
