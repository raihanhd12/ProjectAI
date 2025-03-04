import os
import streamlit as st
from typing import List, Dict, Any, Optional

# Document processing
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and vector storage - Fixed imports
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import chromadb
from langchain_chroma import Chroma  # Updated import

# LLM integration
from langchain_ollama import OllamaLLM  # Updated import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


class LocalRAG:
    def __init__(
        self,
        model_name: str = "qwen2.5",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        data_dir: str = "data",
        db_dir: str = "vectordb",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the LocalRAG system.

        Args:
            model_name: Name of the Ollama model to use
            embedding_model_name: Name of the HuggingFace embedding model
            data_dir: Directory containing documents to index
            db_dir: Directory to store the vector database
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # In the __init__ method of LocalRAG class
        # Initialize embeddings based on model name
        if embedding_model_name == "nomic-embed-text":
            from langchain_ollama import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(model=embedding_model_name)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cuda" if os.environ.get(
                    "CUDA_VISIBLE_DEVICES") else "cpu"},
                cache_folder="./models/embeddings_cache",
                encode_kwargs={"normalize_embeddings": True}
            )

        # Initialize LLM - Updated
        self.llm = OllamaLLM(model=model_name)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Initialize or load vector database
        self._init_vectordb()

        # Create RAG chain
        self._create_rag_chain()

    def _init_vectordb(self):
        """Initialize or load the vector database."""
        # Ensure db_dir exists
        os.makedirs(self.db_dir, exist_ok=True)

        # Flag to track if we're using persistent or in-memory DB
        self.using_persistent_db = True

        try:
            # Method 1: Direct Chroma initialization with persist_directory
            self.vectordb = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings,
                collection_name="document_collection"
            )
            st.sidebar.success("Connected to persistent vector database.")

        except Exception as e1:
            st.sidebar.warning(
                f"Could not initialize persistent database (Error: {str(e1)})")
            st.sidebar.info("Trying alternative initialization method...")

            try:
                # Method 2: Using client and explicit collection
                client = chromadb.PersistentClient(path=self.db_dir)

                # Get or create collection
                collection_name = "document_collection"
                try:
                    collection = client.get_collection(name=collection_name)
                    st.sidebar.success(
                        f"Loaded existing collection: {collection_name}")
                except:
                    collection = client.create_collection(name=collection_name)
                    st.sidebar.success(
                        f"Created new collection: {collection_name}")

                # Initialize with client and collection
                self.vectordb = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.db_dir
                )

            except Exception as e2:
                st.sidebar.error(
                    f"Error initializing persistent database: {str(e2)}")
                st.sidebar.warning(
                    "Falling back to in-memory database (changes won't persist).")

                # Fallback to in-memory
                client = chromadb.EphemeralClient()
                collection_name = "document_collection"
                try:
                    collection = client.create_collection(name=collection_name)
                except:
                    collection = client.get_collection(name=collection_name)

                self.vectordb = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings
                )
                self.using_persistent_db = False

        # Create retriever - do this after database is initialized
        self.retriever = self.vectordb.as_retriever(
            search_kwargs={"k": 5}
        )

        # Show document count if available
        try:
            if hasattr(self.vectordb, "_collection") and self.vectordb._collection is not None:
                count = self.vectordb._collection.count()
                if count > 0:
                    st.sidebar.info(f"Database contains {count} documents.")
                else:
                    st.sidebar.info(
                        "Database is empty. Please ingest documents.")
        except Exception as e:
            pass

    def _create_rag_chain(self):
        """Create an improved RAG chain with better prompting and retrieval."""
        # Enhanced prompt template
        template = """
        You are a helpful AI assistant with expertise in understanding and explaining documents.
        Use ONLY the following pieces of retrieved context to answer the user's question.
        If you don't know the answer or the information is not present in the context, say "I don't have enough information to answer that question."
        Do not use any prior knowledge.
        
        Context:
        {context}
        
        User Question: {question}
        
        When referencing information, specify which part of the context you're using.
        Provide a comprehensive answer that directly addresses the question.
        If the context contains information in a language other than English, respond in that same language.
        """

        prompt = PromptTemplate.from_template(template)

        # Create retriever with metadata filtering options
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7,  # Only return relevant results
                "fetch_k": 20  # Fetch more candidates for MMR
            }
        )

        # Create the RAG chain with better error handling
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ingest_documents(self, uploaded_files):
        """Ingest uploaded documents into the vector database."""
        docs = []

        # Process uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(self.data_dir, uploaded_file.name)

            # Ensure data directory exists
            os.makedirs(self.data_dir, exist_ok=True)

            # Save the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the file based on its extension
            file_ext = os.path.splitext(file_path)[1].lower()

            try:
                if file_ext == ".pdf":
                    try:
                        # First try with extract_images=False to avoid Pillow dependency
                        loader = PyPDFLoader(file_path, extract_images=False)
                        docs.extend(loader.load())
                        st.sidebar.success(f"Loaded PDF: {uploaded_file.name}")
                    except Exception as pdf_error:
                        st.sidebar.warning(
                            f"Warning: Could not extract text from PDF with standard loader: {str(pdf_error)}")
                        st.sidebar.info(
                            "Trying alternative PDF loading approach...")

                        # Alternative: Try using direct text extraction
                        try:
                            from langchain_community.document_loaders import PDFMinerLoader
                            loader = PDFMinerLoader(file_path)
                            docs.extend(loader.load())
                            st.sidebar.success(
                                f"Loaded PDF with PDFMiner: {uploaded_file.name}")
                        except Exception as alt_error:
                            st.sidebar.error(
                                f"Error loading PDF with alternative method: {str(alt_error)}")
                elif file_ext == ".txt":
                    loader = TextLoader(file_path)
                    docs.extend(loader.load())
                    st.sidebar.success(
                        f"Loaded text file: {uploaded_file.name}")
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                    docs.extend(loader.load())
                    st.sidebar.success(
                        f"Loaded Word document: {uploaded_file.name}")
                elif file_ext == ".md":
                    loader = UnstructuredMarkdownLoader(file_path)
                    docs.extend(loader.load())
                    st.sidebar.success(
                        f"Loaded markdown file: {uploaded_file.name}")
                else:
                    st.sidebar.warning(f"Unsupported file type: {file_ext}")
            except Exception as e:
                st.sidebar.error(
                    f"Error loading {uploaded_file.name}: {str(e)}")

        if not docs:
            st.sidebar.warning("No valid documents to ingest.")
            return False

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(docs)
        st.sidebar.info(
            f"Created {len(chunks)} chunks from {len(docs)} documents.")

        # Add documents to the vector database
        try:
            self.vectordb.add_documents(chunks)
            st.sidebar.success(
                f"Added {len(chunks)} document chunks to database.")

            # Attempt to persist only if we're using a persistent database
            if self.using_persistent_db:
                try:
                    # Try the persist method directly
                    self.vectordb.persist()
                    st.sidebar.success(
                        "Vector database persisted successfully.")
                except Exception as persist_error:
                    st.sidebar.warning(
                        f"Note: Could not persist database automatically: {str(persist_error)}")
                    st.sidebar.info(
                        "Your data is still available for this session.")
            else:
                st.sidebar.warning(
                    "Using in-memory database (changes won't persist between sessions).")

            return True

        except Exception as e:
            st.sidebar.error(f"Error adding documents to database: {str(e)}")
            return False

    def query(self, query_text: str) -> str:
        """Process a query through the RAG system."""
        try:
            # First check if we have documents
            has_docs = False
            try:
                if hasattr(self.vectordb, "_collection"):
                    count = self.vectordb._collection.count()
                    has_docs = count > 0
            except:
                pass

            if not has_docs:
                return "The knowledge base is empty. Please ingest documents first."

            return self.rag_chain.invoke(query_text)
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def query_with_sources(self, query_text: str) -> Dict[str, Any]:
        """Process a query and return the answer with source documents."""
        try:
            # First check if we have documents
            has_docs = False
            try:
                if hasattr(self.vectordb, "_collection"):
                    count = self.vectordb._collection.count()
                    has_docs = count > 0
            except:
                pass

            if not has_docs:
                return {"answer": "The knowledge base is empty. Please ingest documents first.", "sources": []}

            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query_text)

            # Use the RAG chain for the answer
            answer = self.rag_chain.invoke(query_text)

            # Extract source information
            sources = []
            for doc in docs:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.append(doc.metadata["source"])
                else:
                    sources.append("Unknown source")

            return {
                "answer": answer,
                "sources": list(set(sources))  # Deduplicate sources
            }
        except Exception as e:
            return {"answer": f"Error processing query: {str(e)}", "sources": []}
