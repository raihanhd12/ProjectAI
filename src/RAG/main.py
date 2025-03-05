import os
import tempfile
import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Define the system prompt for LLM
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# Configuration parameters
AVAILABLE_LLM_MODELS = ["llama3", "gpt-3.5-turbo", "gpt-4"]
DEFAULT_LLM_MODEL = "llama3"
AVAILABLE_EMBEDDING_MODELS = [
    "nomic-embed-text:latest", "openai-embed-text:latest"]
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
VECTORDB_PATH = "./vector-db"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 100

# RAG Model Class


class RAGModel:
    def __init__(self, llm_model_name=DEFAULT_LLM_MODEL, embedding_model_name=DEFAULT_EMBEDDING_MODEL, db_dir=VECTORDB_PATH, data_dir=None, collection_name="rag_app", chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, k_retrieval=10):
        """Initialize RAG model with specified parameters"""
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.db_dir = db_dir
        self.data_dir = data_dir or "./data"
        self.collection_name = collection_name
        self.k_retrieval = k_retrieval
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Define system prompt
        self.system_prompt = system_prompt

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        # Initialize cross-encoder model
        self.encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2")

    def get_vector_collection(self):
        """Gets or creates a ChromaDB collection for vector storage."""
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings", model_name=self.embedding_model_name)
        chroma_client = chromadb.PersistentClient(path=self.db_dir)
        return chroma_client.get_or_create_collection(name=self.collection_name, embedding_function=ollama_ef, metadata={"hnsw:space": "cosine"})

    def process_document(self, uploaded_file: UploadedFile) -> list[Document]:
        """Processes an uploaded PDF file by converting it to text chunks."""
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile(
            "wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)  # Delete temp file
        return self.text_splitter.split_documents(docs)

    def add_to_vector_collection(self, all_splits: list[Document], file_name: str):
        """Adds document splits to a vector collection for semantic search."""
        collection = self.get_vector_collection()
        documents, metadatas, ids = [], [], []
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        st.success("Data added to the vector store!")

    def query_collection(self, prompt: str, n_results: int = 10):
        """Queries the vector collection with a given prompt."""
        collection = self.get_vector_collection()
        return collection.query(query_texts=[prompt], n_results=n_results)

    def re_rank_documents(self, query: str, documents: list[str], top_k: int = 3) -> tuple[str, list[int]]:
        """Re-ranks documents using a cross-encoder model for more accurate relevance scoring."""
        relevant_text = ""
        relevant_text_ids = []
        ranks = self.encoder_model.rank(query, documents, top_k=top_k)
        for rank in ranks:
            relevant_text += documents[rank["corpus_id"]]
            relevant_text_ids.append(rank["corpus_id"])
        return relevant_text, relevant_text_ids

    def call_llm(self, context: str, prompt: str):
        """Calls the language model with context and prompt to generate a response."""
        response = ollama.chat(
            model=self.llm_model_name,
            stream=True,
            messages=[{"role": "system", "content": self.system_prompt}, {
                "role": "user", "content": f"Context: {context}, Question: {prompt}"}]
        )
        for chunk in response:
            if chunk["done"] is False:
                yield chunk["message"]["content"]
            else:
                break

# Sidebar Component


def sidebar_component():
    """Sidebar component for RAG model."""
    st.sidebar.title("RAG Question Answer")

    # Model selection section
    st.sidebar.subheader("Model Configuration")

    # LLM model selection using available models
    llm_models = AVAILABLE_LLM_MODELS
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = DEFAULT_LLM_MODEL

    selected_model = st.sidebar.selectbox(
        "Language Model",
        llm_models,
        index=llm_models.index(
            st.session_state.llm_model) if st.session_state.llm_model in llm_models else 0
    )

    # Update model if changed
    if selected_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_model
        if "rag_model" in st.session_state:
            st.session_state.rag_model.llm_model_name = selected_model
            st.sidebar.success(f"Model updated to {selected_model}")

    # Embedding model selection
    embedding_models = AVAILABLE_EMBEDDING_MODELS
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL

    selected_embedding = st.sidebar.selectbox(
        "Embedding Model",
        embedding_models,
        index=embedding_models.index(
            st.session_state.embedding_model) if st.session_state.embedding_model in embedding_models else 0
    )

    # Update embedding model if changed
    if selected_embedding != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_embedding
        if "rag_model" in st.session_state:
            try:
                st.session_state.rag_model = RAGModel(
                    llm_model_name=st.session_state.llm_model,
                    embedding_model_name=selected_embedding
                )
                st.sidebar.success(
                    f"Embedding model updated to {selected_embedding}")
            except Exception as e:
                st.sidebar.error(f"Error updating embedding model: {str(e)}")

    # Chunking parameters
    st.sidebar.subheader("Chunking Configuration")
    chunk_size = st.sidebar.slider(
        "Chunk Size", min_value=100, max_value=1000, value=DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap", min_value=0, max_value=300, value=DEFAULT_CHUNK_OVERLAP, step=10)

    # Apply chunking parameters if changed
    if "chunk_size" not in st.session_state or "chunk_overlap" not in st.session_state:
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
    elif st.session_state.chunk_size != chunk_size or st.session_state.chunk_overlap != chunk_overlap:
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap
        if "rag_model" in st.session_state:
            st.session_state.rag_model.chunk_size = chunk_size
            st.session_state.rag_model.chunk_overlap = chunk_overlap
            st.session_state.rag_model.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.sidebar.success("Chunking parameters updated")

    # Database info
    st.sidebar.subheader("Database Information")
    st.sidebar.info(f"Using database at: {VECTORDB_PATH}")

# RAG Chat Component


def rag_chat_component():
    """Component for RAG-based question answering with document uploads."""
    # Initialize RAG model
    if "rag_model" not in st.session_state:
        st.session_state.rag_model = RAGModel()

    # Document upload and processing
    uploaded_file = st.sidebar.file_uploader(
        "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False)
    process = st.button("⚡️ Process")

    if uploaded_file and process:
        with st.spinner("Processing document..."):
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"}))
            all_splits = st.session_state.rag_model.process_document(
                uploaded_file)
            st.session_state.rag_model.add_to_vector_collection(
                all_splits, normalize_uploaded_file_name)

    # Question and answer area
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")

    if ask and prompt:
        with st.spinner("Searching for answers..."):
            results = st.session_state.rag_model.query_collection(prompt)
            context = results.get("documents")[0]
            relevant_text, relevant_text_ids = st.session_state.rag_model.re_rank_documents(
                prompt, context)
            response = st.session_state.rag_model.call_llm(
                context=relevant_text, prompt=prompt)
            st.write_stream(response)

            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document ids"):
                st.write(relevant_text_ids)
                st.write(relevant_text)

# Main entry point for RAG tool


def rag():
    """Main function to run the RAG tool."""
    sidebar_component()
    rag_chat_component()
