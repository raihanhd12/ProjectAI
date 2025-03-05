import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.rag import RAGModel
from utils.config import RAGConfig


def sidebar_component():
    """
    Sidebar component for model selection and configuration
    """
    st.sidebar.title("RAG Question Answer")

    # Model selection section
    st.sidebar.subheader("Model Configuration")

    # LLM model selection using available models from config
    llm_models = RAGConfig.AVAILABLE_LLM_MODELS

    # Initialize session state for model if not present
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = RAGConfig.DEFAULT_LLM_MODEL

    selected_model = st.sidebar.selectbox(
        "Language Model",
        llm_models,
        index=llm_models.index(
            st.session_state.llm_model) if st.session_state.llm_model in llm_models else 0
    )

    # Update model if changed
    if selected_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_model
        # Update model in RAG instance if it exists
        if "rag_model" in st.session_state:
            st.session_state.rag_model.llm_model_name = selected_model
            st.sidebar.success(f"Model updated to {selected_model}")

    # Embedding model selection using available models from config
    embedding_models = RAGConfig.AVAILABLE_EMBEDDING_MODELS

    # Initialize session state for embedding model if not present
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = RAGConfig.DEFAULT_EMBEDDING_MODEL

    selected_embedding = st.sidebar.selectbox(
        "Embedding Model",
        embedding_models,
        index=embedding_models.index(
            st.session_state.embedding_model) if st.session_state.embedding_model in embedding_models else 0
    )

    # Update embedding model if changed
    if selected_embedding != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_embedding
        # Reset RAG model instance to use new embedding model
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
        "Chunk Size",
        min_value=100,
        max_value=1000,
        value=RAGConfig.DEFAULT_CHUNK_SIZE,
        step=50
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=300,
        value=RAGConfig.DEFAULT_CHUNK_OVERLAP,
        step=10
    )

    # Apply chunking parameters if they've changed
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
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            )
            st.sidebar.success("Chunking parameters updated")

    # Database information
    st.sidebar.subheader("Database Information")
    st.sidebar.info(f"Using database at: {RAGConfig.VECTORDB_PATH}")
