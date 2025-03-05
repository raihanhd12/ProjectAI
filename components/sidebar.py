import streamlit as st
from models.rag import RAGModel


def sidebar_component():
    """
    Sidebar component for model selection and configuration
    """
    st.sidebar.title("RAG Question Answer")

    # Model selection section
    st.sidebar.subheader("Model Configuration")

    # LLM model selection
    llm_models = [
        "llama3",
        "qwen2.5"
    ]

    # Initialize session state for model if not present
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = llm_models[0]

    selected_model = st.sidebar.selectbox(
        "Language Model",
        llm_models,
        index=llm_models.index(st.session_state.llm_model)
    )

    # Update model if changed
    if selected_model != st.session_state.llm_model:
        st.session_state.llm_model = selected_model
        # Update model in RAG instance if it exists
        if "rag_model" in st.session_state:
            st.session_state.rag_model.model_name = selected_model
            st.sidebar.success(f"Model updated to {selected_model}")

    # Embedding model selection
    embedding_models = [
        "nomic-embed-text:latest",
        "nomic-embed-text",
        "all-MiniLM-L6-v2"
    ]

    # Initialize session state for embedding model if not present
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = embedding_models[0]

    selected_embedding = st.sidebar.selectbox(
        "Embedding Model",
        embedding_models,
        index=embedding_models.index(st.session_state.embedding_model)
    )

    # Update embedding model if changed
    if selected_embedding != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_embedding
        # Reset RAG model instance to use new embedding model
        if "rag_model" in st.session_state:
            try:
                st.session_state.rag_model = RAGModel(
                    model_name=st.session_state.llm_model,
                    embedding_model=selected_embedding
                )
                st.sidebar.success(
                    f"Embedding model updated to {selected_embedding}")
            except Exception as e:
                st.sidebar.error(f"Error updating embedding model: {str(e)}")

    # Chunking parameters
    st.sidebar.subheader("Chunking Configuration")
    chunk_size = st.sidebar.slider(
        "Chunk Size", min_value=100, max_value=1000, value=400, step=50)
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap", min_value=0, max_value=300, value=100, step=10)

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

    # Database configuration
    st.sidebar.subheader("Database Configuration")
    db_path = st.sidebar.text_input("Database Path", value="./demo-rag-chroma")

    # Apply database path if changed
    if "db_path" not in st.session_state:
        st.session_state.db_path = db_path
    elif st.session_state.db_path != db_path:
        st.session_state.db_path = db_path
        if "rag_model" in st.session_state:
            st.session_state.rag_model.db_path = db_path
            st.sidebar.success("Database path updated")
            st.sidebar.warning(
                "You may need to restart the application for this change to take effect")
