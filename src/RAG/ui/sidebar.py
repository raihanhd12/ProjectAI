"""
Sidebar UI components for the RAG module.
"""
import datetime
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config
from .. import db
from .. import models


def navigation_sidebar():
    """
    Render navigation sidebar.

    Returns:
        None
    """
    with st.sidebar:
        st.title("Navigation")
        st.divider()

        # Set default view if not already set
        if "rag_view" not in st.session_state:
            st.session_state.rag_view = "documents"

        # Navigation buttons
        if st.button("📄 Document Management",
                     type="primary" if st.session_state.rag_view == "documents" else "secondary",
                     use_container_width=True):
            st.session_state.rag_view = "documents"
            st.rerun()

        if st.button("💬 Chat with Documents",
                     type="primary" if st.session_state.rag_view == "chat" else "secondary",
                     use_container_width=True):
            st.session_state.rag_view = "chat"
            st.rerun()

        st.divider()


def configuration_sidebar():
    """
    Render configuration sidebar.

    Returns:
        None
    """
    # Check current view from session state
    current_view = st.session_state.get("rag_view", "documents")

    # Display Recent Chats section only when in Chat view
    if current_view == "chat":
        display_recent_chats()
        st.sidebar.divider()

    # RAG configuration section
    st.sidebar.title("RAG Configuration")

    # Model selection section - AS EXPANDER
    with st.sidebar.expander("Model Selection", expanded=True):
        # LLM model selection using available models
        llm_models = config.AVAILABLE_LLM_MODELS
        if "llm_model" not in st.session_state:
            st.session_state.llm_model = config.DEFAULT_LLM_MODEL

        selected_model = st.selectbox(
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
                st.success(f"Model updated to {selected_model}")

        # Embedding model selection
        embedding_models = config.AVAILABLE_EMBEDDING_MODELS
        if "embedding_model" not in st.session_state:
            st.session_state.embedding_model = config.DEFAULT_EMBEDDING_MODEL

        selected_embedding = st.selectbox(
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
                    st.session_state.rag_model = models.RAGModel(
                        llm_model_name=st.session_state.llm_model,
                        embedding_model_name=selected_embedding
                    )
                    st.success(
                        f"Embedding model updated to {selected_embedding}")
                except Exception as e:
                    st.error(f"Error updating embedding model: {str(e)}")

    # Model Settings - AS EXPANDER
    with st.sidebar.expander("Model Settings", expanded=False):
        # Chunking parameters
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=config.DEFAULT_CHUNK_SIZE,
            step=50,
            help="Size of text chunks for processing"
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=300,
            value=config.DEFAULT_CHUNK_OVERLAP,
            step=10,
            help="Overlap between chunks"
        )

        # Other parameters
        st.slider(
            "Context Length",
            min_value=1024,
            max_value=8192,
            value=4096,
            step=1024,
            help="Maximum context length"
        )

        top_k = st.slider(
            "Top K Retrieval",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Number of documents to retrieve"
        )

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
                st.success("Chunking parameters updated")

    # Advanced Options - AS EXPANDER
    with st.sidebar.expander("Advanced Options", expanded=False):
        st.toggle("Keep Model in Memory", value=True)
        st.toggle("Use Re-ranking", value=True)


def display_recent_chats():
    """
    Display recent chat sessions in the sidebar.

    Returns:
        None
    """
    st.sidebar.title("Recent Chats")

    # Get recent chat sessions
    recent_sessions = db.get_recent_chat_sessions(5)  # Show last 5 sessions

    if recent_sessions:
        # Handle confirmation for deletion
        if "delete_confirm" in st.session_state and st.session_state.delete_confirm:
            session_to_delete = st.session_state.delete_confirm
            # Find the question for this session
            session_info = next(
                (s for s in recent_sessions if s[0] == session_to_delete), None)
            if session_info:
                question = session_info[3]
                preview = question if len(
                    question) <= 30 else question[:27] + "..."
                st.sidebar.warning(f"Delete chat: \"{preview}\"?")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("✓ Yes", key="confirm_yes"):
                        success = db.delete_chat_session(session_to_delete)
                        if success:
                            st.sidebar.success("Chat deleted!")
                            if st.session_state.session_id == session_to_delete:
                                # Current session was deleted, create a new one
                                st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        # Clear confirmation state
                        st.session_state.delete_confirm = None
                        st.rerun()
                with col2:
                    if st.button("✗ No", key="confirm_no"):
                        # Clear confirmation state
                        st.session_state.delete_confirm = None
                        st.rerun()
            st.sidebar.divider()

        # List all sessions with delete buttons
        for session_id, start_time, last_time, first_question in recent_sessions:
            # Format timestamps
            from ..utils import format_timestamp, truncate_text
            formatted_time = format_timestamp(last_time)
            preview = truncate_text(first_question, 40)

            # Create two columns for session button and delete button
            col1, col2 = st.sidebar.columns([5, 1])

            # Session button in first column
            with col1:
                if st.button(
                    f"🗨️ {preview}\n📅 {formatted_time}",
                    key=f"history_{session_id}",
                    use_container_width=True
                ):
                    # Switch to this session and reload
                    st.session_state.session_id = session_id
                    st.rerun()

            # Delete button in second column
            with col2:
                st.write("")  # Add some spacing
                if st.button("🗑️", key=f"delete_{session_id}"):
                    st.session_state.delete_confirm = session_id
                    st.rerun()
    else:
        st.sidebar.info("No recent chats found")

    # Add a button to start a new chat session
    if st.sidebar.button("Start New Chat Session", type="primary", use_container_width=True):
        # Generate a new session ID
        st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        st.rerun()


def display_debug_info():
    """
    Display debug information in the sidebar.

    Returns:
        None
    """
    st.sidebar.title("Debug Information")
    st.sidebar.info(f"Project root: {config.project_root}")
    st.sidebar.info(f"Vector DB path: {config.VECTORDB_PATH}")
    st.sidebar.info(f"SQLite DB path: {config.DB_PATH}")
