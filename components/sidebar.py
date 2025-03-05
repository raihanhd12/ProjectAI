import streamlit as st
from utils.chat_history import ChatHistory

# Add necessary imports
from langchain_ollama import OllamaLLM
from models.rag import ImprovedRAG  # Import LocalRAG directly


def display_sidebar(app_modes, DB_DIR, DATA_DIR):
    """
    Display the sidebar with app controls and chat history.

    Args:
        app_modes: List of available application modes
        DB_DIR: Directory for vector database
        DATA_DIR: Directory for data files
    """
    st.sidebar.title("AI Assistant")

    # Initialize embedding_model in session state if not present
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Initialize llm_model in session state if not present
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "qwen2.5"

    # App mode selection
    selected_mode = st.sidebar.selectbox(
        "Select Application",
        app_modes,
        index=app_modes.index(st.session_state.app_mode)
    )

    if selected_mode != st.session_state.app_mode:
        # Create a new conversation for the new app mode
        new_id = ChatHistory.add_conversation(app_mode=selected_mode)
        st.session_state.current_conversation_id = new_id
        st.session_state.app_mode = selected_mode
        st.rerun()

    st.sidebar.divider()

    # Chat history section
    st.sidebar.subheader("Chat History")

    # New conversation button
    if st.sidebar.button("New Conversation"):
        new_id = ChatHistory.add_conversation(
            app_mode=st.session_state.app_mode)
        st.session_state.current_conversation_id = new_id
        st.rerun()

    # Display existing conversations FILTERED BY CURRENT APP MODE
    history = ChatHistory.load_history()

    # Filter conversations by current app mode
    filtered_conversations = {
        conv_id: data for conv_id, data in history.items()
        if data.get("app_mode") == st.session_state.app_mode
    }

    conversation_ids = list(filtered_conversations.keys())

    if conversation_ids:
        for conv_id in conversation_ids:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                # Highlight the current conversation
                is_current = st.session_state.current_conversation_id == conv_id
                button_label = filtered_conversations[conv_id]["title"]

                if is_current:
                    button_label = f"➤ {button_label}"

                if st.button(button_label, key=f"btn_{conv_id}"):
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    ChatHistory.delete_conversation(conv_id)
                    if st.session_state.current_conversation_id == conv_id:
                        if len(conversation_ids) > 1:
                            # Set current to another conversation
                            other_ids = [
                                cid for cid in conversation_ids if cid != conv_id]
                            st.session_state.current_conversation_id = other_ids[0]
                        else:
                            # Create a new conversation
                            new_id = ChatHistory.add_conversation(
                                app_mode=st.session_state.app_mode)
                            st.session_state.current_conversation_id = new_id
                    st.rerun()
    else:
        st.sidebar.info("No conversations for this tool.")

    # Display model information
    st.sidebar.divider()
    st.sidebar.caption("Model Information")

    if hasattr(st.session_state, "rag_system"):
        st.sidebar.caption(f"LLM: {st.session_state.rag_system.model_name}")
        st.sidebar.caption(
            f"Embeddings: {st.session_state.rag_system.embedding_model_name.split('/')[-1]}")

    st.sidebar.divider()

    # Model selection section
    st.sidebar.subheader("Models Configuration")

    # Embedding model selection - FIX: Include all models in the list
# Embedding model selection
    embedding_models = [
        "sentence-transformers/all-MiniLM-L6-v2",  # HuggingFace model
        "BAAI/bge-small-en-v1.5",                  # HuggingFace model
        "intfloat/e5-small-v2",                    # HuggingFace model
        # Ollama model (no sentence-transformers/ prefix)
        "nomic-embed-text"
    ]

    selected_embedding = st.sidebar.selectbox(
        "Embedding Model",
        embedding_models,
        index=embedding_models.index(
            st.session_state.embedding_model) if st.session_state.embedding_model in embedding_models else 0
    )

    if selected_embedding != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_embedding
        if "rag_system" in st.session_state:
            # Reinitialize with new embedding model
            try:
                st.session_state.rag_system = ImprovedRAG(  # Changed from LocalRAG
                    db_dir=DB_DIR,
                    data_dir=DATA_DIR,
                    embedding_model_name=selected_embedding,
                    llm_model_name=st.session_state.llm_model  # Changed from model_name
                )
                st.sidebar.success(
                    f"Embedding model updated to {selected_embedding}")
            except Exception as e:
                st.sidebar.error(f"Error updating embedding model: {str(e)}")
        st.rerun()

    # LLM model selection
    llm_models = [
        "qwen2.5",
        "qwen2.5-14b",
        "llama3",
        "mistral"
    ]

    selected_llm = st.sidebar.selectbox(
        "LLM Model",
        llm_models,
        index=llm_models.index(
            st.session_state.llm_model) if st.session_state.llm_model in llm_models else 0
    )

    if selected_llm != st.session_state.llm_model:
        st.session_state.llm_model = selected_llm
        if "rag_system" in st.session_state:
            # Update LLM model
            try:
                # No need to initialize OllamaLLM separately, just update model_name
                st.session_state.rag_system.llm_model_name = selected_llm
                st.sidebar.success(f"LLM model updated to {selected_llm}")
            except Exception as e:
                st.sidebar.error(f"Error updating LLM model: {str(e)}")
        st.rerun()
