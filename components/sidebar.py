import streamlit as st
from utils.chat_history import ChatHistory
from models.rag import RAGModel  # Import the RAG model


def sidebar_menu(app_modes, DB_DIR, DATA_DIR):
    """
    Display the sidebar with app navigation and chat history.

    Args:
        app_modes: List of available application modes
        DB_DIR: Directory for vector database
        DATA_DIR: Directory for data files
    """
    # Logo and title
    st.sidebar.image(
        "https://via.placeholder.com/100x100.png?text=AI", width=100)
    st.sidebar.title("AI Assistant")
    st.sidebar.markdown("---")

    # Navigation menu
    st.sidebar.subheader("Tools")

    # Create buttons for each app mode with icons
    icons = {
        "Home": "🏠",
        "RAG Chat": "💬",
        "Summarizer": "📝",
        # Add more tools with icons here
    }

    for mode in app_modes:
        icon = icons.get(mode, "🔧")
        if st.sidebar.button(f"{icon} {mode}", use_container_width=True,
                             key=f"nav_{mode}",
                             type="primary" if st.session_state.app_mode == mode else "secondary"):
            # If changing to a new mode, create a new conversation for that mode
            if mode != st.session_state.app_mode:
                if mode != "Home":  # Don't create conversation for Home
                    new_id = ChatHistory.add_conversation(app_mode=mode)
                    st.session_state.current_conversation_id = new_id
                st.session_state.app_mode = mode
                st.session_state.show_settings = False  # Hide settings when switching
                st.rerun()

    st.sidebar.markdown("---")

    # Display model info for reference
    if st.session_state.app_mode != "Home":
        st.sidebar.caption("Currently using:")
        st.sidebar.caption(f"LLM: {st.session_state.llm_model}")
        st.sidebar.caption(
            f"Embeddings: {st.session_state.embedding_model.split('/')[-1]}")

    # Only show chat history for chat-based tools
    if st.session_state.app_mode == "RAG Chat":
        display_chat_history()


def display_chat_history():
    """Display chat history for the current application mode"""
    st.sidebar.markdown("---")

    # Chat history section
    st.sidebar.subheader("💾 Chat History")

    # New conversation button
    if st.sidebar.button("➕ New Conversation", use_container_width=True, type="primary"):
        new_id = ChatHistory.add_conversation(
            app_mode=st.session_state.app_mode)
        st.session_state.current_conversation_id = new_id
        st.rerun()

    # Display existing conversations filtered by current app mode
    history = ChatHistory.load_history()

    # Filter conversations by current app mode
    filtered_conversations = {
        conv_id: data for conv_id, data in history.items()
        if data.get("app_mode") == st.session_state.app_mode
    }

    # Sort conversations by creation time (newest first)
    sorted_conversations = sorted(
        filtered_conversations.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True
    )

    if sorted_conversations:
        for conv_id, data in sorted_conversations:
            col1, col2 = st.sidebar.columns([4, 1])

            with col1:
                # Highlight the current conversation
                is_current = st.session_state.current_conversation_id == conv_id
                button_label = data["title"]

                if is_current:
                    button_label = f"➤ {button_label}"

                # Truncate very long titles
                if len(button_label) > 25:
                    button_label = button_label[:22] + "..."

                if st.button(button_label, key=f"btn_{conv_id}"):
                    st.session_state.current_conversation_id = conv_id
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    ChatHistory.delete_conversation(conv_id)
                    if st.session_state.current_conversation_id == conv_id:
                        # After deleting, select another conversation
                        if len(sorted_conversations) > 1:
                            # Find next conversation to select
                            next_conversations = [
                                cid for cid, _ in sorted_conversations if cid != conv_id]
                            if next_conversations:
                                st.session_state.current_conversation_id = next_conversations[0]
                        else:
                            # Create a new conversation if this was the last one
                            new_id = ChatHistory.add_conversation(
                                app_mode=st.session_state.app_mode)
                            st.session_state.current_conversation_id = new_id
                    st.rerun()
    else:
        st.sidebar.info("No conversations yet.")

    st.sidebar.markdown("---")

    # Model selection toggle
    if st.sidebar.button("⚙️ Settings", use_container_width=True):
        st.session_state.show_settings = not st.session_state.show_settings
        st.rerun()
