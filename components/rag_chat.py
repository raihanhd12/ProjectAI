import streamlit as st
import os
from streamlined_rag import StreamlinedRAG
from utils.chat_history import ChatHistory

# Constants
DB_DIR = "vectordb"
DATA_DIR = "data"

# Ensure directories exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="💬",
        layout="wide"
    )

    # Initialize session state
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "RAG Chat"

    if "current_conversation_id" not in st.session_state:
        # Initialize with a new conversation
        new_id = ChatHistory.add_conversation(
            app_mode=st.session_state.app_mode)
        st.session_state.current_conversation_id = new_id

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "nomic-embed-text"

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "llama3"

    # Initialize RAG system if not present
    if "rag_system" not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = StreamlinedRAG(
                llm_model_name=st.session_state.llm_model,
                embedding_model_name=st.session_state.embedding_model,
                db_dir=DB_DIR,
                data_dir=DATA_DIR
            )

    # Display sidebar
    display_sidebar()

    # Display chat interface
    rag_chat_app()


def display_sidebar():
    """Display the sidebar with app controls and chat history"""
    st.sidebar.title("AI Assistant")

    # Chat history section
    st.sidebar.subheader("Chat History")

    # New conversation button
    if st.sidebar.button("New Conversation"):
        new_id = ChatHistory.add_conversation(
            app_mode=st.session_state.app_mode)
        st.session_state.current_conversation_id = new_id
        st.rerun()

    # Display existing conversations
    history = ChatHistory.load_history()
    conversation_ids = list(history.keys())

    if conversation_ids:
        for conv_id in conversation_ids:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                # Highlight the current conversation
                is_current = st.session_state.current_conversation_id == conv_id
                button_label = history[conv_id]["title"]

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
        st.sidebar.info("No conversations yet.")

    # Display model information
    st.sidebar.divider()
    st.sidebar.caption("Model Information")

    st.sidebar.caption(f"LLM: {st.session_state.rag_system.llm_model_name}")
    st.sidebar.caption(
        f"Embeddings: {st.session_state.rag_system.embedding_model_name}")

    st.sidebar.divider()

    # Model selection section
    st.sidebar.subheader("Models Configuration")

    # Embedding model selection
    embedding_models = [
        "nomic-embed-text",
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-small-v2"
    ]

    selected_embedding = st.sidebar.selectbox(
        "Embedding Model",
        embedding_models,
        index=embedding_models.index(st.session_state.embedding_model)
        if st.session_state.embedding_model in embedding_models else 0
    )

    # Update embedding model if changed
    if selected_embedding != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_embedding
        st.session_state.rag_system.embedding_model_name = selected_embedding
        st.sidebar.success(f"Embedding model updated to {selected_embedding}")
        st.rerun()

    # LLM model selection
    llm_models = [
        "llama3",
        "llama3.2:3b",
        "qwen2.5",
        "mistral"
    ]

    selected_llm = st.sidebar.selectbox(
        "LLM Model",
        llm_models,
        index=llm_models.index(st.session_state.llm_model)
        if st.session_state.llm_model in llm_models else 0
    )

    # Update LLM model if changed
    if selected_llm != st.session_state.llm_model:
        st.session_state.llm_model = selected_llm
        st.session_state.rag_system.llm_model_name = selected_llm
        st.sidebar.success(f"LLM model updated to {selected_llm}")
        st.rerun()


def rag_chat_app():
    """RAG Chat application interface with streaming responses"""
    st.title("RAG Chat Assistant")

    # Document upload section
    with st.expander("Upload Documents to Knowledge Base", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX, MD)",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "md"]
        )

        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                st.session_state.rag_system.ingest_documents(uploaded_files)

    # Display conversation
    st.subheader("Conversation")

    messages = ChatHistory.get_messages(
        st.session_state.current_conversation_id)

    for message in messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("Sources"):
                        # Display sources with metadata if available
                        if "source_metadata" in message and message["source_metadata"]:
                            for source in message["source_metadata"]:
                                source_info = f"**{source['source']}**"
                                if "relevance" in source and source["relevance"]:
                                    source_info += f" (Relevance: {source['relevance']:.1%})"
                                if "page" in source:
                                    source_info += f" - Page {source['page']}"
                                st.markdown(source_info)
                        else:
                            st.write("\n".join(message["sources"]))

    # Query input
    if query := st.chat_input("Ask a question about your documents..."):
        st.chat_message("user").write(query)

        # Add user message to history
        ChatHistory.add_message(
            st.session_state.current_conversation_id,
            "user",
            query
        )

        # Process query with streaming response
        with st.chat_message("assistant"):
            # First get the relevant context
            with st.spinner("Searching for relevant information..."):
                # Query with sources
                result = st.session_state.rag_system.query_with_sources(query)

                # Extract response and sources
                full_response = result["answer"]
                sources = result.get("sources", [])
                source_metadata = result.get("source_metadata", [])

            # Display the response
            st.write(full_response)

            # Show sources
            if sources:
                with st.expander("Sources"):
                    for source in source_metadata:
                        source_info = f"**{source['source']}**"
                        if "relevance" in source and source["relevance"]:
                            source_info += f" (Relevance: {source['relevance']:.1%})"
                        if "page" in source:
                            source_info += f" - Page {source['page']}"
                        st.markdown(source_info)

        # Add assistant message to history with sources as metadata
        ChatHistory.add_message(
            st.session_state.current_conversation_id,
            "assistant",
            full_response,
            {
                "sources": sources,
                "source_metadata": source_metadata
            }
        )


if __name__ == "__main__":
    main()
