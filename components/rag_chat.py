import streamlit as st
import os
from models.rag import RAGModel
from utils.chat_history import ChatHistory


def initialize_rag_system(db_dir, data_dir):
    """Initialize the RAG system with proper parameters"""
    try:
        # Get chunking parameters from session state or set defaults
        chunk_size = st.session_state.get("chunk_size", 400)
        chunk_overlap = st.session_state.get("chunk_overlap", 100)
        k_retrieval = st.session_state.get("k_retrieval", 10)

        # Initialize RAG system with current settings
        rag_system = RAGModel(
            llm_model_name=st.session_state.llm_model,
            embedding_model_name=st.session_state.embedding_model,
            db_dir=db_dir,
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k_retrieval=k_retrieval
        )
        return rag_system
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        st.error(
            "Please check that Ollama is running and the required models are available.")
        return None


def settings_dialog():
    """Show settings dialog for RAG configuration"""
    with st.sidebar:
        st.subheader("⚙️ RAG Settings")

        # Model configuration section
        st.write("**Model Configuration**")

        # LLM model selection
        llm_models = [
            "llama3",
            "mistral",
            "qwen2.5",
            "qwen2.5-14b"
        ]

        selected_llm = st.selectbox(
            "LLM Model",
            llm_models,
            index=llm_models.index(
                st.session_state.llm_model) if st.session_state.llm_model in llm_models else 0
        )

        # Embedding model selection
        embedding_models = [
            "nomic-embed-text",
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "intfloat/e5-small-v2"
        ]

        selected_embedding = st.selectbox(
            "Embedding Model",
            embedding_models,
            index=embedding_models.index(st.session_state.embedding_model)
            if st.session_state.embedding_model in embedding_models else 0
        )

        # Chunking configuration
        st.write("**Chunking Configuration**")

        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=st.session_state.get("chunk_size", 400),
            step=50
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=300,
            value=st.session_state.get("chunk_overlap", 100),
            step=10
        )

        k_retrieval = st.slider(
            "Number of Retrieved Documents",
            min_value=1,
            max_value=20,
            value=st.session_state.get("k_retrieval", 10),
            step=1
        )

        # Apply settings button
        if st.button("Apply Settings", type="primary", use_container_width=True):
            # Update session state
            st.session_state.llm_model = selected_llm
            st.session_state.embedding_model = selected_embedding
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.k_retrieval = k_retrieval

            # Reinitialize RAG system with new settings
            try:
                st.session_state.rag_system = None  # Force reinitialization
                st.success(
                    "Settings applied successfully! System will reinitialize.")
                st.session_state.show_settings = False  # Hide settings after applying
                st.rerun()
            except Exception as e:
                st.error(f"Error applying settings: {str(e)}")

        # Close settings button
        if st.button("Close Settings", use_container_width=True):
            st.session_state.show_settings = False
            st.rerun()


def rag_chat_page(db_dir, data_dir):
    """RAG Chat application interface with streaming responses and chat history"""
    st.title("💬 RAG Chat Assistant")

    # Show settings if requested
    if st.session_state.show_settings:
        settings_dialog()

    # Initialize RAG system if not already in session
    if "rag_system" not in st.session_state or st.session_state.rag_system is None:
        st.session_state.rag_system = initialize_rag_system(db_dir, data_dir)
        if st.session_state.rag_system is None:
            st.stop()

    # Create conversation if none exists
    if st.session_state.current_conversation_id is None:
        new_id = ChatHistory.add_conversation(app_mode="RAG Chat")
        st.session_state.current_conversation_id = new_id

    # Document upload section
    with st.expander("📄 Upload Documents", expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            uploaded_files = st.file_uploader(
                "Upload documents (PDF, TXT, DOCX, MD)",
                accept_multiple_files=True,
                type=["pdf", "txt", "docx", "md"]
            )

        with col2:
            st.write("")
            st.write("")  # Spacing for alignment
            process_button = st.button(
                "Process Documents", type="primary", use_container_width=True)

        if uploaded_files and process_button:
            with st.spinner("Processing documents..."):
                success = st.session_state.rag_system.ingest_documents(
                    uploaded_files)
                if success:
                    st.success(
                        f"Successfully processed {len(uploaded_files)} documents!")
                else:
                    st.error("Failed to process documents.")

    # Main chat interface
    st.subheader("Conversation")

    # Get messages for current conversation
    messages = ChatHistory.get_messages(
        st.session_state.current_conversation_id)

    # Display messages with expandable sources
    for message in messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        # Display sources with metadata if available
                        if "source_metadata" in message and message["source_metadata"]:
                            for source in message["source_metadata"]:
                                source_info = f"**{source['source']}**"
                                if "relevance" in source and source["relevance"] is not None:
                                    source_info += f" (Relevance: {source['relevance']:.1%})"
                                if "page" in source:
                                    source_info += f" - Page {source['page']}"
                                st.markdown(source_info)
                        else:
                            st.write("\n".join(message["sources"]))

    # Query input
    if query := st.chat_input("Ask a question about your documents..."):
        # Add user message to UI
        st.chat_message("user").write(query)

        # Add user message to history
        ChatHistory.add_message(
            st.session_state.current_conversation_id,
            "user",
            query
        )

        # Check if RAG system is ready
        if st.session_state.rag_system is None:
            st.session_state.rag_system = initialize_rag_system(
                db_dir, data_dir)
            if st.session_state.rag_system is None:
                st.error(
                    "Failed to initialize RAG system. Please check settings and try again.")
                st.stop()

        # Process query with response
        with st.chat_message("assistant"):
            # Get relevant context and response
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
                with st.expander("📚 Sources"):
                    for source in source_metadata:
                        source_info = f"**{source['source']}**"
                        if "relevance" in source and source["relevance"] is not None:
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
