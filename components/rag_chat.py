import streamlit as st
import os
from models.rag import StreamlinedRAG
from utils.chat_history import ChatHistory


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

        # Process query with response
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
            response_placeholder = st.empty()
            response_placeholder.write(full_response)

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
