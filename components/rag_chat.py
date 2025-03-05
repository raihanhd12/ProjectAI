import streamlit as st
from utils.chat_history import ChatHistory


def rag_chat_app():
    """RAG Chat application interface with streaming responses."""
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
                # Find relevant documents
                results = st.session_state.rag_system.collection.query(
                    query_texts=[query],
                    n_results=st.session_state.rag_system.k_retrieval,
                    include=["documents", "metadatas", "distances"]
                )

                documents = results.get("documents")[0]
                metadatas = results.get("metadatas")[0]

                # Re-rank for relevance
                relevant_text, relevant_ids = st.session_state.rag_system.re_rank_documents(
                    query, documents
                )

                # Extract sources
                sources = []
                source_metadata = []
                seen_sources = set()

                for idx in relevant_ids:
                    if idx < len(metadatas):
                        metadata = metadatas[idx]
                        if "source" in metadata:
                            source = metadata["source"]
                            if source not in seen_sources:
                                seen_sources.add(source)
                                sources.append(source)

                                # Extract additional metadata
                                source_meta = {
                                    "source": source,
                                    "relevance": 1 - (results.get("distances", [[0]])[0][idx] / 2)
                                    if "distances" in results else None
                                }
                                for key in ["page", "file_type", "date_processed"]:
                                    if key in metadata:
                                        source_meta[key] = metadata[key]

                                source_metadata.append(source_meta)

            # Display the streaming response
            response_placeholder = st.empty()
            full_response = ""

            # Stream the response
            for chunk in st.session_state.rag_system.stream_llm_response(relevant_text, query):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            # Replace the placeholder with the full response
            response_placeholder.markdown(full_response)

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
