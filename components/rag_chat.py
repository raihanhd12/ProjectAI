import streamlit as st
from utils.chat_history import ChatHistory


def rag_chat_app():
    """RAG Chat application interface."""
    st.title("RAG Chat Assistant")
    
    # Document upload section
    with st.expander("Upload Documents to Knowledge Base"):
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
    
    messages = ChatHistory.get_messages(st.session_state.current_conversation_id)
    
    for message in messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("Sources"):
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
        
        # Process query
        with st.spinner("Thinking..."):
            result = st.session_state.rag_system.query_with_sources(query)
        
        # Display response
        with st.chat_message("assistant"):
            st.write(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    st.write("\n".join(result["sources"]))
        
        # Add assistant message to history with sources as metadata
        ChatHistory.add_message(
            st.session_state.current_conversation_id,
            "assistant",
            result["answer"],
            {"sources": result["sources"]}
        )