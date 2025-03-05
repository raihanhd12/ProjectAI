import streamlit as st
from models.rag import RAGModel


def rag_chat_component():
    """
    Component for RAG-based question answering with document uploads
    """
    # Initialize RAG model in session state if not already present
    if "rag_model" not in st.session_state:
        st.session_state.rag_model = RAGModel()

    # Document Upload Area
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**",
            type=["pdf"],
            accept_multiple_files=False
        )

        process = st.button("⚡️ Process")

        if uploaded_file and process:
            with st.spinner("Processing document..."):
                normalize_uploaded_file_name = uploaded_file.name.translate(
                    str.maketrans({"-": "_", ".": "_", " ": "_"})
                )
                all_splits = st.session_state.rag_model.process_document(
                    uploaded_file)

                if st.session_state.rag_model.add_to_vector_collection(all_splits, normalize_uploaded_file_name):
                    st.success("Data added to the vector store!")
                else:
                    st.error("Failed to add data to the vector store.")

    # Question and Answer Area
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")

    if ask and prompt:
        with st.spinner("Searching for answers..."):
            # Query the vector database
            results = st.session_state.rag_model.query_collection(prompt)

            if not results or not results.get("documents") or not results.get("documents")[0]:
                st.warning(
                    "No relevant documents found. Please upload some documents first.")
                return

            context = results.get("documents")[0]

            # Re-rank documents for better relevance
            relevant_text, relevant_text_ids = st.session_state.rag_model.re_rank_documents(
                prompt, context)

            # Get answer from LLM
            response = st.session_state.rag_model.call_llm(
                context=relevant_text, prompt=prompt)
            st.write_stream(response)

            # Show debug information in expanders
            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document ids"):
                st.write(relevant_text_ids)
                st.write(relevant_text)
