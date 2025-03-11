"""
Document management UI components for the RAG module.
"""
import time
import streamlit as st

from .. import db
from .. import utils
from .. import models


def document_management_component():
    """
    Component for document management.

    Returns:
        None
    """
    # Initialize database
    db_init_result = db.init_db()
    if not db_init_result:
        st.error("Failed to initialize database. Check sidebar for details.")

    # Initialize RAG model if not already initialized
    if "rag_model" not in st.session_state:
        st.session_state.rag_model = models.RAGModel()

    # Document Management Section
    st.header("Document Management")

    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        process_button = st.button(
            "Process Documents", type="primary", use_container_width=True)

    # Process documents
    if uploaded_files and process_button:
        with st.status("Processing documents...") as status:
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")
                normalized_filename = utils.normalize_filename(
                    uploaded_file.name)
                all_splits = st.session_state.rag_model.process_document(
                    uploaded_file)
                st.write(f"Created {len(all_splits)} text chunks")

                chunks_added = st.session_state.rag_model.add_to_vector_collection(
                    all_splits, normalized_filename)

                # Save document to DB
                db.save_document(uploaded_file.name, len(all_splits))

                st.success(
                    f"Added {chunks_added} chunks from {uploaded_file.name}")

            status.update(
                label=f"Documents processed successfully!", state="complete")

    display_document_list()


def display_document_list():
    """
    Display list of indexed documents.

    Returns:
        None
    """
    # Display processed documents
    documents = db.get_documents()
    if documents:
        st.subheader("Indexed Documents")

        # Create columns for the header
        col1, col2, col3, col4 = st.columns([0.1, 0.6, 0.2, 0.1])
        col1.write("**#**")
        col2.write("**Document**")
        col3.write("**Chunks**")
        col4.write("**Action**")

        st.divider()

        # List documents with delete buttons
        for i, doc in enumerate(documents):
            col1, col2, col3, col4 = st.columns([0.1, 0.6, 0.2, 0.1])
            col1.write(f"{i+1}")
            col2.write(doc['name'])
            col3.write(doc['chunks'])
            if col4.button("🗑️", key=f"delete_doc_{i}", help=f"Delete {doc['name']}"):
                delete_document(doc['name'])

        st.divider()

        # Add option to delete all documents
        if st.button("Reset Vector Database", type="tertiary"):
            reset_vector_database()
    else:
        st.info("No documents indexed. Upload and process documents to start.")


def delete_document(doc_name):
    """
    Delete a document and update the UI.

    Args:
        doc_name (str): Document name

    Returns:
        None
    """
    # Delete from database
    db_success = db.delete_document(doc_name)

    # Delete from vector store
    if "rag_model" in st.session_state:
        vector_success = st.session_state.rag_model.delete_document_from_vector_store(
            doc_name)
    else:
        vector_success = False

    if db_success:
        st.success(f"Deleted {doc_name}")
        time.sleep(1)  # Give user time to see the message
        st.rerun()
    else:
        st.error(f"Failed to delete {doc_name}")


def reset_vector_database():
    """
    Reset the vector database and update the UI.

    Returns:
        None
    """
    success = db.reset_vector_database()
    if success:
        # Re-initialize the RAG model with a fresh vector store
        st.session_state.rag_model = models.RAGModel()
        st.success("Vector database has been reset")
        time.sleep(1)  # Give user time to see the message
        st.rerun()
    else:
        st.error("Failed to reset vector database")
