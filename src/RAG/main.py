"""
Main entry point for the RAG module.
"""
import os
import streamlit as st

from . import config
from . import db
from . import models
from . import utils
from .ui import sidebar
from .ui import documents
from .ui import chat


def rag():
    """Main function to run the RAG tool."""
    # Make sure required directories exist - using just the directory paths
    utils.ensure_directories([
        config.VECTORDB_PATH,
        os.path.dirname(config.DB_PATH)
    ])

    # Initialize the database
    db.init_db()

    # Initialize RAG model if not already initialized
    if "rag_model" not in st.session_state:
        st.session_state.rag_model = models.RAGModel()

    # First do sidebar navigation
    sidebar.navigation_sidebar()

    # Then call the sidebar component with all the settings
    sidebar.configuration_sidebar()

    # Display debug information in the sidebar (optional)
    # sidebar.display_debug_info()

    # Display the selected view
    if st.session_state.get("rag_view", "documents") == "documents":
        documents.document_management_component()
    else:
        chat.chat_component()


# Auto-run for testing during development
if __name__ == "__main__":
    rag()
