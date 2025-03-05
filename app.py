import streamlit as st
from components.sidebar import sidebar_component
from components.rag_chat import rag_chat_component


def main():
    """
    Main application entry point.
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="RAG Question Answer",
        page_icon="📚",
        layout="wide"
    )

    # Set up sidebar for configuration
    sidebar_component()

    # Main RAG chat interface
    rag_chat_component()


if __name__ == "__main__":
    main()
