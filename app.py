import streamlit as st
from components.sidebar import sidebar_menu
from components.home import home_page
from components.rag_chat import rag_chat_page

# Define application constants
DB_DIR = "./vectordb"
DATA_DIR = "./data"


def main():
    """
    Main application entry point with multi-tool navigation.
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="AI Assistant Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session states for application
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "Home"

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "llama3"

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "nomic-embed-text"

    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None

    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False

    # Available application modes
    app_modes = ["Home", "RAG Chat", "Summarizer"]

    # Set up sidebar navigation
    sidebar_menu(app_modes, DB_DIR, DATA_DIR)

    # Display selected application mode
    if st.session_state.app_mode == "Home":
        home_page()
    elif st.session_state.app_mode == "RAG Chat":
        rag_chat_page(DB_DIR, DATA_DIR)
    # Add other tools here as elif statements

    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 AI Assistant Platform")


if __name__ == "__main__":
    main()
