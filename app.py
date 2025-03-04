import os
import streamlit as st

# Import components
from components.sidebar import display_sidebar
from components.rag_chat import rag_chat_app
from components.summarizer import text_summarizer_app
from components.other_tools import other_tools_app

# Import models
from models.rag import LocalRAG

# Import utilities
from utils.chat_history import ChatHistory

# App configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "vectordb")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# App modes
APP_MODES = ["RAG Chat", "Text Summarizer", "Other Tools"]


def initialize_session_state():
    """Initialize session state variables."""
    # Initialize embedding_model in session state if not present
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
    # Initialize llm_model in session state if not present
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "qwen2.5"
    
    # Initialize RAG system (always needed for RAG chat)
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = LocalRAG(
            db_dir=DB_DIR, 
            data_dir=DATA_DIR,
            embedding_model_name=st.session_state.embedding_model,
            model_name=st.session_state.llm_model
        )
    
    # Note: Summarizer will be initialized on demand in the component
    # to avoid loading all models at startup
    
    # Initialize chat history and conversation state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    
    # Initialize app mode first
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = APP_MODES[0]
    
    # Then handle current conversation based on app mode
    if "current_conversation_id" not in st.session_state:
        # Find an existing conversation for the current app mode
        history = ChatHistory.load_history()
        matching_conversations = [
            conv_id for conv_id, data in history.items() 
            if data.get("app_mode") == st.session_state.app_mode
        ]
        
        if matching_conversations:
            # Use the most recent existing conversation for this app mode
            st.session_state.current_conversation_id = matching_conversations[0]
        else:
            # Create a new conversation for this app mode
            conversation_id = ChatHistory.add_conversation(app_mode=st.session_state.app_mode)
            st.session_state.current_conversation_id = conversation_id


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar with DB_DIR and DATA_DIR parameters
    display_sidebar(APP_MODES, DB_DIR, DATA_DIR)
    
    # Display the selected app
    if st.session_state.app_mode == "RAG Chat":
        rag_chat_app()
    elif st.session_state.app_mode == "Text Summarizer":
        text_summarizer_app()
    else:
        other_tools_app()


if __name__ == "__main__":
    main()