import streamlit as st

# Set up page config as the first Streamlit command
st.set_page_config(page_title="AI Document Assistant", layout="wide")

# Now import modules after page config
# We'll use a function to import them on-demand to avoid any module-level Streamlit commands


def import_rag():
    try:
        from src.RAG.main import rag
        return rag
    except Exception as e:
        def error_rag():
            st.error(f"Failed to load RAG module: {str(e)}")
        return error_rag


def main():
    # Initialize session state for page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    # Main navigation sidebar
    with st.sidebar:
        st.title("AI Document Assistant")
        st.divider()

        # Navigation buttons
        if st.button("🏠 Home", use_container_width=True,
                     type="primary" if st.session_state.current_page == "home" else "secondary"):
            st.session_state.current_page = "home"
            st.rerun()

        if st.button("🔍 RAG Tools", use_container_width=True,
                     type="primary" if st.session_state.current_page == "rag" else "secondary"):
            st.session_state.current_page = "rag"
            # Save state before rerun
            st.rerun()

        # # OCR navigation button
        # if st.button("📷 OCR Tools", use_container_width=True,
        #              type="primary" if st.session_state.current_page == "ocr" else "secondary"):
        #     st.session_state.current_page = "ocr"
        #     st.rerun()

        st.divider()
        st.caption("© 2024 My Company")

    # Render the selected page
    if st.session_state.current_page == "home":
        home_page()
    elif st.session_state.current_page == "rag":
        # Import and call RAG only when needed
        rag_function = import_rag()
        rag_function()
    # elif st.session_state.current_page == "ocr":
    #     # Import and call OCR only when needed
    #     # from src.OCR.main import ocr
    #     # ocr()
    #     pass


def home_page():
    """Render the home page"""
    st.title("Welcome to AI Document Assistant")

    # Introduction
    st.markdown("""
    This application provides powerful tools for document analysis and interaction:
    
    - **RAG (Retrieval-Augmented Generation)**: Chat with your documents using AI
    - **OCR (Coming Soon)**: Extract text from images and scanned documents
    """)

    # Features section
    st.subheader("Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info("### 📚 RAG Tools")
        st.markdown("""
        - Upload and process PDF documents
        - Ask questions about your documents
        - Get AI-powered answers based on document content
        - Save and manage conversation history
        """)

        # Button to go to RAG page
        if st.button("Go to RAG Tools", use_container_width=True):
            st.session_state.current_page = "rag"
            st.rerun()

    with col2:
        st.info("### 📷 OCR Tools (Coming Soon)")
        st.markdown("""
        - Upload images or scanned documents
        - Extract text using optical character recognition
        - Process and analyze extracted text
        - Save results in various formats
        """)

    # Usage tips
    st.subheader("Getting Started")
    st.markdown("""
    1. Navigate to the RAG Tools section
    2. Upload your PDF documents
    3. Process the documents to index them
    4. Start asking questions about your documents
    5. View and manage your conversation history
    """)


if __name__ == "__main__":
    main()
