import streamlit as st


def main():
    """Main entry point for the Home page."""
    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# 🏠")
    with col2:
        st.header("AI Tools Collection")

    st.markdown(
        "Welcome to our suite of AI-powered tools for document analysis and information extraction.")

    st.divider()

    # Cards for available tools
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 📚 RAG Question Answering")
            st.markdown("""
            Upload documents and ask questions to get AI-powered answers based on your content.
            
            Features:
            - Support for multiple PDF documents
            - Advanced semantic search
            - Multilingual response (English/Indonesian)
            """)
            # Ubah ke navigasi via radio button
            if st.button("Launch RAG", use_container_width=True):
                st.session_state.page = "RAG Question Answer"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### 🔍 OCR (Image to Text)")
            st.markdown("""
            Extract text from images with powerful OCR technology.
            
            Features:
            - Support for PNG, JPG, and JPEG formats
            - Clear text extraction
            - Export results to text files
            """)
            # Ubah ke navigasi via radio button
            if st.button("Launch OCR", use_container_width=True):
                st.session_state.page = "OCR (Image to Text)"
                st.rerun()

    st.divider()

    # Additional information
    st.subheader("About")
    st.markdown("""
    These tools use state-of-the-art AI technologies:
    
    - **Large Language Models**: Using Ollama to run powerful models locally
    - **Vector Databases**: Storing document embeddings for semantic search
    - **OCR Technologies**: Converting images to editable text
    
    Developed to make AI accessible and useful for everyday document tasks.
    """)

    # Footer
    st.divider()
    st.caption("Powered by LLMs and Vector Database Technology")


if __name__ == "__main__":
    main()
