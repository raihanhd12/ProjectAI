import streamlit as st


def home_page():
    """
    Display an attractive home page with available tools and information.
    """
    # Hero section with logo and title
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image("https://via.placeholder.com/200x200.png?text=AI", width=200)
        st.title("🤖 AI Assistant Platform")
        st.markdown("### Your all-in-one solution for document intelligence")

        st.markdown("""
        Welcome to the AI Assistant Platform, a powerful suite of tools designed to help you 
        interact with and extract insights from your documents using state-of-the-art AI models.
        """)

    # Divider
    st.markdown("---")

    # Available tools section
    st.header("📚 Available Tools")

    # Tool cards in a grid
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("💬 RAG Chat")
            st.markdown("""
            **Retrieval-Augmented Generation**
            
            Chat with your documents! Upload PDFs, Word docs, or text files and ask 
            questions to get accurate, sourced answers based on their content.
            
            **Perfect for:**
            - Extracting specific information from large documents
            - Answering questions about complex content
            - Research and document analysis
            """)

            if st.button("Launch RAG Chat", type="primary", key="home_rag"):
                st.session_state.app_mode = "RAG Chat"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("📝 Summarizer")
            st.markdown("""
            **Intelligent Document Summarization**
            
            Upload documents and get concise, accurate summaries of their content.
            Control summary length and focus areas.
            
            **Perfect for:**
            - Quickly understanding lengthy documents
            - Creating executive summaries
            - Distilling key points from research papers
            """)

            if st.button("Launch Summarizer", type="primary", key="home_summarizer"):
                st.session_state.app_mode = "Summarizer"
                st.rerun()

    # System information
    st.markdown("---")
    st.header("ℹ️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Available Models")
        st.markdown("""
        - **LLM Models**: llama3, mistral, qwen2.5
        - **Embedding Models**: nomic-embed-text, all-MiniLM-L6-v2
        """)

    with col2:
        st.subheader("💻 Technical Requirements")
        st.markdown("""
        - Ollama running locally on port 11434
        - Required models installed via Ollama
        - Python 3.9+ with required packages
        """)

    # Getting started guide
    st.markdown("---")
    st.header("🚀 Getting Started")

    with st.expander("How to use the platform"):
        st.markdown("""
        1. **Select a tool** from the sidebar menu
        2. **Upload documents** relevant to your task
        3. **Interact** with your documents through questions or commands
        4. **View and save** the AI-generated outputs
        5. **Adjust settings** to customize the behavior of each tool
        
        Your chat history is automatically saved for future reference.
        """)
