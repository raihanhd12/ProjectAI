import streamlit as st
from src.RAG.main import rag  # Importing rag() function for RAG
from src.OCR.main import ocr  # Importing ocr() function for OCR


def main():
    """Main application entry point."""
    st.set_page_config(page_title="AI Tools", page_icon="🔧", layout="wide")

    # Sidebar for selecting tool
    st.sidebar.title("Select Tool")
    tool = st.sidebar.radio(
        "Choose a tool:", ["RAG Question Answer", "OCR (Image to Text)"])

    if tool == "RAG Question Answer":
        rag()  # Call the RAG tool function
    elif tool == "OCR (Image to Text)":
        ocr()  # Call the OCR tool function


if __name__ == "__main__":
    main()
