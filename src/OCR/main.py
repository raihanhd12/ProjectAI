import streamlit as st
from PIL import Image
import pytesseract


def ocr():
    """Main function to run the OCR tool."""
    st.title("OCR Tool - Extract Text from Image")

    uploaded_file = st.file_uploader(
        "Upload an image for OCR", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Open the image file
        image = Image.open(uploaded_file)

        # Use Tesseract to do OCR on the uploaded image
        text = pytesseract.image_to_string(image)

        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.subheader("Extracted Text:")
        st.write(text)
