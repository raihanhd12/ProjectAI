# utils/text_extractor.py
import io
from typing import Optional
from PyPDF2 import PdfReader
from docx import Document
import textract


class TextExtractor:
    @staticmethod
    def extract_from_bytes(file_content: bytes, content_type: str) -> Optional[str]:
        try:
            if "pdf" in content_type:
                with io.BytesIO(file_content) as file:
                    reader = PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            elif "docx" in content_type:
                with io.BytesIO(file_content) as file:
                    doc = Document(file)
                    return "\n".join([para.text for para in doc.paragraphs])
            elif "text/plain" in content_type:
                return file_content.decode('utf-8', errors='replace')
            else:
                # Fallback to textract for other formats
                return textract.process(file_content).decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None
