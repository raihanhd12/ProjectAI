import io
import re
from typing import Optional
from PyPDF2 import PdfReader
from docx import Document


class TextExtractor:
    @staticmethod
    def extract_from_bytes(file_content: bytes, content_type: str) -> Optional[str]:
        try:
            if "pdf" in content_type:
                with io.BytesIO(file_content) as file:
                    reader = PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        # Clean up excessive whitespace
                        page_text = re.sub(r'\n\s*\n', '\n', page_text)
                        # Remove redundant spaces
                        page_text = re.sub(r' +', ' ', page_text)
                        text += page_text + "\n"
                    return text
            elif "docx" in content_type:
                with io.BytesIO(file_content) as file:
                    doc = Document(file)
                    return "\n".join([para.text for para in doc.paragraphs])
            elif "text/plain" in content_type:
                return file_content.decode('utf-8', errors='replace')
            else:
                # For unsupported types, return a message
                return f"Text extraction not supported for {content_type}"
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None
