from transformers import AutoTokenizer
from typing import List, Dict
import io
from PyPDF2 import PdfReader


class TextChunker:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", max_tokens=512):
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def split_pdf_by_pages(self, pdf_bytes):
        chunks = []

        with io.BytesIO(pdf_bytes) as pdf_file:
            reader = PdfReader(pdf_file)
            total_pages = len(reader.pages)

            for i in range(total_pages):
                page = reader.pages[i]
                raw_text = page.extract_text()

                # Clean up formatting
                # Remove multiple newlines
                text = re.sub(r'\n\s*\n', '\n', raw_text)
                # Remove multiple spaces
                text = re.sub(r' +', ' ', text)

                # Check token count
                tokens = self.tokenizer.encode(text)
                if len(tokens) <= self.max_tokens:
                    chunks.append({
                        'text': text,
                        'metadata': {'page_number': i+1, 'total_pages': total_pages}
                    })
                else:
                    # Page exceeds token limit, need to split
                    sub_chunks = self._split_text_by_tokens(text)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            'text': sub_chunk,
                            'metadata': {
                                'page_number': i+1,
                                'chunk_number': j+1,
                                'total_chunks': len(sub_chunks),
                                'total_pages': total_pages
                            }
                        })

        return chunks

    def _split_text_by_tokens(self, text):
        # Split long text into chunks respecting token limits
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))

            if current_tokens + sentence_tokens <= self.max_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Save current chunk and start a new one
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        # Add final chunk if not empty
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks
