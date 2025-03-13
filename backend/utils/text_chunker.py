# utils/text_chunker.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self.text_splitter.split_text(text)
