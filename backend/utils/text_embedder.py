# utils/text_embedder.py
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Single text embedding for queries"""
        if not text:
            return [0] * self.embedding_dimension
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        return self.embedding_dimension
