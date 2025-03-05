import os
import tempfile

import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile


class RAGModel:
    def __init__(
        self,
        llm_model_name="llama3",
        embedding_model_name="nomic-embed-text:latest",
        db_dir="./demo-rag-chroma",
        data_dir=None,
        collection_name="rag_app",
        chunk_size=400,
        chunk_overlap=100,
        k_retrieval=10
    ):
        """Initialize RAG model with specified parameters"""
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.db_dir = db_dir
        self.data_dir = data_dir or "./data"
        self.collection_name = collection_name
        self.k_retrieval = k_retrieval
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Define system prompt
        self.system_prompt = """
        You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

        context will be passed as "Context:"
        user question will be passed as "Question:"

        To answer the question:
        1. Thoroughly analyze the context, identifying key information relevant to the question.
        2. Organize your thoughts and plan your response to ensure a logical flow of information.
        3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
        4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
        5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

        Format your response as follows:
        1. Use clear, concise language.
        2. Organize your answer into paragraphs for readability.
        3. Use bullet points or numbered lists where appropriate to break down complex information.
        4. If relevant, include any headings or subheadings to structure your response.
        5. Ensure proper grammar, punctuation, and spelling throughout your answer.

        Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
        """

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )

        # Initialize cross-encoder model
        self.encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2")

    def get_vector_collection(self):
        """Gets or creates a ChromaDB collection for vector storage."""
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=self.embedding_model_name,
        )

        chroma_client = chromadb.PersistentClient(path=self.db_dir)
        return chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"},
        )

    def process_document(self, uploaded_file: UploadedFile) -> list[Document]:
        """Processes an uploaded PDF file by converting it to text chunks."""
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile(
            "wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)  # Delete temp file

        return self.text_splitter.split_documents(docs)

    def add_to_vector_collection(self, all_splits: list[Document], file_name: str):
        """Adds document splits to a vector collection for semantic search."""
        collection = self.get_vector_collection()
        documents, metadatas, ids = [], [], []

        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")

        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        return True

    def query_collection(self, prompt: str, n_results: int = 10):
        """Queries the vector collection with a given prompt."""
        collection = self.get_vector_collection()
        results = collection.query(query_texts=[prompt], n_results=n_results)
        return results

    def re_rank_documents(self, query: str, documents: list[str], top_k: int = 3) -> tuple[str, list[int]]:
        """Re-ranks documents using a cross-encoder model for more accurate relevance scoring."""
        relevant_text = ""
        relevant_text_ids = []

        ranks = self.encoder_model.rank(query, documents, top_k=top_k)
        for rank in ranks:
            relevant_text += documents[rank["corpus_id"]]
            relevant_text_ids.append(rank["corpus_id"])

        return relevant_text, relevant_text_ids

    def call_llm(self, context: str, prompt: str, stream=True):
        """Calls the language model with context and prompt to generate a response."""
        response = ollama.chat(
            model=self.llm_model_name,
            stream=stream,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}, Question: {prompt}",
                },
            ],
        )

        if stream:
            for chunk in response:
                if chunk["done"] is False:
                    yield chunk["message"]["content"]
                else:
                    break
        else:
            return response["message"]["content"]
