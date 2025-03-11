"""
RAG model implementation.
"""
import os
import tempfile
import time
import chromadb
import requests
import json
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

from . import config
from . import utils


class RAGModel:
    """Retrieval-Augmented Generation model implementation."""

    def __init__(self,
                 llm_model_name=config.DEFAULT_LLM_MODEL,
                 embedding_model_name=config.DEFAULT_EMBEDDING_MODEL,
                 db_dir=config.VECTORDB_PATH,
                 collection_name="rag_app",
                 chunk_size=config.DEFAULT_CHUNK_SIZE,
                 chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
                 k_retrieval=10):
        """
        Initialize RAG model with specified parameters.

        Args:
            llm_model_name (str): Name of the LLM model
            embedding_model_name (str): Name of the embedding model
            db_dir (str): Directory for vector database
            collection_name (str): Name of the vector collection
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            k_retrieval (int): Number of documents to retrieve
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.k_retrieval = k_retrieval
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Define system prompt
        self.system_prompt = config.SYSTEM_PROMPT

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        # Initialize cross-encoder model
        self.encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-12-v2")

    def get_vector_collection(self):
        """
        Gets or creates a ChromaDB collection for vector storage.

        Returns:
            chromadb.Collection: Vector collection
        """
        try:
            ollama_ef = OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name=self.embedding_model_name
            )
            chroma_client = chromadb.PersistentClient(path=self.db_dir)
            return chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=ollama_ef,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.error(f"Error connecting to vector database: {e}")
            return None

    def process_document(self, uploaded_file: UploadedFile) -> list[Document]:
        """
        Processes an uploaded PDF file by converting it to text chunks.

        Args:
            uploaded_file (UploadedFile): Uploaded PDF file

        Returns:
            list[Document]: List of document chunks
        """
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile(
            "wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)  # Delete temp file
        return self.text_splitter.split_documents(docs)

    def add_to_vector_collection(self, all_splits: list[Document], file_name: str):
        """
        Adds document splits to a vector collection for semantic search.

        Args:
            all_splits (list[Document]): Document chunks
            file_name (str): Document file name

        Returns:
            int: Number of chunks added
        """
        collection = self.get_vector_collection()
        if not collection:
            return 0

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
        return len(documents)

    def query_collection(self, prompt: str, n_results: int = 10):
        """
        Queries the vector collection with a given prompt.

        Args:
            prompt (str): Query prompt
            n_results (int): Number of results to return

        Returns:
            dict: Query results
        """
        collection = self.get_vector_collection()
        if not collection:
            return {"documents": [[]], "ids": [[]], "distances": [[]]}

        return collection.query(query_texts=[prompt], n_results=n_results)

    def re_rank_documents(self, query: str, documents: list[str], top_k: int = 3) -> tuple[str, list[int]]:
        """
        Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

        Args:
            query (str): Query string
            documents (list[str]): List of documents
            top_k (int): Number of top documents to return

        Returns:
            tuple[str, list[int]]: Relevant text and relevant text IDs
        """
        if not documents or len(documents) == 0:
            return "", []

        relevant_text = ""
        relevant_text_ids = []
        ranks = self.encoder_model.rank(
            query, documents, top_k=min(top_k, len(documents)))
        for rank in ranks:
            relevant_text += documents[rank["corpus_id"]]
            relevant_text_ids.append(rank["corpus_id"])
        return relevant_text, relevant_text_ids

    def call_llm(self, context: str, prompt: str):
        """
        Calls the language model with Digital Ocean GenAI Agent API to generate a response.

        Args:
            context (str): Context for the LLM
            prompt (str): User prompt

        Yields:
            str: Response chunks
        """
        try:
            # Digital Ocean GenAI Agent API endpoint
            api_url = config.DIGITAL_OCEAN_API_URL

            # Authorization headers
            headers = {
                "Authorization": config.DIGITAL_OCEAN_API_KEY,
                "Content-Type": "application/json"
            }

            # Prepare the request payload
            payload = {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Context: {context}, Question: {prompt}"}
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": True,
                "include_retrieval_info": True
            }

            # Make a streaming request to the Digital Ocean endpoint
            response = requests.post(
                api_url, headers=headers, json=payload, stream=True)

            if response.status_code != 200:
                yield f"Error: Received status code {response.status_code} from API"
                return

            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        # Remove 'data: ' prefix if it exists (common in SSE streams)
                        if line.startswith(b'data: '):
                            line = line[6:]

                        # Skip empty lines or heartbeats
                        if not line or line == b':' or line == b'data: [DONE]':
                            continue

                        # Parse the JSON
                        chunk = json.loads(line)

                        # Extract the content based on Digital Ocean API response format
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    yield content
                            elif "message" in choice and "content" in choice["message"]:
                                yield choice["message"]["content"]

                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        continue
                    except Exception as e:
                        yield f"Error parsing response: {str(e)}"

            # Add a fallback in case the stream closes without proper indication
            yield ""

        except Exception as e:
            yield f"Error calling Digital Ocean GenAI API: {str(e)}"

    def delete_document_from_vector_store(self, doc_name):
        """
        Delete a document from the vector store.

        Args:
            doc_name (str): Document name

        Returns:
            bool: True if successful
        """
        collection = self.get_vector_collection()
        if not collection:
            return False

        try:
            all_data = collection.get()
            all_ids = all_data["ids"]
            all_metadatas = all_data["metadatas"]

            # Find document IDs by looking at metadata
            doc_ids = []
            for i, metadata in enumerate(all_metadatas):
                # Check if this metadata contains file information that matches our document
                source = metadata.get('source', '')
                if doc_name in source or doc_name == source:
                    doc_ids.append(all_ids[i])

            # If no matches by metadata, try prefix matching
            if not doc_ids:
                normalize_doc_name = utils.normalize_filename(doc_name)
                doc_ids = [id for id in all_ids if id.startswith(
                    f"{normalize_doc_name}_")]

            # Delete the chunks
            if doc_ids:
                collection.delete(ids=doc_ids)
                st.info(f"Deleted {len(doc_ids)} chunks from vector store")
                return True
        except Exception as e:
            st.warning(f"Vector store operation issue: {e}")
            return False

        return False
