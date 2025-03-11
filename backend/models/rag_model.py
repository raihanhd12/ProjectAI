"""
RAG model implementation.
"""
import os
import tempfile
import time
import chromadb
import requests
import json
from typing import List, Tuple, Dict, Any, Optional, Generator, Union
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

import config
from utils import helpers


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
            
        # Ensure vector DB directory exists
        os.makedirs(self.db_dir, exist_ok=True)

    def get_vector_collection(self):
        """
        Gets or creates a ChromaDB collection for vector storage.

        Returns:
            chromadb.Collection: Vector collection
        """
        try:
            ollama_ef = OllamaEmbeddingFunction(
                url=f"{config.OLLAMA_API_BASE}/api/embeddings",
                model_name=self.embedding_model_name
            )
            chroma_client = chromadb.PersistentClient(path=self.db_dir)
            return chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=ollama_ef,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Error connecting to vector database: {e}")
            return None

    def process_document(self, file_path: str, file_name: str = "") -> List[Document]:
        """
        Processes a PDF file by converting it to text chunks.

        Args:
            file_path (str): Path to the PDF file
            file_name (str): Optional filename for metadata

        Returns:
            List[Document]: List of document chunks
        """
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        # Add filename to metadata if provided
        if file_name:
            for doc in docs:
                doc.metadata["source"] = file_name
                
        return self.text_splitter.split_documents(docs)

    def add_to_vector_collection(self, all_splits: List[Document], file_name: str) -> int:
        """
        Adds document splits to a vector collection for semantic search.

        Args:
            all_splits (List[Document]): Document chunks
            file_name (str): Document file name

        Returns:
            int: Number of chunks added
        """
        collection = self.get_vector_collection()
        if not collection:
            return 0

        documents, metadatas, ids = [], [], []
        normalized_filename = helpers.normalize_filename(file_name)
        
        for idx, split in enumerate(all_splits):
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{normalized_filename}_{idx}")
            
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        return len(documents)

    def query_collection(self, prompt: str, n_results: int = 10) -> Dict:
        """
        Queries the vector collection with a given prompt.

        Args:
            prompt (str): Query prompt
            n_results (int): Number of results to return

        Returns:
            Dict: Query results
        """
        collection = self.get_vector_collection()
        if not collection:
            return {"documents": [[]], "ids": [[]], "distances": [[]]}

        return collection.query(query_texts=[prompt], n_results=n_results)

    def re_rank_documents(self, query: str, documents: List[str], top_k: int = 3) -> Tuple[str, List[int]]:
        """
        Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

        Args:
            query (str): Query string
            documents (List[str]): List of documents
            top_k (int): Number of top documents to return

        Returns:
            Tuple[str, List[int]]: Relevant text and relevant text IDs
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

    def call_llm(self, context: str, prompt: str) -> Generator[str, None, None]:
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

    def delete_document_from_vector_store(self, doc_name: str) -> bool:
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
                normalize_doc_name = helpers.normalize_filename(doc_name)
                doc_ids = [id for id in all_ids if id.startswith(
                    f"{normalize_doc_name}_")]

            # Delete the chunks
            if doc_ids:
                collection.delete(ids=doc_ids)
                print(f"Deleted {len(doc_ids)} chunks from vector store")
                return True
        except Exception as e:
            print(f"Vector store operation issue: {e}")
            return False

        return False