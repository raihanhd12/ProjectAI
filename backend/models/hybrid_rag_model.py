"""
Updates to HybridRAGModel to support user-specific operations.
"""
import os
import tempfile
import time
import requests
import json
from typing import List, Tuple, Dict, Any, Optional, Generator, Union
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

import config
from utils import helpers
from db.vector_db_qdrant import QdrantDB
from db.elastic_db import ElasticDB


class HybridRAGModel:
    """Hybrid Retrieval-Augmented Generation model implementation with user filtering."""

    def __init__(self,
                 llm_model_name=config.DEFAULT_LLM_MODEL,
                 embedding_model_name=config.DEFAULT_EMBEDDING_MODEL,
                 collection_name="documents",
                 chunk_size=config.DEFAULT_CHUNK_SIZE,
                 chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
                 k_retrieval=10):
        """
        Initialize Hybrid RAG model with specified parameters.

        Args:
            llm_model_name (str): Name of the LLM model
            embedding_model_name (str): Name of the embedding model
            collection_name (str): Name of the vector/document collection
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            k_retrieval (int): Number of documents to retrieve
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
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

        # Initialize vector database
        self.vector_db = QdrantDB(collection_name=collection_name)

        # Initialize elastic database
        self.elastic_db = ElasticDB(index_name=collection_name)

        # Initialize cross-encoder model for re-ranking
        self.encoder_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-12-v2")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using Ollama API.

        Args:
            texts (List[str]): List of text chunks to embed

        Returns:
            List[List[float]]: List of embeddings
        """
        embeddings = []

        for text in texts:
            try:
                response = requests.post(
                    f"{config.OLLAMA_API_BASE}/api/embeddings",
                    json={
                        "model": self.embedding_model_name,
                        "prompt": text
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", [])
                    embeddings.append(embedding)
                else:
                    print(f"Error getting embedding: {response.text}")
                    # Add empty embedding as placeholder
                    embeddings.append([0.0] * config.EMBEDDING_DIMENSION)
            except Exception as e:
                print(f"Exception getting embedding: {e}")
                # Add empty embedding as placeholder
                embeddings.append([0.0] * config.EMBEDDING_DIMENSION)

        return embeddings

    def process_document(self, file_path: str, file_name: str = "",
                         metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Processes a PDF file by converting it to text chunks.

        Args:
            file_path (str): Path to the PDF file
            file_name (str): Optional filename for metadata
            metadata (Dict): Additional metadata to include

        Returns:
            List[Dict[str, Any]]: List of document chunks with text and metadata
        """
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Add filename and additional metadata if provided
        if metadata is None:
            metadata = {}

        for doc in docs:
            doc.metadata["source"] = file_name
            # Add any additional metadata
            for key, value in metadata.items():
                doc.metadata[key] = value

        # Split documents
        split_docs = self.text_splitter.split_documents(docs)

        # Convert to dictionary format
        chunks = []
        for doc in split_docs:
            chunks.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        return chunks

    def add_document(self, file_path: str, file_name: str,
                     metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a document to both vector and keyword search.

        Args:
            file_path (str): Path to the PDF file
            file_name (str): Document file name
            metadata (Dict): Additional metadata to include (e.g., user_id)

        Returns:
            int: Number of chunks added
        """
        # Process document into chunks
        document_chunks = self.process_document(file_path, file_name, metadata)

        if not document_chunks:
            return 0

        # Get text content for embeddings
        texts = [chunk["text"] for chunk in document_chunks]

        # Get embeddings
        embeddings = self.get_embeddings(texts)

        # Add to vector database
        vector_chunks = self.vector_db.add_documents(
            document_chunks, embeddings, file_name)

        # Add to Elasticsearch
        elastic_chunks = self.elastic_db.add_documents(
            document_chunks, file_name)

        return min(vector_chunks, elastic_chunks)

    def hybrid_search(self, query: str, k: int = 10,
                      user_id: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Perform hybrid search using vector and keyword search.

        Args:
            query (str): Search query
            k (int): Number of results to return
            user_id (Optional[int]): Filter results to specific user

        Returns:
            Tuple[List[Dict], List[int]]: Combined results and top indices
        """
        if not config.ENABLE_HYBRID_SEARCH:
            # Just use vector search if hybrid search is disabled
            query_embedding = self.get_embeddings([query])[0]
            vector_results = self.vector_db.search(
                query_embedding, limit=k, filter_metadata={"user_id": user_id} if user_id else None)

            if not vector_results:
                return [], []

            # Re-rank with cross-encoder
            texts = [doc["text"] for doc in vector_results]
            relevant_text_ids = self._rerank_documents(
                query, texts, min(k, len(texts)))

            # Return only the top k results after reranking
            top_results = [vector_results[i] for i in relevant_text_ids]
            return top_results, relevant_text_ids

        # Get vector search results
        query_embedding = self.get_embeddings([query])[0]
        vector_results = self.vector_db.search(
            query_embedding, limit=k, filter_metadata={"user_id": user_id} if user_id else None)

        # Get keyword search results
        keyword_results = self.elastic_db.search(
            query, limit=k, filter_metadata={"user_id": user_id} if user_id else None)

        # Combine results
        combined_results = self._merge_search_results(
            vector_results, keyword_results)

        if not combined_results:
            return [], []

        # Re-rank with cross-encoder
        texts = [doc["text"] for doc in combined_results]
        relevant_text_ids = self._rerank_documents(
            query, texts, min(k, len(texts)))

        # Return only the top k results after reranking
        top_results = [combined_results[i] for i in relevant_text_ids]
        return top_results, relevant_text_ids

    def _merge_search_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and normalize results from vector and keyword search.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search

        Returns:
            List[Dict]: Combined and normalized results
        """
        # Create a map to combine duplicate documents
        result_map = {}

        # Normalize and add vector results
        for i, doc in enumerate(vector_results):
            doc_id = doc["id"]
            if doc_id not in result_map:
                # Normalize score to 0-1 range (already normalized by Qdrant)
                vector_score = doc["score"]
                result_map[doc_id] = {
                    "id": doc_id,
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "source": doc["source"],
                    "vector_score": vector_score,
                    "keyword_score": 0.0,
                    "combined_score": vector_score * config.VECTOR_WEIGHT
                }

        # Normalize keyword results scores
        max_keyword_score = max(
            [doc["score"] for doc in keyword_results]) if keyword_results else 1.0

        # Add keyword results to combined map
        for doc in keyword_results:
            doc_id = doc["id"]
            # Normalize score to 0-1 range
            keyword_score = doc["score"] / \
                max_keyword_score if max_keyword_score > 0 else 0

            if doc_id in result_map:
                # Document already exists from vector search, update scores
                result_map[doc_id]["keyword_score"] = keyword_score
                result_map[doc_id]["combined_score"] += keyword_score * \
                    config.KEYWORD_WEIGHT
            else:
                # New document from keyword search
                result_map[doc_id] = {
                    "id": doc_id,
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "source": doc["source"],
                    "vector_score": 0.0,
                    "keyword_score": keyword_score,
                    "combined_score": keyword_score * config.KEYWORD_WEIGHT
                }

        # Convert map to list and sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

        return combined_results

    def _rerank_documents(self, query: str, documents: List[str], top_k: int = 3) -> List[int]:
        """
        Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

        Args:
            query (str): Query string
            documents (List[str]): List of documents
            top_k (int): Number of top documents to return

        Returns:
            List[int]: Relevant text IDs in order of relevance
        """
        if not documents or len(documents) == 0:
            return []

        relevant_text_ids = []
        ranks = self.encoder_model.rank(
            query, documents, top_k=min(top_k, len(documents)))

        for rank in ranks:
            relevant_text_ids.append(rank["corpus_id"])

        return relevant_text_ids

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

    def delete_document(self, doc_name: str, user_id: Optional[int] = None) -> bool:
        """
        Delete a document from both vector and keyword search.

        Args:
            doc_name (str): Document name
            user_id (Optional[int]): User ID to filter deletion (for multi-user systems)

        Returns:
            bool: True if successful
        """
        vector_success = self.vector_db.delete_document(
            doc_name, filter_metadata={"user_id": user_id} if user_id else None)

        elastic_success = self.elastic_db.delete_document(
            doc_name, filter_metadata={"user_id": user_id} if user_id else None)

        return vector_success and elastic_success

    def reset_databases(self, user_id: Optional[int] = None) -> bool:
        """
        Reset both vector and keyword search databases.
        If user_id is provided, only delete that user's documents.

        Args:
            user_id (Optional[int]): User ID to filter deletion

        Returns:
            bool: True if successful
        """
        if user_id:
            # Delete all documents for this user
            vector_success = self.vector_db.delete_by_filter(
                filter_metadata={"user_id": user_id})

            elastic_success = self.elastic_db.delete_by_filter(
                filter_metadata={"user_id": user_id})
        else:
            # Reset entire databases
            vector_success = self.vector_db.reset()
            elastic_success = self.elastic_db.reset()

        return vector_success and elastic_success
