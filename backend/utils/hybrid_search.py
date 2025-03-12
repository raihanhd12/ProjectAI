"""
Hybrid search utility combining Qdrant for vector search and Elasticsearch for keyword search.
"""
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models

import config
from utils import helpers


class HybridSearch:
    """Hybrid search implementation using Qdrant and Elasticsearch."""

    def __init__(self,
                 qdrant_host=config.QDRANT_HOST,
                 qdrant_port=config.QDRANT_PORT,
                 es_host=config.ELASTICSEARCH_HOST,
                 es_port=config.ELASTICSEARCH_PORT,
                 collection_name="document_chunks",
                 es_index_name="document_chunks"):
        """
        Initialize hybrid search with Qdrant and Elasticsearch.

        Args:
            qdrant_host (str): Qdrant host address
            qdrant_port (int): Qdrant port
            es_host (str): Elasticsearch host address
            es_port (int): Elasticsearch port
            collection_name (str): Qdrant collection name
            es_index_name (str): Elasticsearch index name
        """
        self.collection_name = collection_name
        self.es_index_name = es_index_name

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Initialize Elasticsearch client
        self.es_client = Elasticsearch(f"http://{es_host}:{es_port}")

    def init_qdrant_collection(self, vector_size=768):
        """
        Initialize or confirm Qdrant collection exists.

        Args:
            vector_size (int): Size of embedding vectors

        Returns:
            bool: True if successful
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")

            return True
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")
            return False

    def init_elasticsearch_index(self):
        """
        Initialize or confirm Elasticsearch index exists.

        Returns:
            bool: True if successful
        """
        try:
            # Check if index exists
            if not self.es_client.indices.exists(index=self.es_index_name):
                # Create index with appropriate mappings
                self.es_client.indices.create(
                    index=self.es_index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "content": {
                                    "type": "text",
                                    "analyzer": "standard"
                                },
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "keyword"},
                                        "page": {"type": "integer"}
                                    }
                                },
                                "chunk_id": {"type": "keyword"}
                            }
                        },
                        "settings": {
                            "analysis": {
                                "analyzer": {
                                    "standard": {
                                        "type": "standard"
                                    }
                                }
                            }
                        }
                    }
                )
                print(f"Created Elasticsearch index: {self.es_index_name}")

            return True
        except Exception as e:
            print(f"Error initializing Elasticsearch index: {e}")
            return False

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]], doc_ids: List[str]) -> int:
        """
        Add documents to both Qdrant and Elasticsearch.

        Args:
            documents (List[Dict]): List of document dictionaries with 'content' and 'metadata'
            embeddings (List[List[float]]): List of embeddings as vectors
            doc_ids (List[str]): List of document IDs

        Returns:
            int: Number of documents added
        """
        if not documents or not embeddings or not doc_ids:
            return 0

        if len(documents) != len(embeddings) or len(documents) != len(doc_ids):
            print("Error: documents, embeddings, and doc_ids must have the same length")
            return 0

        try:
            # Add documents to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            "content": doc["content"],
                            "metadata": doc["metadata"]
                        }
                    )
                    for doc, embedding, doc_id in zip(documents, embeddings, doc_ids)
                ]
            )

            # Add documents to Elasticsearch
            bulk_operations = []
            for doc, doc_id in zip(documents, doc_ids):
                # Index action
                bulk_operations.append(
                    {"index": {"_index": self.es_index_name, "_id": doc_id}})
                # Document data
                bulk_operations.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "chunk_id": doc_id
                })

            if bulk_operations:
                self.es_client.bulk(operations=bulk_operations, refresh=True)

            return len(documents)
        except Exception as e:
            print(f"Error adding documents to search stores: {e}")
            return 0

    def delete_documents(self, source: str) -> bool:
        """
        Delete documents by source from both Qdrant and Elasticsearch.

        Args:
            source (str): Document source to delete

        Returns:
            bool: True if successful
        """
        try:
            # Delete from Qdrant using filter
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(value=source)
                            )
                        ]
                    )
                )
            )

            # Delete from Elasticsearch
            self.es_client.delete_by_query(
                index=self.es_index_name,
                query={
                    "term": {
                        "metadata.source": source
                    }
                },
                refresh=True
            )

            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    def semantic_search(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """
        Perform semantic search in Qdrant.

        Args:
            query_vector (List[float]): Query embedding vector
            limit (int): Maximum number of results

        Returns:
            List[Dict]: Search results
        """
        try:
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

            return [
                {
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "metadata": hit.payload.get("metadata", {}),
                    "score": hit.score
                }
                for hit in search_result
            ]
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def keyword_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform keyword search in Elasticsearch.

        Args:
            query (str): Search query
            limit (int): Maximum number of results

        Returns:
            List[Dict]: Search results
        """
        try:
            search_result = self.es_client.search(
                index=self.es_index_name,
                query={
                    "match": {
                        "content": query
                    }
                },
                size=limit
            )

            return [
                {
                    "id": hit["_id"],
                    "content": hit["_source"].get("content", ""),
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"]
                }
                for hit in search_result["hits"]["hits"]
            ]
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []

    def hybrid_search(self, query: str, query_vector: List[float], limit: int = 10,
                      semantic_weight: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query (str): Text query
            query_vector (List[float]): Query embedding vector
            limit (int): Maximum number of results
            semantic_weight (float): Weight given to semantic results (0-1)

        Returns:
            List[Dict]: Combined search results
        """
        try:
            # Get more results than needed from both sources
            # Get more results to have a good pool for reranking
            expanded_limit = min(limit * 3, 30)
            semantic_results = self.semantic_search(
                query_vector, expanded_limit)
            keyword_results = self.keyword_search(query, expanded_limit)

            # Combine results
            all_results = {}

            # Add semantic results with weight
            for result in semantic_results:
                doc_id = result["id"]
                all_results[doc_id] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "semantic_score": result["score"],
                    "keyword_score": 0.0
                }

            # Add or update with keyword results
            for result in keyword_results:
                doc_id = result["id"]
                if doc_id in all_results:
                    all_results[doc_id]["keyword_score"] = result["score"]
                else:
                    all_results[doc_id] = {
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "semantic_score": 0.0,
                        "keyword_score": result["score"]
                    }

            # Normalize scores
            max_semantic = max(
                [r["semantic_score"] for r in all_results.values()]) if semantic_results else 1.0
            max_keyword = max(
                [r["keyword_score"] for r in all_results.values()]) if keyword_results else 1.0

            for doc_id in all_results:
                if max_semantic > 0:
                    all_results[doc_id]["semantic_score"] /= max_semantic
                if max_keyword > 0:
                    all_results[doc_id]["keyword_score"] /= max_keyword

                # Calculate combined score
                all_results[doc_id]["combined_score"] = (
                    all_results[doc_id]["semantic_score"] * semantic_weight +
                    all_results[doc_id]["keyword_score"] *
                    (1 - semantic_weight)
                )

            # Sort by combined score and return top results
            results_list = [
                {
                    "id": doc_id,
                    "content": data["content"],
                    "metadata": data["metadata"],
                    "semantic_score": data["semantic_score"],
                    "keyword_score": data["keyword_score"],
                    "score": data["combined_score"]
                }
                for doc_id, data in all_results.items()
            ]

            results_list.sort(key=lambda x: x["score"], reverse=True)
            return results_list[:limit]

        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def reset(self) -> bool:
        """
        Reset Qdrant collection and Elasticsearch index.

        Returns:
            bool: True if successful
        """
        try:
            # Delete and recreate Qdrant collection
            try:
                self.qdrant_client.delete_collection(
                    collection_name=self.collection_name)
            except:
                pass  # Collection might not exist

            # Delete and recreate Elasticsearch index
            try:
                self.es_client.indices.delete(index=self.es_index_name)
            except:
                pass  # Index might not exist

            # Recreate both
            self.init_qdrant_collection()
            self.init_elasticsearch_index()

            return True
        except Exception as e:
            print(f"Error resetting search stores: {e}")
            return False
