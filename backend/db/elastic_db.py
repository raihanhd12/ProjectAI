"""
Elasticsearch database operations.
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

import config
from utils import helpers


class ElasticDB:
    """Elasticsearch database manager for keyword search"""

    def __init__(self, index_name="documents"):
        """
        Initialize Elasticsearch client connection.

        Args:
            index_name (str): Name of the index to use
        """
        self.index_name = index_name

        # Connect to Elasticsearch
        self.client = Elasticsearch(
            hosts=[config.ELASTICSEARCH_URL],
            basic_auth=(config.ELASTICSEARCH_USER,
                        config.ELASTICSEARCH_PASSWORD),
            verify_certs=config.ELASTICSEARCH_VERIFY_CERTS
        )

        # Create index if it doesn't exist
        self._create_index_if_not_exists()

    def _create_index_if_not_exists(self):
        """Create Elasticsearch index if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
            # Create index with settings and mappings
            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "custom_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "stop", "snowball"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "text": {
                                "type": "text",
                                "analyzer": "custom_analyzer",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword"
                                    }
                                }
                            },
                            "source": {
                                "type": "keyword"
                            },
                            "chunk_id": {
                                "type": "keyword"
                            },
                            "metadata": {
                                "type": "object",
                                "enabled": True
                            }
                        }
                    }
                }
            )
            print(f"Created new Elasticsearch index: {self.index_name}")

    def add_documents(self,
                      document_chunks: List[Dict[str, Any]],
                      document_name: str) -> int:
        """
        Add document chunks to Elasticsearch.

        Args:
            document_chunks: List of document chunks with text and metadata
            document_name: Name of the document

        Returns:
            int: Number of chunks added
        """
        if not document_chunks:
            return 0

        actions = []
        normalized_filename = helpers.normalize_filename(document_name)

        for i, chunk in enumerate(document_chunks):
            # Create unique ID for this chunk
            chunk_id = f"{normalized_filename}_{i}"

            # Create document for indexing
            actions.append({
                "_index": self.index_name,
                "_id": chunk_id,
                "_source": {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "source": document_name,
                    "chunk_id": chunk_id
                }
            })

        # Bulk index documents
        success, failed = bulk(self.client, actions, refresh=True)

        return success

    def search(self,
               query: str,
               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents using keyword search.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of document chunks with relevance scores
        """
        response = self.client.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^3", "metadata.*"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": limit
            }
        )

        documents = []
        for hit in response["hits"]["hits"]:
            documents.append({
                "id": hit["_id"],
                "text": hit["_source"]["text"],
                "metadata": hit["_source"]["metadata"],
                "source": hit["_source"]["source"],
                "score": hit["_score"]
            })

        return documents

    def delete_document(self, document_name: str) -> bool:
        """
        Delete all chunks belonging to a document.

        Args:
            document_name: Name of document to delete

        Returns:
            bool: True if successful
        """
        try:
            # Delete by query matching the document name
            response = self.client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "match": {
                            "source": document_name
                        }
                    }
                },
                refresh=True
            )

            deleted = response.get("deleted", 0)
            print(f"Deleted {deleted} chunks from Elasticsearch")
            return deleted > 0

        except Exception as e:
            print(f"Error deleting document from Elasticsearch: {e}")
            return False

    def reset(self) -> bool:
        """
        Reset the Elasticsearch index by deleting and recreating it.

        Returns:
            bool: True if successful
        """
        try:
            # Delete index if it exists
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)

            # Recreate index
            self._create_index_if_not_exists()

            return True
        except Exception as e:
            print(f"Error resetting Elasticsearch index: {e}")
            return False
