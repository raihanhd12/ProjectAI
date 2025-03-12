"""
Qdrant vector database operations.
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import numpy as np

import config
from utils import helpers


class QdrantDB:
    """Qdrant vector database manager"""

    def __init__(self, collection_name="documents"):
        """
        Initialize Qdrant client connection.

        Args:
            collection_name (str): Name of the collection to use
        """
        self.collection_name = collection_name

        # Connect to Qdrant
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )

        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """Create vector collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            # Get embedding dimension from config
            vector_size = config.EMBEDDING_DIMENSION

            # Create the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")

    def add_documents(self,
                      document_chunks: List[Dict[str, Any]],
                      embeddings: List[List[float]],
                      document_name: str) -> int:
        """
        Add document chunks to the vector database.

        Args:
            document_chunks: List of document chunks with text and metadata
            embeddings: List of vector embeddings for each chunk
            document_name: Name of the document

        Returns:
            int: Number of chunks added
        """
        if not document_chunks or not embeddings:
            return 0

        points = []
        normalized_filename = helpers.normalize_filename(document_name)

        for i, (chunk, embedding) in enumerate(zip(document_chunks, embeddings)):
            # Create unique ID for this chunk
            point_id = f"{normalized_filename}_{i}"

            # Create point with embedding and metadata
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "source": document_name
                    }
                )
            )

        # Upsert points to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return len(points)

    def search(self,
               query_embedding: List[float],
               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of results to return

        Returns:
            List of document chunks with similarity scores
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        documents = []
        for result in results:
            documents.append({
                "id": result.id,
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {}),
                "source": result.payload.get("source", ""),
                "score": result.score
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
            # Find all points with matching source
            normalized_name = helpers.normalize_filename(document_name)
            prefix_filter = models.FieldCondition(
                key="id",
                match=models.MatchValue(value=f"{normalized_name}_"),
                match_text=models.MatchText(text=f"{normalized_name}_")
            )

            # Also check the source field
            source_filter = models.FieldCondition(
                key="source",
                match=models.MatchValue(value=document_name)
            )

            # Combine filters with OR
            filter_condition = models.Filter(
                should=[prefix_filter, source_filter]
            )

            # Find matching points
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                filter=filter_condition,
                limit=1000  # Adjust if you have more chunks per document
            )

            points = search_result[0]

            if points:
                # Extract IDs
                point_ids = [point.id for point in points]

                # Delete by IDs
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )

                print(f"Deleted {len(point_ids)} chunks from vector store")
                return True

            return False

        except Exception as e:
            print(f"Error deleting document from Qdrant: {e}")
            return False

    def reset(self) -> bool:
        """
        Reset the vector database by recreating the collection.

        Returns:
            bool: True if successful
        """
        try:
            # Delete collection
            self.client.delete_collection(collection_name=self.collection_name)

            # Recreate collection
            self._create_collection_if_not_exists()

            return True
        except Exception as e:
            print(f"Error resetting Qdrant database: {e}")
            return False
