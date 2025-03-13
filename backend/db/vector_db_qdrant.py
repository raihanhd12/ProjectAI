"""
Qdrant vector database operations with user filtering.
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
import numpy as np

import config
from utils import helpers


class QdrantDB:
    """Qdrant vector database manager with user filtering"""

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
            # Generate UUID as point ID
            point_id = str(uuid.uuid4())

            # Create point with embedding and metadata
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "source": document_name,
                        "custom_id": f"{normalized_filename}_{i}"  # Opsional
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
               limit: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of results to return
            filter_metadata: Optional metadata to filter results (e.g., user_id)

        Returns:
            List of document chunks with similarity scores
        """
        try:
            # Build search filter if metadata filter provided
            search_filter = None
            if filter_metadata and isinstance(filter_metadata, dict):
                filter_conditions = []

                for key, value in filter_metadata.items():
                    if value is not None:  # Only add non-None filters
                        # Create filter condition for this metadata key
                        filter_conditions.append(
                            FieldCondition(
                                key=f"metadata.{key}",
                                # Convert to string for consistent matching
                                match=MatchValue(value=str(value))
                            )
                        )

                if filter_conditions:
                    search_filter = Filter(
                        must=filter_conditions
                    )

            # Perform search with optional filter
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                filter=search_filter
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
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def delete_document(self, document_name: str,
                        filter_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Delete all chunks belonging to a document.

        Args:
            document_name: Name of document to delete
            filter_metadata: Optional metadata to filter deletion (e.g., user_id)

        Returns:
            bool: True if successful
        """
        try:
            # Base filter conditions
            filter_conditions = []

            # Add document name filter
            normalized_name = helpers.normalize_filename(document_name)

            # ID prefix filter
            prefix_filter = FieldCondition(
                key="id",
                match=models.MatchText(text=f"{normalized_name}_")
            )
            filter_conditions.append(prefix_filter)

            # Source field filter
            source_filter = FieldCondition(
                key="source",
                match=MatchValue(value=document_name)
            )
            filter_conditions.append(source_filter)

            # Add user_id or other metadata filter if provided
            if filter_metadata and isinstance(filter_metadata, dict):
                for key, value in filter_metadata.items():
                    if value is not None:  # Only add non-None filters
                        metadata_filter = FieldCondition(
                            key=f"metadata.{key}",
                            # Convert to string for consistent matching
                            match=MatchValue(value=str(value))
                        )
                        filter_conditions.append(metadata_filter)

            # Combine with AND logic if we have metadata filters
            # Otherwise use OR logic between prefix and source
            filter_obj = None
            if filter_metadata:
                # If we have metadata filters, documents must match BOTH
                # (document name AND metadata)
                filter_obj = Filter(
                    must=[
                        # Document must match name (using OR between ID prefix and source)
                        Filter(should=[prefix_filter, source_filter]),
                        # AND must match all metadata filters
                        *[FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=str(value))
                        ) for key, value in filter_metadata.items() if value is not None]
                    ]
                )
            else:
                # Just match document name
                filter_obj = Filter(should=filter_conditions)

            # Find matching points
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                filter=filter_obj,
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

    def delete_by_filter(self, filter_metadata: Dict[str, Any]) -> bool:
        """
        Delete points by metadata filter.

        Args:
            filter_metadata: Metadata to filter deletion (e.g., user_id)

        Returns:
            bool: True if successful
        """
        try:
            if not filter_metadata:
                return False

            # Build filter for metadata
            filter_conditions = []
            for key, value in filter_metadata.items():
                if value is not None:  # Only add non-None filters
                    filter_conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            # Convert to string for consistent matching
                            match=MatchValue(value=str(value))
                        )
                    )

            if not filter_conditions:
                return False

            filter_obj = Filter(must=filter_conditions)

            # Find all points matching filter
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                filter=filter_obj,
                limit=1000  # Adjust if needed
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

                print(
                    f"Deleted {len(point_ids)} chunks by filter from vector store")
                return True

            return True  # Return true if no points to delete

        except Exception as e:
            print(f"Error deleting by filter from Qdrant: {e}")
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
