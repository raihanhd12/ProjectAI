# db/vector_db.py
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid


class VectorDB:
    def __init__(self, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)

    def create_collection(self, collection_name: str, vector_size: int) -> bool:
        try:
            collections = self.client.get_collections().collections
            if collection_name in [c.name for c in collections]:
                return True

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False

    def upsert_vectors(self, collection_name: str, vectors: List[List[float]],
                       metadata_list: List[Dict[str, Any]]) -> List[str]:
        try:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
            points = [
                models.PointStruct(id=id, vector=vector, payload=metadata)
                for id, vector, metadata in zip(ids, vectors, metadata_list)
            ]

            self.client.upsert(collection_name=collection_name, points=points)
            return ids
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return []

    def search(self, collection_name: str, query_vector: List[float],
               limit: int = 5, filter_conditions: Optional[Dict] = None) -> List[Dict]:
        try:
            filter_query = None
            if filter_conditions:
                filter_query = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )

            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_query
            )

            return [
                {
                    "id": str(result.id),
                    "score": result.score,
                    "metadata": result.payload
                }
                for result in results
            ]
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
