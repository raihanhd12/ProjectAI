from typing import Optional
import os
import json
import requests
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from db.models import Document, DocumentChunk
from utils.text_extractor import TextExtractor
from utils.text_chunker import TextChunker
from utils.text_embedder import TextEmbedder
from db.vector_db import VectorDB


class DocumentProcessor:
    def __init__(self, doc_storage, collection_name: str = "documents"):
        self.doc_storage = doc_storage
        self.chunker = TextChunker()
        self.embedder = TextEmbedder()
        self.vector_db = VectorDB()
        self.collection_name = collection_name

        # Get Elasticsearch URL from environment or use default
        self.es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
        self.enable_hybrid_search = os.getenv(
            'ENABLE_HYBRID_SEARCH', 'True').lower() == 'true'

        # Initialize vector collection
        self.vector_db.create_collection(
            collection_name=self.collection_name,
            vector_size=self.embedder.get_dimension()
        )

        # Initialize Elasticsearch index if hybrid search is enabled
        if self.enable_hybrid_search:
            self._initialize_elasticsearch_index()

    def _initialize_elasticsearch_index(self):
        """Create Elasticsearch index with proper mappings if it doesn't exist."""
        try:
            # Check if index exists
            response = requests.head(f"{self.es_url}/documents")

            if response.status_code != 200:
                # Create index with mappings
                mapping = {
                    "mappings": {
                        "properties": {
                            "document_id": {"type": "integer"},
                            "chunk_index": {"type": "integer"},
                            "filename": {"type": "keyword"},
                            "content_type": {"type": "keyword"},
                            "user_id": {"type": "integer"},
                            "text": {"type": "text"}
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }

                create_response = requests.put(
                    f"{self.es_url}/documents",
                    json=mapping
                )

                if create_response.status_code in (200, 201):
                    print("✅ Created Elasticsearch index for documents")
                else:
                    print(
                        f"⚠️ Failed to create Elasticsearch index: {create_response.status_code}")
                    print(create_response.text)
        except Exception as e:
            print(f"Error initializing Elasticsearch: {e}")

    async def process_document(self, db: Session, doc_id: int) -> bool:
        try:
            # Get document metadata
            document = db.query(Document).filter(Document.id == doc_id).first()
            if not document:
                return False

            # Get file content
            file_content = self.doc_storage.get_file_content(
                document.object_name)
            if not file_content:
                return False

            # Extract text
            text = TextExtractor.extract_from_bytes(
                file_content, document.content_type)
            if not text:
                return False

            # Split into chunks
            chunks = self.chunker.split_text(text)

            # Generate embeddings
            embeddings = self.embedder.embed_texts(chunks)

            # Store in vector database with metadata
            metadata_list = [
                {
                    "document_id": document.id,
                    "chunk_index": idx,
                    "filename": document.name,
                    "content_type": document.content_type,
                    "user_id": document.user_id,
                    "text": chunk
                }
                for idx, chunk in enumerate(chunks)
            ]

            vector_ids = self.vector_db.upsert_vectors(
                collection_name=self.collection_name,
                vectors=embeddings,
                metadata_list=metadata_list
            )

            # Update database with chunks
            for idx, (chunk, vector_id) in enumerate(zip(chunks, vector_ids)):
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=idx,
                    chunk_text=chunk,
                    embedding_id=vector_id,
                    chunk_metadata={
                        "vector_id": vector_id,
                        "collection": self.collection_name
                    }
                )
                db.add(chunk_record)

            # If hybrid search is enabled, index in Elasticsearch
            if self.enable_hybrid_search:
                try:
                    # Index each chunk in Elasticsearch
                    for idx, chunk in enumerate(chunks):
                        doc_data = {
                            "document_id": document.id,
                            "chunk_index": idx,
                            "filename": document.name,
                            "content_type": document.content_type,
                            "user_id": document.user_id,
                            "text": chunk
                        }

                        # Use bulk indexing for better performance with multiple chunks
                        if idx % 10 == 0 and idx > 0:
                            print(f"Indexed {idx} chunks in Elasticsearch")

                        es_response = requests.post(
                            f"{self.es_url}/documents/_doc",
                            json=doc_data
                        )

                        if es_response.status_code not in (200, 201):
                            print(
                                f"⚠️ Elasticsearch indexing issue: {es_response.status_code}")

                except Exception as es_error:
                    print(f"Error indexing in Elasticsearch: {es_error}")
                    # Continue processing - failure in ES shouldn't fail the whole process

            db.commit()
            return True

        except Exception as e:
            db.rollback()
            print(f"Error processing document: {e}")
            return False
