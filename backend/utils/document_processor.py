# utils/document_processor.py
from typing import Optional
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

        # Initialize collection
        self.vector_db.create_collection(
            collection_name=self.collection_name,
            vector_size=self.embedder.get_dimension()
        )

    async def process_document(self, db: Session, doc_id: int) -> bool:
        try:
            # Get document metadata
            document = db.query(Document).filter(Document.id == doc_id).first()
            if not document:
                print(f"Document not found: {doc_id}")
                return False

            # Get file content
            file_content = self.doc_storage.get_file_content(
                document.object_name)
            if not file_content:
                print(
                    f"Could not retrieve file content for document: {doc_id}")
                return False

            # Extract text
            text = TextExtractor.extract_from_bytes(
                file_content, document.content_type)
            if not text:
                print(f"Could not extract text from document: {doc_id}")
                return False

            # Split into chunks
            chunks = self.chunker.split_text(text)
            if not chunks:
                print(f"No chunks generated for document: {doc_id}")
                return False

            print(f"Generated {len(chunks)} chunks for document: {doc_id}")

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

            if not vector_ids:
                print(f"Failed to insert vectors for document: {doc_id}")
                return False

            print(
                f"Successfully inserted {len(vector_ids)} vectors for document: {doc_id}")

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

            db.commit()
            print(f"Document {doc_id} successfully processed")
            return True

        except Exception as e:
            db.rollback()
            print(f"Error processing document {doc_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
