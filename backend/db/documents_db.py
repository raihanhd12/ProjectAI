"""
Document database operations with MySQL database.
"""
import os
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from db.models import Document, User


def save_document(db: Session, name: str, user_id: Optional[int] = None,
                  object_name: Optional[str] = None, file_size: Optional[int] = None,
                  content_type: Optional[str] = None) -> Optional[int]:
    """
    Save document metadata to the database.

    Returns:
        int: Document ID if successful, None otherwise
    """
    try:
        # Check if document already exists for this user
        query = db.query(Document).filter(Document.name == name)
        if user_id:
            query = query.filter(Document.user_id == user_id)

        existing_doc = query.first()

        if existing_doc:
            # Update existing document
            if object_name:
                existing_doc.object_name = object_name
            if file_size:
                existing_doc.file_size = file_size
            if content_type:
                existing_doc.content_type = content_type
            db.commit()
            return existing_doc.id

        # Create new document record
        new_doc = Document(
            name=name,
            user_id=user_id,
            object_name=object_name,
            file_size=file_size,
            content_type=content_type
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        return new_doc.id

    except Exception as e:
        db.rollback()
        print(f"Error saving document: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def delete_vectors(self, collection_name: str, filter_conditions: Optional[Dict] = None, vector_ids: Optional[List[str]] = None) -> bool:
    """
    Delete vectors from the collection based on filter or IDs.

    Args:
        collection_name: Name of the collection
        filter_conditions: Filter conditions to match vectors to delete
        vector_ids: List of vector IDs to delete

    Returns:
        bool: True if deletion was successful
    """
    try:
        if vector_ids:
            # Delete by IDs
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=vector_ids
                )
            )
            return True
        elif filter_conditions:
            # Delete by filter
            filter_query = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
            )

            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=filter_query
                )
            )
            return True
        else:
            print("Error: Either vector_ids or filter_conditions must be provided")
            return False
    except Exception as e:
        print(f"Error deleting vectors: {e}")
        return False


def get_documents(db: Session, user_id: Optional[int] = None):
    try:
        query = db.query(Document).order_by(desc(Document.timestamp))

        # Add debug printing
        print(f"Looking for documents with user_id: {user_id}")

        if user_id:
            query = query.filter(Document.user_id == user_id)

        documents = query.all()
        print(f"Found {len(documents)} documents")

        # Check if to_dict() is implemented
        return [doc.to_dict() for doc in documents]
    except Exception as e:
        print(f"Database error retrieving documents: {e}")
        return []


def get_document_by_id(db: Session, doc_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Get a document by ID.

    Args:
        db: Database session
        doc_id: Document ID
        user_id: Optional user ID to restrict to user's documents

    Returns:
        Optional[Dict[str, Any]]: Document data or None if not found
    """
    try:
        query = db.query(Document).filter(Document.id == doc_id)

        # Filter by user if provided
        if user_id:
            query = query.filter(Document.user_id == user_id)

        document = query.first()
        return document.to_dict() if document else None
    except Exception as e:
        print(f"Database error retrieving document: {e}")
        return None
