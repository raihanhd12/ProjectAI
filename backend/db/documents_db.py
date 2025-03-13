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


def delete_document(db: Session, doc_name: str, user_id: Optional[int] = None) -> bool:
    """
    Delete a document from the database.

    Args:
        db: Database session
        doc_name: Document name
        user_id: Optional user ID to restrict to user's documents

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find the document
        query = db.query(Document).filter(Document.name == doc_name)
        if user_id:
            query = query.filter(Document.user_id == user_id)

        doc = query.first()
        if not doc:
            return False

        # Delete the document
        db.delete(doc)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error deleting document: {e}")
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
