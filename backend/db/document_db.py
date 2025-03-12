"""
Document database operations with MySQL database.
"""
import os
import shutil
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

import config
from db.models import Document, User
from utils import helpers


def save_document(db: Session, name: str, chunks: int, user_id: Optional[int] = None,
                  object_name: Optional[str] = None, file_size: Optional[int] = None,
                  content_type: Optional[str] = None) -> bool:
    """
    Save document metadata to the database.

    Args:
        db: Database session
        name: Document name
        chunks: Number of text chunks
        user_id: ID of the user who owns this document
        object_name: Object name in storage
        file_size: Size of the file in bytes
        content_type: Content type of the file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if document already exists for this user
        query = db.query(Document).filter(Document.name == name)
        if user_id:
            query = query.filter(Document.user_id == user_id)

        existing_doc = query.first()

        if existing_doc:
            # Update existing document
            existing_doc.chunks = chunks
            if object_name:
                existing_doc.object_name = object_name
            if file_size:
                existing_doc.file_size = file_size
            if content_type:
                existing_doc.content_type = content_type
            db.commit()
            return True

        # Create new document record
        new_doc = Document(
            name=name,
            chunks=chunks,
            user_id=user_id,
            object_name=object_name,
            file_size=file_size,
            content_type=content_type
        )
        db.add(new_doc)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error saving document: {e}")
        return False


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


def get_documents(db: Session, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all documents from the database.

    Args:
        db: Database session
        user_id: Optional user ID to restrict to user's documents

    Returns:
        List[Dict[str, Any]]: List of documents
    """
    try:
        query = db.query(Document).order_by(desc(Document.timestamp))

        # Filter by user if provided
        if user_id:
            query = query.filter(Document.user_id == user_id)

        documents = query.all()
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


def reset_vector_database(db: Session, user_id: Optional[int] = None) -> bool:
    """
    Reset the vector database by clearing all documents.

    Args:
        db: Database session
        user_id: Optional user ID to restrict to user's documents

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Clear the vector database directory
        if os.path.exists(config.VECTORDB_PATH):
            shutil.rmtree(config.VECTORDB_PATH)
        os.makedirs(config.VECTORDB_PATH, exist_ok=True)

        # Clear the documents table
        query = db.query(Document)
        if user_id:
            query = query.filter(Document.user_id == user_id)

        query.delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error resetting vector database: {e}")
        return False


def get_document_count(db: Session, user_id: Optional[int] = None) -> int:
    """
    Get the count of documents.

    Args:
        db: Database session
        user_id: Optional user ID to restrict to user's documents

    Returns:
        int: Count of documents
    """
    try:
        query = db.query(Document)
        if user_id:
            query = query.filter(Document.user_id == user_id)

        return query.count()
    except Exception as e:
        print(f"Database error getting document count: {e}")
        return 0
