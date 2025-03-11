"""
Document database operations with MySQL database.
"""
import os
import shutil
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

import config
from db.models import Document
from utils import helpers


def save_document(db: Session, name: str, chunks: int) -> bool:
    """
    Save document metadata to the database.

    Args:
        db: Database session
        name: Document name
        chunks: Number of text chunks

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if document already exists
        existing_doc = db.query(Document).filter(Document.name == name).first()
        if existing_doc:
            # Update existing document
            existing_doc.chunks = chunks
            db.commit()
            return True

        # Create new document record
        new_doc = Document(name=name, chunks=chunks)
        db.add(new_doc)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error saving document: {e}")
        return False


def delete_document(db: Session, doc_name: str) -> bool:
    """
    Delete a document from the database.

    Args:
        db: Database session
        doc_name: Document name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find the document
        doc = db.query(Document).filter(Document.name == doc_name).first()
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


def get_documents(db: Session) -> List[Dict[str, Any]]:
    """
    Get all documents from the database.

    Args:
        db: Database session

    Returns:
        List[Dict[str, Any]]: List of documents
    """
    try:
        documents = db.query(Document).order_by(desc(Document.timestamp)).all()
        return [doc.to_dict() for doc in documents]
    except Exception as e:
        print(f"Database error retrieving documents: {e}")
        return []


def reset_vector_database(db: Session) -> bool:
    """
    Reset the vector database by clearing all documents.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Clear the vector database directory
        if os.path.exists(config.VECTORDB_PATH):
            shutil.rmtree(config.VECTORDB_PATH)
        os.makedirs(config.VECTORDB_PATH, exist_ok=True)

        # Clear the documents table
        db.query(Document).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error resetting vector database: {e}")
        return False
