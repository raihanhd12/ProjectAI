"""
Document database operations.
"""
import os
import sqlite3
import shutil
from typing import List, Dict, Any, Optional

import config
from utils import helpers


def init_db() -> bool:
    """
    Initialize the SQLite database for documents.

    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)

    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()

        # Create documents table
        c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            chunks INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        return False


def save_document(name: str, chunks: int) -> bool:
    """
    Save document metadata to the database.

    Args:
        name (str): Document name
        chunks (int): Number of text chunks

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO documents (name, chunks) VALUES (?, ?)", (name, chunks))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error saving document: {e}")
        return False


def delete_document(doc_name: str) -> bool:
    """
    Delete a document from the database.

    Args:
        doc_name (str): Document name

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM documents WHERE name = ?", (doc_name,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error deleting document: {e}")
        return False


def get_documents() -> List[Dict[str, Any]]:
    """
    Get all documents from the database.

    Returns:
        List[Dict[str, Any]]: List of documents
    """
    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, chunks, timestamp FROM documents ORDER BY timestamp DESC")
        documents = [{"name": name, "chunks": chunks, "timestamp": timestamp}
                     for name, chunks, timestamp in c.fetchall()]
        conn.close()
        return documents
    except sqlite3.Error as e:
        print(f"Database error retrieving documents: {e}")
        return []


def reset_vector_database() -> bool:
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
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM documents")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error resetting vector database: {e}")
        return False