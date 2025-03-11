"""
Database operations for the RAG module.
"""
import os
import sqlite3
import datetime
import json
import streamlit as st
from . import config
from . import utils


def init_db():
    """
    Initialize the SQLite database for chat history and documents.

    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure directory exists (just the parent directory of the DB file)
    utils.ensure_directories(
        [os.path.dirname(config.DB_PATH), config.VECTORDB_PATH])

    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()

        # Create chats table
        c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            thinking_content TEXT,
            query_results TEXT,
            relevant_text_ids TEXT,
            relevant_text TEXT
        )
        ''')

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
        st.error(f"Database initialization error: {e}")
        return False


def get_session_id():
    """
    Get or create a session ID.

    Returns:
        str: Session ID
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return st.session_state.session_id


def save_message(role, content, thinking_content=None, query_results=None, relevant_text_ids=None, relevant_text=None):
    """
    Save a message to the database.

    Args:
        role (str): Message role (user or assistant)
        content (str): Message content
        thinking_content (str, optional): AI thinking process
        query_results (dict, optional): Query results from vector store
        relevant_text_ids (list, optional): IDs of relevant text chunks
        relevant_text (str, optional): Relevant text content

    Returns:
        bool: True if successful, False otherwise
    """
    session_id = get_session_id()

    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()

        # Convert non-string data to JSON string
        query_results_json = json.dumps(
            query_results) if query_results is not None else None
        relevant_text_ids_json = json.dumps(
            relevant_text_ids) if relevant_text_ids is not None else None

        c.execute("""
            INSERT INTO chats 
            (session_id, role, content, thinking_content, query_results, relevant_text_ids, relevant_text) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, role, content, thinking_content, query_results_json, relevant_text_ids_json, relevant_text))

        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Error saving message: {e}")
        return False


def save_document(name, chunks):
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
        st.error(f"Error saving document: {e}")
        return False


def delete_document(doc_name):
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
        st.error(f"Error deleting document: {e}")
        return False


def get_chat_history():
    """
    Get chat history for the current session.

    Returns:
        list: List of chat messages
    """
    session_id = get_session_id()

    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()
        c.execute(
            """SELECT role, content, thinking_content, query_results, relevant_text_ids, relevant_text 
            FROM chats WHERE session_id = ? ORDER BY timestamp""",
            (session_id,)
        )

        chat_history = []
        for row in c.fetchall():
            role, content, thinking_content, query_results, relevant_text_ids, relevant_text = row

            # Parse JSON strings back to objects if they exist
            if query_results and query_results.strip():
                try:
                    query_results = json.loads(query_results)
                except json.JSONDecodeError:
                    query_results = None

            if relevant_text_ids and relevant_text_ids.strip():
                try:
                    relevant_text_ids = json.loads(relevant_text_ids)
                except json.JSONDecodeError:
                    relevant_text_ids = None

            chat_history.append({
                "role": role,
                "content": content,
                "thinking_content": thinking_content,
                "query_results": query_results,
                "relevant_text_ids": relevant_text_ids,
                "relevant_text": relevant_text
            })

        conn.close()
        return chat_history
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []


def get_recent_chat_sessions(limit=5):
    """
    Get recent chat sessions.

    Args:
        limit (int, optional): Maximum number of sessions to return

    Returns:
        list: List of recent chat sessions
    """
    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()

        # Get unique sessions with first question and last timestamp
        c.execute("""
            SELECT 
                c.session_id, 
                MIN(c.timestamp) as start_time,
                MAX(c.timestamp) as last_time,
                (SELECT content FROM chats WHERE session_id = c.session_id AND role = 'user' ORDER BY timestamp ASC LIMIT 1) as first_question
            FROM chats c
            GROUP BY c.session_id
            ORDER BY last_time DESC
            LIMIT ?
        """, (limit,))

        recent_sessions = c.fetchall()
        conn.close()
        return recent_sessions
    except sqlite3.Error as e:
        st.error(f"Database error retrieving recent sessions: {e}")
        return []


def get_documents():
    """
    Get all documents from the database.

    Returns:
        list: List of documents
    """
    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, chunks FROM documents ORDER BY timestamp DESC")
        documents = [{"name": name, "chunks": chunks}
                     for name, chunks in c.fetchall()]
        conn.close()
        return documents
    except sqlite3.Error as e:
        st.error(f"Database error retrieving documents: {e}")
        return []


def delete_chat_session(session_id):
    """
    Delete a chat session from the database.

    Args:
        session_id (str): Session ID

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM chats WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Error deleting session: {e}")
        return False


def reset_vector_database():
    """
    Reset the vector database by clearing all documents.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import shutil
        shutil.rmtree(config.VECTORDB_PATH)
        utils.ensure_directories([config.VECTORDB_PATH])

        # Clear the documents table
        conn = sqlite3.connect(config.DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM documents")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error resetting vector database: {e}")
        return False
