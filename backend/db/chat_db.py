"""
Chat database operations.
"""
import os
import sqlite3
import datetime
import json
from typing import List, Dict, Any, Optional, Tuple

import config
from utils import helpers


def init_db() -> bool:
    """
    Initialize the SQLite database for chat history.

    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)

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

        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        return False


def create_session(session_id: Optional[str] = None) -> str:
    """
    Create a new chat session.

    Args:
        session_id (str, optional): Custom session ID

    Returns:
        str: Session ID
    """
    if not session_id:
        session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return session_id


def save_message(
    session_id: str,
    role: str,
    content: str,
    thinking_content: Optional[str] = None,
    query_results: Optional[Dict] = None,
    relevant_text_ids: Optional[List] = None,
    relevant_text: Optional[str] = None
) -> bool:
    """
    Save a message to the database.

    Args:
        session_id (str): Session ID
        role (str): Message role (user or assistant)
        content (str): Message content
        thinking_content (str, optional): AI thinking process
        query_results (Dict, optional): Query results from vector store
        relevant_text_ids (List, optional): IDs of relevant text chunks
        relevant_text (str, optional): Relevant text content

    Returns:
        bool: True if successful, False otherwise
    """
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
        print(f"Error saving message: {e}")
        return False


def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Get chat history for a session.

    Args:
        session_id (str): Session ID

    Returns:
        List[Dict[str, Any]]: List of chat messages
    """
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
        print(f"Database error: {e}")
        return []


def get_recent_chat_sessions(limit: int = 5) -> List[Tuple]:
    """
    Get recent chat sessions.

    Args:
        limit (int, optional): Maximum number of sessions to return

    Returns:
        List[Tuple]: List of recent chat sessions
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
        print(f"Database error retrieving recent sessions: {e}")
        return []


def delete_chat_session(session_id: str) -> bool:
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
        print(f"Error deleting session: {e}")
        return False