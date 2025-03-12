"""
Chat database operations with MySQL database, now with user associations.
"""
import datetime
import json
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from db import engine, Base
from db.models import ChatSession, ChatMessage


def init_db():
    """
    Initialize the database by creating tables if they don't exist.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False


def create_session(db: Session, user_id: Optional[int] = None, session_id: Optional[str] = None) -> str:
    """
    Create a new chat session.

    Args:
        db: Database session
        user_id: User ID to associate with the session
        session_id: Custom session ID (optional)

    Returns:
        str: Session ID
    """
    if not session_id:
        session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Check if session already exists
    existing_session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id).first()
    if existing_session:
        return session_id

    # Create new session
    new_session = ChatSession(session_id=session_id, user_id=user_id)
    db.add(new_session)
    db.commit()
    return session_id


def save_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    thinking_content: Optional[str] = None,
    query_results: Optional[Dict] = None,
    relevant_text_ids: Optional[List] = None,
    relevant_text: Optional[str] = None,
    user_id: Optional[int] = None
) -> bool:
    """
    Save a message to the database.

    Args:
        db: Database session
        session_id: Session ID
        role: Message role (user or assistant)
        content: Message content
        thinking_content: AI thinking process (optional)
        query_results: Query results from vector store (optional)
        relevant_text_ids: IDs of relevant text chunks (optional)
        relevant_text: Relevant text content (optional)
        user_id: User ID to associate with the session (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if session exists
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id).first()
        if not session:
            # Create session if it doesn't exist
            session_id = create_session(db, user_id, session_id)
            session = db.query(ChatSession).filter(
                ChatSession.session_id == session_id).first()

        # Update user_id if provided and not already set
        if user_id and not session.user_id:
            session.user_id = user_id
            db.commit()

        # Convert complex objects to JSON strings
        query_results_json = json.dumps(
            query_results) if query_results is not None else None
        relevant_text_ids_json = json.dumps(
            relevant_text_ids) if relevant_text_ids is not None else None

        # Create new message
        new_message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            thinking_content=thinking_content,
            query_results=query_results_json,
            relevant_text_ids=relevant_text_ids_json,
            relevant_text=relevant_text
        )

        db.add(new_message)
        db.commit()

        # Update session's last_message_at
        session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id).first()
        session.last_message_at = datetime.datetime.utcnow()
        db.commit()

        return True
    except Exception as e:
        db.rollback()
        print(f"Error saving message: {e}")
        return False


def get_chat_history(db: Session, session_id: str) -> List[Dict[str, Any]]:
    """
    Get chat history for a session.

    Args:
        db: Database session
        session_id: Session ID

    Returns:
        List[Dict[str, Any]]: List of chat messages
    """
    try:
        # Get all messages for the session
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp).all()

        chat_history = []
        for message in messages:
            # Parse JSON strings back to objects
            query_results = None
            if message.query_results and message.query_results.strip():
                try:
                    query_results = json.loads(message.query_results)
                except json.JSONDecodeError:
                    query_results = None

            relevant_text_ids = None
            if message.relevant_text_ids and message.relevant_text_ids.strip():
                try:
                    relevant_text_ids = json.loads(message.relevant_text_ids)
                except json.JSONDecodeError:
                    relevant_text_ids = None

            chat_history.append({
                "role": message.role,
                "content": message.content,
                "thinking_content": message.thinking_content,
                "query_results": query_results,
                "relevant_text_ids": relevant_text_ids,
                "relevant_text": message.relevant_text
            })

        return chat_history
    except Exception as e:
        print(f"Database error: {e}")
        return []


def get_recent_chat_sessions(db: Session, user_id: Optional[int] = None, limit: int = 5) -> List[Tuple]:
    """
    Get recent chat sessions.

    Args:
        db: Database session
        user_id: Optional user ID to filter sessions
        limit: Maximum number of sessions to return

    Returns:
        List[Tuple]: List of recent chat sessions
    """
    try:
        # Build the query
        query = db.query(
            ChatSession.session_id,
            ChatSession.created_at,
            ChatSession.last_message_at,
            ChatSession.user_id,
            # Subquery to get the first user message for each session
            db.query(ChatMessage.content)
            .filter(
                ChatMessage.session_id == ChatSession.session_id,
                ChatMessage.role == 'user'
            )
            .order_by(ChatMessage.timestamp)
            .limit(1)
            .scalar_subquery()
            .label('first_question')
        )

        # Filter by user_id if provided
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)

        # Order and limit
        query = query.order_by(
            desc(ChatSession.last_message_at)
        ).limit(limit)

        # Execute query
        sessions = query.all()

        return [
            (
                session.session_id,
                session.created_at.isoformat() if session.created_at else None,
                session.last_message_at.isoformat() if session.last_message_at else None,
                session.first_question or "New conversation",
                session.user_id
            )
            for session in sessions
        ]
    except Exception as e:
        print(f"Database error retrieving recent sessions: {e}")
        return []


def delete_chat_session(db: Session, session_id: str, user_id: Optional[int] = None) -> bool:
    """
    Delete a chat session and all its messages.

    Args:
        db: Database session
        session_id: Session ID
        user_id: Optional user ID to ensure ownership

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find the session
        query = db.query(ChatSession).filter(
            ChatSession.session_id == session_id)

        # Add user filter if provided
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)

        session = query.first()
        if not session:
            return False

        # Delete the session (messages will be cascaded)
        db.delete(session)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error deleting session: {e}")
        return False


def get_user_chat_count(db: Session, user_id: int) -> int:
    """
    Get the number of chat sessions for a user.

    Args:
        db: Database session
        user_id: User ID

    Returns:
        int: Number of chat sessions
    """
    try:
        return db.query(ChatSession).filter(
            ChatSession.user_id == user_id).count()
    except Exception as e:
        print(f"Database error counting user chat sessions: {e}")
        return 0
