"""
SQLAlchemy models for the database.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from datetime import datetime

from db import Base


class Document(Base):
    """Document model for indexed documents."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    chunks = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "chunks": self.chunks,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class ChatSession(Base):
    """Chat session model."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to messages
    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None
        }


class ChatMessage(Base):
    """Chat message model."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), ForeignKey(
        "chat_sessions.session_id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    thinking_content = Column(Text, nullable=True)
    query_results = Column(Text, nullable=True)  # JSON string
    relevant_text_ids = Column(Text, nullable=True)  # JSON string
    relevant_text = Column(Text, nullable=True)

    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "role": self.role,
            "content": self.content,
            "thinking_content": self.thinking_content,
            "query_results": self.query_results,
            "relevant_text_ids": self.relevant_text_ids,
            "relevant_text": self.relevant_text
        }
