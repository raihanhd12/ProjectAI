"""
SQLAlchemy models for the database.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from passlib.hash import bcrypt

from db import Base


class Document(Base):
    """Document model for indexed documents."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    chunks = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Add user association
    user_id = Column(Integer, ForeignKey(
        "users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="documents")

    # Add object storage reference
    object_name = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    content_type = Column(String(100), nullable=True)

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "chunks": self.chunks,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id,
            "object_name": self.object_name,
            "file_size": self.file_size,
            "content_type": self.content_type
        }


class ChatSession(Base):
    """Chat session model."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Add user association
    user_id = Column(Integer, ForeignKey(
        "users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="chat_sessions")

    # Relationship to messages
    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
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


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Add relationships
    documents = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan")

    def verify_password(self, password):
        """Verify password against stored hash."""
        return bcrypt.verify(password, self.password_hash)

    @staticmethod
    def get_password_hash(password):
        """Generate password hash."""
        return bcrypt.hash(password)

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active
        }


class TokenBlacklist(Base):
    """Model for blacklisted JWT tokens."""
    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(500), nullable=False, index=True)
    blacklisted_on = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
