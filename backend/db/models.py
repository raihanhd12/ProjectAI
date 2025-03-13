# db/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func, Boolean, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from passlib.hash import bcrypt
from db import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    documents = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship(
        "UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")

    # Methods
    def verify_password(self, password):
        return bcrypt.verify(password, self.password_hash)

    @staticmethod
    def get_password_hash(password):
        return bcrypt.hash(password)


class TokenBlacklist(Base):
    __tablename__ = "token_blacklist"
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(500), nullable=False, index=True)
    blacklisted_on = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey(
        "users.id"), nullable=True, index=True)
    object_name = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    content_type = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    # Add this method
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "user_id": self.user_id,
            "object_name": self.object_name,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_id = Column(String(255), nullable=True)
    chunk_metadata = Column(JSON, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey(
        "users.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    title = Column(String(255), nullable=True)

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), ForeignKey(
        "chat_sessions.session_id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    thinking_content = Column(Text, nullable=True)
    query_results = Column(Text, nullable=True)
    relevant_text_ids = Column(Text, nullable=True)
    relevant_text = Column(Text, nullable=True)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class VectorIndex(Base):
    __tablename__ = "vector_indexes"
    id = Column(Integer, primary_key=True, index=True)
    collection_name = Column(String(255), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    dimension = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"),
                     nullable=False, unique=True)
    theme = Column(String(20), nullable=True)
    language = Column(String(10), nullable=True)
    model_preference = Column(String(50), nullable=True)
    settings = Column(JSON, nullable=True)

    # Relationships
    user = relationship("User", back_populates="preferences")
