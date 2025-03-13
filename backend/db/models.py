"""
SQLAlchemy models for the database.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from passlib.hash import bcrypt

from db import Base


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

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


class Document(Base):
    """Document model for stored documents."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Add user association
    user_id = Column(Integer, ForeignKey(
        "users.id"), nullable=True, index=True)
    user = relationship("User", back_populates="documents")

    # File storage reference
    file_path = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=True)
    content_type = Column(String(100), nullable=True)

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id,
            "file_size": self.file_size,
            "content_type": self.content_type
        }
