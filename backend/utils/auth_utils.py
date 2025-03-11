"""
Authentication utilities.
"""
from datetime import datetime, timedelta
from typing import Optional
import jwt
from fastapi import HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

import config


class TokenData(BaseModel):
    username: str
    exp: datetime


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create JWT access token.

    Args:
        data: Data to encode
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str, db: Session = None):
    """
    Decode JWT access token.

    Args:
        token: JWT token
        db: Optional database session to check blacklist

    Returns:
        TokenData object

    Raises:
        HTTPException: If token validation fails
    """
    try:
        # Check if token is blacklisted
        if db and is_token_blacklisted(db, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )

        payload = jwt.decode(token, config.JWT_SECRET_KEY,
                             algorithms=[config.JWT_ALGORITHM])
        username = payload.get("sub")
        exp = payload.get("exp")

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return TokenData(username=username, exp=exp)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def blacklist_token(db: Session, token: str, token_data: TokenData):
    """
    Add a token to the blacklist.

    Args:
        db: Database session
        token: JWT token
        token_data: Decoded token data with expiration

    Returns:
        bool: True if successful
    """
    from db.models import TokenBlacklist

    blacklist_token = TokenBlacklist(
        token=token,
        expires_at=datetime.fromtimestamp(token_data.exp)
    )

    try:
        db.add(blacklist_token)
        db.commit()
        return True
    except:
        db.rollback()
        return False


def is_token_blacklisted(db: Session, token: str):
    """
    Check if token is blacklisted.

    Args:
        db: Database session
        token: JWT token

    Returns:
        bool: True if blacklisted
    """
    from db.models import TokenBlacklist

    return db.query(TokenBlacklist).filter(TokenBlacklist.token == token).first() is not None
