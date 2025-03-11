"""
Dependency injection for FastAPI routes.
"""
from fastapi import Header, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
from sqlalchemy.orm import Session

import config
from db import get_db
from db.models import User
from utils.auth_utils import decode_access_token

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Get current authenticated user.

    Args:
        token: JWT token
        db: Database session

    Returns:
        User object
    """
    token_data = decode_access_token(token)
    user = db.query(User).filter(User.username == token_data.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user


async def verify_token(x_api_key: Optional[str] = Header(None)):
    """
    Verify API token for authenticated endpoints.

    Args:
        x_api_key: API token from header

    Returns:
        Validated token if successful

    Raises:
        HTTPException: If token validation fails
    """
    # Skip token validation if not enabled
    if not config.ENABLE_TOKEN_AUTH:
        return "token_validation_disabled"

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key is missing")

    if x_api_key != config.API_ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key

# Export both dependencies for convenience
get_db_session = get_db
verify_api_token = verify_token
