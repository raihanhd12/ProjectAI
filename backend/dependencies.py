"""
Dependency injection for FastAPI routes.
"""
from fastapi import Header, HTTPException, Depends
from typing import Optional
from sqlalchemy.orm import Session

import config
from db import get_db


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
