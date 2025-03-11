"""
Health and monitoring routes for the AI Document Assistant API.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.dependencies import get_db_session
import config

# Create router
router = APIRouter()


@router.get("/")
async def health_check(db: Session = Depends(get_db_session)):
    """
    Health check endpoint.

    Returns:
        Health status of the API and its dependencies
    """
    try:
        # Check database connection
        db_status = "healthy"
        try:
            # Execute a simple query to check database connection
            db.execute("SELECT 1")
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"

        # Check vector DB path
        vector_db_path = config.VECTORDB_PATH
        vector_db_status = "healthy" if vector_db_path and import_chromadb_client() else "unhealthy"

        # Check model configuration
        model_status = "healthy" if config.DEFAULT_LLM_MODEL and config.DEFAULT_EMBEDDING_MODEL else "misconfigured"

        # Overall status
        overall_status = "healthy" if all(s == "healthy" for s in [
                                          db_status, vector_db_status, model_status]) else "degraded"

        return {
            "status": overall_status,
            "components": {
                "api": "healthy",
                "database": db_status,
                "vector_db": vector_db_status,
                "models": model_status
            },
            "config": {
                "llm_model": config.DEFAULT_LLM_MODEL,
                "embedding_model": config.DEFAULT_EMBEDDING_MODEL,
                "vector_db_path": config.VECTORDB_PATH,
                "auth_enabled": config.ENABLE_TOKEN_AUTH
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def import_chromadb_client():
    """
    Import ChromaDB client to check if it can be initialized.

    Returns:
        bool: True if ChromaDB can be imported, False otherwise
    """
    try:
        import chromadb
        return True
    except ImportError:
        return False
