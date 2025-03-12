"""
Health and monitoring routes for the AI Document Assistant API.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import requests

from dependencies import get_db_session
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

        # Check search store status
        if config.USE_HYBRID_SEARCH:
            # Check Qdrant
            qdrant_status = check_qdrant_status()

            # Check Elasticsearch
            es_status = check_elasticsearch_status()

            # Combined status
            vector_db_status = "healthy" if qdrant_status == "healthy" and es_status == "healthy" else "degraded"
        else:
            # Check ChromaDB
            vector_db_status = "healthy" if config.VECTORDB_PATH and import_chromadb_client() else "unhealthy"

        # Check model configuration
        model_status = "healthy" if config.DEFAULT_LLM_MODEL and config.DEFAULT_EMBEDDING_MODEL else "misconfigured"

        # Overall status
        all_statuses = [db_status, vector_db_status, model_status]
        if config.USE_HYBRID_SEARCH:
            all_statuses.extend([qdrant_status, es_status])

        overall_status = "healthy" if all(
            s == "healthy" for s in all_statuses) else "degraded"

        return {
            "status": overall_status,
            "components": {
                "api": "healthy",
                "database": db_status,
                "search_store": {
                    "type": "hybrid" if config.USE_HYBRID_SEARCH else "chromadb",
                    "status": vector_db_status,
                    "qdrant": qdrant_status if config.USE_HYBRID_SEARCH else None,
                    "elasticsearch": es_status if config.USE_HYBRID_SEARCH else None
                },
                "models": model_status
            },
            "config": {
                "llm_model": config.DEFAULT_LLM_MODEL,
                "embedding_model": config.DEFAULT_EMBEDDING_MODEL,
                "vector_db_path": config.VECTORDB_PATH if not config.USE_HYBRID_SEARCH else None,
                "qdrant": f"{config.QDRANT_HOST}:{config.QDRANT_PORT}" if config.USE_HYBRID_SEARCH else None,
                "elasticsearch": f"{config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}" if config.USE_HYBRID_SEARCH else None,
                "auth_enabled": config.ENABLE_TOKEN_AUTH,
                "semantic_weight": config.SEMANTIC_WEIGHT if config.USE_HYBRID_SEARCH else None
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_qdrant_status():
    """
    Check Qdrant connection status.

    Returns:
        str: Status description
    """
    try:
        # Try to connect to Qdrant and check health
        response = requests.get(
            f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/healthz")
        if response.status_code == 200:
            return "healthy"
        else:
            return f"unhealthy: Status code {response.status_code}"
    except Exception as e:
        return f"unhealthy: {str(e)}"


def check_elasticsearch_status():
    """
    Check Elasticsearch connection status.

    Returns:
        str: Status description
    """
    try:
        # Try to connect to Elasticsearch and check health
        response = requests.get(
            f"http://{config.ELASTICSEARCH_HOST}:{config.ELASTICSEARCH_PORT}")
        # 401 is ok, just means authentication is required
        if response.status_code in [200, 401]:
            return "healthy"
        else:
            return f"unhealthy: Status code {response.status_code}"
    except Exception as e:
        return f"unhealthy: {str(e)}"


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
