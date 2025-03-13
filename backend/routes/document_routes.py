"""
Updated routes to handle per-user vector database reset.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import List, Optional
import shutil
import json
from sqlalchemy.orm import Session

# Import models
from models.hybrid_rag_model import HybridRAGModel
from db import document_db
from db.models import User, Document
from utils.document_storage import DocumentStorage

# Import dependencies
from dependencies import verify_api_token, get_db_session, get_current_user

# Create router
router = APIRouter()

# Initialize RAG model
rag_model = HybridRAGModel()

# Initialize document storage
doc_storage = DocumentStorage()


@router.get("/")
async def get_documents(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get all indexed documents for the current user.

    Returns:
        List of documents with metadata
    """
    try:
        documents = document_db.get_documents(db, user_id=current_user.id)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get documents: {str(e)}")


@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process documents for the current user.

    Args:
        files: List of files to upload
        background_tasks: FastAPI background tasks for async processing
        db: Database session
        current_user: Authenticated user

    Returns:
        Processing results
    """
    results = []

    for file in files:
        try:
            # Check file type
            if not file.filename.lower().endswith('.pdf'):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Only PDF files are supported"
                })
                continue

            # Upload file to object storage
            success, object_name = await doc_storage.upload_fastapi_file(
                file,
                metadata={"user_id": str(current_user.id)}
            )

            if not success:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Failed to upload file to storage"
                })
                continue

            # Create temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                # Reset file position
                await file.seek(0)

                # Copy uploaded file to temp file
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name

            # Process document (this can be moved to background task if needed)
            chunks_added = rag_model.add_documents(
                temp_path,
                file.filename,
                # Add user_id to metadata
                # Hilangkan named parameter metadata=
                {"user_id": current_user.id}
            )

            # Save document metadata to DB
            document_db.save_document(
                db,
                file.filename,
                chunks_added,
                user_id=current_user.id,
                object_name=object_name,
                file_size=len(content),
                content_type=file.content_type
            )

            # Clean up temp file
            os.unlink(temp_path)

            # Generate time-limited access URL
            download_url = doc_storage.get_file_url(object_name, expires=3600)

            results.append({
                "filename": file.filename,
                "status": "success",
                "chunks_added": chunks_added,
                "object_name": object_name,
                "download_url": download_url
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })

    return {"results": results}


@router.delete("/{document_name}")
async def delete_document(
    document_name: str,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a document from the system.

    Args:
        document_name: Name of the document to delete
        db: Database session
        current_user: Authenticated user

    Returns:
        Deletion status
    """
    try:
        # Get document to find object_name
        documents = document_db.get_documents(db, user_id=current_user.id)
        document = next(
            (doc for doc in documents if doc["name"] == document_name), None)

        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_name} not found or you don't have access"
            )

        # Delete from vector and elastic databases - with user_id for filtering
        vector_store_success = rag_model.delete_document(
            document_name,
            user_id=current_user.id
        )

        # Delete from storage if object_name exists
        storage_success = True
        if document.get("object_name"):
            storage_success = doc_storage.delete_file(document["object_name"])

        # Delete from database
        db_success = document_db.delete_document(
            db, document_name, user_id=current_user.id)

        if vector_store_success and db_success and storage_success:
            return {"status": "success", "message": f"Document {document_name} deleted successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Partial failure deleting document: Vector stores: {vector_store_success}, Storage: {storage_success}, Database: {db_success}"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.delete("/")
async def reset_vector_database(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Reset the vector database by clearing all documents for the current user.

    Returns:
        Reset status
    """
    global rag_model  # Global declaration at the beginning of the function

    try:
        # Get list of user's documents
        documents = document_db.get_documents(db, user_id=current_user.id)

        # Delete each document from vector stores and object storage
        for doc in documents:
            # Delete from vector and elastic databases with user filtering
            if doc.get("name"):
                rag_model.delete_document(doc["name"], user_id=current_user.id)

            # Delete from object storage
            if doc.get("object_name"):
                doc_storage.delete_file(doc["object_name"])

        # Reset document database for this user only
        db_success = document_db.reset_vector_database(
            db, user_id=current_user.id)

        if db_success:
            return {"status": "success", "message": f"Documents for user {current_user.username} have been reset"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reset user documents in database"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset user's vector database: {str(e)}")
