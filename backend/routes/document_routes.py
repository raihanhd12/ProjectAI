"""
Document management routes for the AI Document Assistant API.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
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

# Import dependencies
from dependencies import verify_api_token, get_db_session

# Create router
router = APIRouter()

# Initialize RAG model
rag_model = HybridRAGModel()


@router.get("/")
async def get_documents(
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Get all indexed documents.

    Returns:
        List of documents with metadata
    """
    try:
        documents = document_db.get_documents(db)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get documents: {str(e)}")


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Upload and process documents.

    Args:
        files: List of files to upload

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

            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                # Copy uploaded file to temp file
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name

            # Process document
            chunks_added = rag_model.add_document(temp_path, file.filename)

            # Save document metadata to DB
            document_db.save_document(db, file.filename, chunks_added)

            # Clean up temp file
            os.unlink(temp_path)

            results.append({
                "filename": file.filename,
                "status": "success",
                "chunks_added": chunks_added
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
    token: str = Depends(verify_api_token)
):
    """
    Delete a document from the system.

    Args:
        document_name: Name of the document to delete

    Returns:
        Deletion status
    """
    try:
        # Delete from vector and elastic databases
        vector_store_success = rag_model.delete_document(document_name)

        # Delete from database
        db_success = document_db.delete_document(db, document_name)

        if vector_store_success and db_success:
            return {"status": "success", "message": f"Document {document_name} deleted successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Partial failure deleting document: Vector stores: {vector_store_success}, Database: {db_success}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.delete("/")
async def reset_vector_database(
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Reset the vector database by clearing all documents.

    Returns:
        Reset status
    """
    try:
        # Reset vector and elastic databases
        vector_success = rag_model.reset_databases()

        # Reset document database
        db_success = document_db.reset_vector_database(db)

        if vector_success and db_success:
            # Re-initialize the RAG model
            global rag_model
            rag_model = HybridRAGModel()
            return {"status": "success", "message": "Vector databases have been reset"}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Partial failure resetting databases: Vector stores: {vector_success}, Database: {db_success}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset vector databases: {str(e)}")
