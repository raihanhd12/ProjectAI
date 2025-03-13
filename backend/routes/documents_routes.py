"""
Document routes for the ToolXpert API.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Response
from fastapi.responses import RedirectResponse
import os
from typing import List
from sqlalchemy.orm import Session

from db import documents_db
from db.models import User, Document
from utils.documents_storage import DocumentStorage
from dependencies import get_current_user, get_db_session

# Create router
router = APIRouter()

# Initialize document storage
doc_storage = DocumentStorage()


@router.get("/")
async def get_documents(
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get all documents for the current user.

    Returns:
        List of documents with metadata
    """
    try:
        documents = documents_db.get_documents(db, user_id=current_user.id)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get documents: {str(e)}")


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    try:
        # Save file to MinIO
        success, object_name = await doc_storage.upload_file(
            file,
            metadata={"user_id": str(current_user.id)}
        )

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to upload file to storage")

        # Get file size - perbaiki bagian ini
        file_size = 0
        try:
            # Reposition file pointer to beginning
            await file.seek(0)
            # Read file content
            content = await file.read()
            file_size = len(content)
        except Exception as e:
            print(f"Error getting file size: {e}")
            file_size = 0  # Default if can't determine size

        # Save metadata to database
        documents_db.save_document(
            db,
            file.filename,
            user_id=current_user.id,
            object_name=object_name,
            file_size=file_size,
            content_type=file.content_type
        )
        save_result = documents_db.save_document(
            db,
            file.filename,
            user_id=current_user.id,
            object_name=object_name,
            file_size=file_size,
            content_type=file.content_type
        )
        if not save_result:
            raise HTTPException(
                status_code=500, detail="Failed to save document metadata to database")

        # Generate download URL
        try:
            download_url = doc_storage.get_file_url(object_name, expires=3600)
        except Exception as e:
            print(f"Error generating URL: {e}")
            download_url = None

        return {
            "filename": file.filename,
            "size": file_size,
            "object_name": object_name,
            "content_type": file.content_type,
            "download_url": download_url,
            "message": "File uploaded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error details: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a document.

    Args:
        doc_id: Document ID

    Returns:
        Deletion status
    """
    try:
        # Get document metadata
        document = documents_db.get_document_by_id(db, doc_id, current_user.id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Delete from MinIO
        object_name = document.get("object_name")
        if object_name:
            doc_storage.delete_file(object_name)

        # Delete from database
        query = db.query(Document).filter(Document.id == doc_id)
        if current_user.id:
            query = query.filter(Document.user_id == current_user.id)

        doc = query.first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        db.delete(doc)
        db.commit()

        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Delete failed: {str(e)}")
