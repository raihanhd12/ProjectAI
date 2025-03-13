"""
Document routes for the ToolXpert API.
"""
from dependencies import get_current_user, get_db_session
from utils.documents_storage import DocumentStorage
from db.models import User, Document
from db import documents_db
from sqlalchemy.orm import Session
from typing import List
import os
from fastapi.responses import RedirectResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Response, BackgroundTasks, Form
from typing import List, Optional


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
    background_tasks: BackgroundTasks,
    # This expects a form field named "files"
    files: List[UploadFile] = File(...),
    process: bool = Form(True),  # Use Form() for form data
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    results = []

    for file in files:
        try:
            # Save file to MinIO
            success, object_name = await doc_storage.upload_file(
                file,
                metadata={"user_id": str(current_user.id)}
            )

            if not success:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "message": "Failed to upload file to storage"
                })
                continue

            # Get file size
            file_size = 0
            try:
                await file.seek(0)
                content = await file.read()
                file_size = len(content)
            except Exception as e:
                print(f"Error getting file size: {e}")
                file_size = 0

            # Save metadata to database
            document_id = documents_db.save_document(
                db,
                file.filename,
                user_id=current_user.id,
                object_name=object_name,
                file_size=file_size,
                content_type=file.content_type
            )

            if not document_id:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "message": "Failed to save document metadata to database"
                })
                continue

            # Process document if requested
            if process and document_id:
                from utils.document_processor import DocumentProcessor
                processor = DocumentProcessor(doc_storage)

                # Process in background
                background_tasks.add_task(
                    processor.process_document,
                    db=db,
                    doc_id=document_id
                )

            # Generate download URL
            download_url = None
            try:
                download_url = doc_storage.get_file_url(
                    object_name, expires=3600)
            except Exception as e:
                print(f"Error generating URL: {e}")

            results.append({
                "filename": file.filename,
                "success": True,
                "size": file_size,
                "object_name": object_name,
                "content_type": file.content_type,
                "download_url": download_url,
                "message": "File uploaded successfully"
            })

        except Exception as e:
            print(f"Upload error details for {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "success": False,
                "message": f"Upload failed: {str(e)}"
            })

    return {"files": results}


@router.get("/search")
async def search_documents(
    query: str,
    limit: int = 5,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    try:
        # Initialize components
        from utils.text_embedder import TextEmbedder
        from db.vector_db import VectorDB

        embedder = TextEmbedder()
        vector_db = VectorDB()

        # Create query embedding
        query_embedding = embedder.embed_text(query)

        # Search vector database with user filter
        results = vector_db.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=limit,
            filter_conditions={"user_id": current_user.id}
        )

        # Format and return results
        return {
            "query": query,
            "results": [
                {
                    "document_id": result["metadata"]["document_id"],
                    "document_name": result["metadata"]["filename"],
                    "score": result["score"],
                    "text": result["metadata"]["text"]
                }
                for result in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


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
