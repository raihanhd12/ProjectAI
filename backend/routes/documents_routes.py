"""
Document routes for the ToolXpert API.
"""
from dependencies import get_current_user, get_db_session
from utils.documents_storage import DocumentStorage
from db.models import User, Document, DocumentChunk
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
        import requests
        import json
        import config

        embedder = TextEmbedder()
        vector_db = VectorDB()

        # Determine if we should use hybrid search
        enable_hybrid = config.ENABLE_HYBRID_SEARCH if hasattr(
            config, 'ENABLE_HYBRID_SEARCH') else False
        vector_weight = float(os.getenv("VECTOR_WEIGHT", "0.7"))
        keyword_weight = float(os.getenv("KEYWORD_WEIGHT", "0.3"))

        # Get vector search results
        query_embedding = embedder.embed_texts([query])[0]
        vector_results = vector_db.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=limit * 2,  # Get more results for reranking
            filter_conditions={"user_id": current_user.id}
        )

        # If hybrid search is enabled, get keyword search results from Elasticsearch
        if enable_hybrid:
            try:
                # Query Elasticsearch
                es_url = f"{os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')}/documents/_search"
                es_query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"text": query}},
                                {"term": {"user_id": current_user.id}}
                            ]
                        }
                    },
                    "size": limit * 2
                }

                es_response = requests.post(es_url, json=es_query)
                if es_response.status_code == 200:
                    keyword_results = es_response.json().get("hits", {}).get("hits", [])

                    # Combine and rerank results
                    combined_results = {}

                    # Add vector search results with weight
                    for result in vector_results:
                        doc_id = result["metadata"]["document_id"]
                        combined_results[doc_id] = {
                            "document_id": doc_id,
                            "document_name": result["metadata"]["filename"],
                            "text": result["metadata"]["text"],
                            "score": result["score"] * vector_weight,
                            "source": "vector"
                        }

                    # Add keyword search results with weight
                    for hit in keyword_results:
                        doc_id = hit["_source"]["document_id"]
                        if doc_id in combined_results:
                            # Document already in results from vector search
                            combined_results[doc_id]["score"] += hit["_score"] * \
                                keyword_weight
                            combined_results[doc_id]["source"] = "hybrid"
                        else:
                            combined_results[doc_id] = {
                                "document_id": doc_id,
                                "document_name": hit["_source"]["filename"],
                                "text": hit["_source"]["text"],
                                "score": hit["_score"] * keyword_weight,
                                "source": "keyword"
                            }

                    # Sort by score and limit results
                    final_results = sorted(
                        combined_results.values(),
                        key=lambda x: x["score"],
                        reverse=True
                    )[:limit]

                    return {
                        "query": query,
                        "hybrid_search": True,
                        "results": final_results
                    }
            except Exception as e:
                print(f"Elasticsearch error: {e}")
                # Fall back to vector search only

        # Format and return vector search results
        return {
            "query": query,
            "hybrid_search": False,
            "results": [
                {
                    "document_id": result["metadata"]["document_id"],
                    "document_name": result["metadata"]["filename"],
                    "score": result["score"],
                    "text": result["metadata"]["text"]
                }
                for result in vector_results[:limit]
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

        # Get all chunks for the document to get the vector IDs
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc_id).all()
        vector_ids = [
            chunk.embedding_id for chunk in chunks if chunk.embedding_id]

        # Delete from Qdrant if there are vector IDs
        if vector_ids:
            from db.vector_db import VectorDB
            vector_db = VectorDB()
            vector_db.delete_vectors(
                collection_name="documents",
                vector_ids=vector_ids
            )

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
