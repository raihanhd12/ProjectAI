"""
Chat routes for the AI Document Assistant API.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import asyncio
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

# Import models
from models.hybrid_rag_model import HybridRAGModel
from db import chat_db

# Import dependencies
from dependencies import verify_api_token, get_db_session

# Create router
router = APIRouter()

# Initialize RAG model
rag_model = HybridRAGModel()

# Define request models


class ChatQuery(BaseModel):
    """Chat query request model."""
    prompt: str
    session_id: Optional[str] = None


class NewSession(BaseModel):
    """New chat session request model."""
    session_id: Optional[str] = None

# Define response models


class RetrievalResult(BaseModel):
    """Document retrieval result model."""
    retrieval_results: Optional[List[Dict[str, Any]]] = None
    relevant_text: Optional[str] = None
    relevant_text_ids: Optional[List[Any]] = None


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str
    thinking_content: Optional[str] = None
    retrieval_result: Optional[RetrievalResult] = None


@router.get("/history")
async def get_chat_history(
    session_id: Optional[str] = None,
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Get chat history for a session.

    Args:
        session_id: Optional session ID (uses current if not provided)

    Returns:
        List of chat messages
    """
    try:
        # If no session ID provided, use the most recent one
        if not session_id:
            recent_sessions = chat_db.get_recent_chat_sessions(db, 1)
            if recent_sessions:
                session_id = recent_sessions[0][0]
            else:
                return {"messages": [], "session_id": None}

        messages = chat_db.get_chat_history(db, session_id)
        return {"messages": messages, "session_id": session_id}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get chat history: {str(e)}")


@router.get("/sessions")
async def get_chat_sessions(
    limit: int = 5,
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Get recent chat sessions.

    Args:
        limit: Maximum number of sessions to return

    Returns:
        List of recent chat sessions
    """
    try:
        sessions = chat_db.get_recent_chat_sessions(db, limit)
        formatted_sessions = []

        for session_id, start_time, last_time, first_question in sessions:
            formatted_sessions.append({
                "session_id": session_id,
                "start_time": start_time,
                "last_time": last_time,
                "first_question": first_question
            })

        return {"sessions": formatted_sessions}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get chat sessions: {str(e)}")


@router.post("/sessions")
async def create_chat_session(
    session_data: NewSession = None,
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Create a new chat session.

    Args:
        session_data: Optional custom session ID

    Returns:
        New session ID
    """
    try:
        session_id = session_data.session_id if session_data else None
        new_session_id = chat_db.create_session(db, session_id)
        return {"session_id": new_session_id}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create chat session: {str(e)}")


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Delete a chat session.

    Args:
        session_id: Session ID to delete

    Returns:
        Deletion status
    """
    try:
        success = chat_db.delete_chat_session(db, session_id)
        if success:
            return {"status": "success", "message": f"Session {session_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to delete session")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.post("/query")
async def query_chat(
    chat_query: ChatQuery,
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Query the RAG model (non-streaming version).

    Args:
        chat_query: Chat query

    Returns:
        Model response
    """
    try:
        # Save user message
        session_id = chat_query.session_id or chat_db.create_session(db)
        chat_db.save_message(db, session_id, "user", chat_query.prompt)

        # Perform hybrid search
        retrieval_results, relevant_text_ids = rag_model.hybrid_search(
            chat_query.prompt)

        if retrieval_results and len(retrieval_results) > 0:
            # Extract relevant text from search results
            relevant_text = "\n\n".join([doc["text"]
                                        for doc in retrieval_results])

            # Generate response
            response_content = ""
            thinking_content = ""

            # Non-streaming response
            for chunk in rag_model.call_llm(context=relevant_text, prompt=chat_query.prompt):
                # Check for thinking tags
                if "<think>" in chunk:
                    parts = chunk.split("<think>")
                    if len(parts) > 0:
                        response_content += parts[0]
                    if len(parts) > 1:
                        thinking_content += parts[1]
                elif "</think>" in chunk:
                    parts = chunk.split("</think>")
                    thinking_content += parts[0]
                    if len(parts) > 1:
                        response_content += parts[1]
                else:
                    response_content += chunk

            # Save assistant message
            chat_db.save_message(
                db,
                session_id,
                "assistant",
                response_content,
                thinking_content,
                {"results": [doc["id"] for doc in retrieval_results]},
                relevant_text_ids,
                relevant_text
            )

            return {
                "session_id": session_id,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                    "thinking_content": thinking_content,
                    "retrieval_result": {
                        "retrieval_results": retrieval_results,
                        "relevant_text": relevant_text,
                        "relevant_text_ids": relevant_text_ids
                    }
                }
            }
        else:
            return {
                "session_id": session_id,
                "message": {
                    "role": "assistant",
                    "content": "I couldn't find any relevant information in the documents to answer your question. Please try a different question or upload more documents.",
                    "thinking_content": "No relevant documents found in the vector database.",
                    "retrieval_result": None
                }
            }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process query: {str(e)}")


@router.post("/query/stream")
async def stream_chat_query(
    chat_query: ChatQuery,
    db: Session = Depends(get_db_session),
    token: str = Depends(verify_api_token)
):
    """
    Stream the RAG model response.

    Args:
        chat_query: Chat query

    Returns:
        Streaming response
    """
    async def event_generator():
        try:
            # Save user message
            session_id = chat_query.session_id or chat_db.create_session(db)
            chat_db.save_message(db, session_id, "user", chat_query.prompt)

            # Perform hybrid search
            retrieval_results, relevant_text_ids = rag_model.hybrid_search(
                chat_query.prompt)

            if retrieval_results and len(retrieval_results) > 0:
                # Extract relevant text from search results
                relevant_text = "\n\n".join(
                    [doc["text"] for doc in retrieval_results])

                # Send retrieval results
                retrieval_data = {
                    "type": "retrieval",
                    "session_id": session_id,
                    "data": {
                        "retrieval_results": retrieval_results,
                        "relevant_text": relevant_text,
                        "relevant_text_ids": relevant_text_ids
                    }
                }
                yield json.dumps(retrieval_data)

                # Variables for response
                full_response = ""
                thinking_content = ""
                in_thinking_section = False

                # Stream response
                for chunk in rag_model.call_llm(context=relevant_text, prompt=chat_query.prompt):
                    # Handle thinking tags
                    if "<think>" in chunk:
                        parts = chunk.split("<think>")
                        if len(parts) > 0:
                            full_response += parts[0]
                        if len(parts) > 1:
                            thinking_content += parts[1]
                        in_thinking_section = True

                        # Send thinking update
                        thinking_data = {
                            "type": "thinking",
                            "data": thinking_content
                        }
                        yield json.dumps(thinking_data)

                    elif "</think>" in chunk and in_thinking_section:
                        parts = chunk.split("</think>")
                        thinking_content += parts[0]
                        if len(parts) > 1:
                            full_response += parts[1]
                        in_thinking_section = False

                        # Send thinking update
                        thinking_data = {
                            "type": "thinking",
                            "data": thinking_content
                        }
                        yield json.dumps(thinking_data)

                        # Send content update
                        content_data = {
                            "type": "content",
                            "data": parts[1] if len(parts) > 1 else ""
                        }
                        yield json.dumps(content_data)

                    elif in_thinking_section:
                        thinking_content += chunk

                        # Send thinking update
                        thinking_data = {
                            "type": "thinking",
                            "data": thinking_content
                        }
                        yield json.dumps(thinking_data)

                    else:
                        full_response += chunk

                        # Send content update
                        content_data = {
                            "type": "content",
                            "data": chunk
                        }
                        yield json.dumps(content_data)

                # Save assistant message
                chat_db.save_message(
                    db,
                    session_id,
                    "assistant",
                    full_response,
                    thinking_content,
                    {"results": [doc["id"] for doc in retrieval_results]},
                    relevant_text_ids,
                    relevant_text
                )

                # Send complete message
                complete_data = {
                    "type": "complete",
                    "session_id": session_id,
                    "data": {
                        "role": "assistant",
                        "content": full_response,
                        "thinking_content": thinking_content
                    }
                }
                yield json.dumps(complete_data)

            else:
                # No relevant documents found
                error_data = {
                    "type": "error",
                    "data": "No relevant documents found. Please try a different question or upload more documents."
                }
                yield json.dumps(error_data)

                # Save assistant message with error
                chat_db.save_message(
                    db,
                    session_id,
                    "assistant",
                    "I couldn't find any relevant information in the documents to answer your question. Please try a different question or upload more documents.",
                    "No relevant documents found in the vector database.",
                    None,
                    None,
                    None
                )

        except Exception as e:
            # Send error
            error_data = {
                "type": "error",
                "data": f"Error processing query: {str(e)}"
            }
            yield json.dumps(error_data)

    return EventSourceResponse(event_generator())
