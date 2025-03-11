"""
AI Document Assistant Backend API
"""
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import os
import io
import tempfile
import shutil
from typing import List, Optional, Dict, Any
import numpy as np

# Import routes
from routes import document_routes, chat_routes

# Create FastAPI app
app = FastAPI(
    title="AI Document Assistant API",
    description="Backend API for AI Document Assistant",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(document_routes.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat_routes.router, prefix="/api/chat", tags=["chat"])

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Document Assistant API",
        "documentation": "/docs",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Run using Uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)