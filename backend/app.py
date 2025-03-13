"""
ToolXpert Backend API
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from sqlalchemy.orm import Session
from routes import auth_routes

# Import routes
try:
    from routes import document_routes, chat_routes, health_routes
except ImportError as e:
    print(
        f"Import error: {e}. Please run this script from the backend directory.")
    import sys
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title="ToolXpert API",
    description="Backend API for ToolXpert",
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
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])
app.include_router(document_routes.router,
                   prefix="/api/documents", tags=["documents"])
app.include_router(chat_routes.router, prefix="/api/chat", tags=["chat"])
app.include_router(health_routes.router, prefix="/api/health", tags=["health"])

# Root endpoint


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ToolXpert API",
        "documentation": "/docs",
        "version": "1.0.0"
    }

# Initialize database tables on startup


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    from db import Base, engine

    # Import models to ensure they're registered with Base
    from db.models import Document, ChatSession, ChatMessage

    # Create database tables
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized")

if __name__ == "__main__":
    # Run using Uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
