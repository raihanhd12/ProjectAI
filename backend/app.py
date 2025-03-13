"""
ToolXpert Backend API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import routes
from routes import auth_routes, documents_routes

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
app.include_router(documents_routes.router,
                   prefix="/api/documents", tags=["documents"])

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
    from db.models import User, TokenBlacklist

    # Create database tables
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized")

if __name__ == "__main__":
    # Run using Uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
