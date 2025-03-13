"""
Configuration settings for the ToolXpert backend.
"""
import os
import dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

# Get the absolute path of the project root directory
BASE_DIR = Path(__file__).resolve().parent

# Database configuration
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "toolxpert"),
}

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "documents")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

# JWT Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# API Settings
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "your-secret-token")
ENABLE_TOKEN_AUTH = os.getenv("ENABLE_TOKEN_AUTH", "False").lower() == "true"
