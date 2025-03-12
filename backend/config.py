"""
Configuration settings for the AI Document Assistant backend.
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
    "database": os.getenv("MYSQL_DATABASE", "project_ai"),
}

# JWT Settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Vector DB path (legacy - kept for compatibility)
VECTORDB_PATH = os.getenv(
    "VECTORDB_PATH", os.path.join(BASE_DIR, "..", "db", "vector"))

# Make sure the directories exist
os.makedirs(VECTORDB_PATH, exist_ok=True)

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
# Default for nomic-embed
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Elasticsearch configuration
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER", "")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")
ELASTICSEARCH_VERIFY_CERTS = os.getenv(
    "ELASTICSEARCH_VERIFY_CERTS", "False").lower() == "true"

# Hybrid search configuration
ENABLE_HYBRID_SEARCH = os.getenv(
    "ENABLE_HYBRID_SEARCH", "True").lower() == "true"
# Weight for vector search results
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
# Weight for keyword search results
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))

# Model configuration
AVAILABLE_LLM_MODELS = os.getenv("AVAILABLE_LLM_MODELS", "").split(",")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "")
AVAILABLE_EMBEDDING_MODELS = os.getenv(
    "AVAILABLE_EMBEDDING_MODELS", "").split(",")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "")

# Chunking parameters
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))

# API configuration
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
DIGITAL_OCEAN_API_URL = os.getenv("DIGITAL_OCEAN_API_URL", "")
DIGITAL_OCEAN_API_KEY = os.getenv("DIGITAL_OCEAN_API_KEY", "")

# System prompt for the LLM
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: 
- Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
- If the question is in Indonesian, answer in Indonesian. If the question is in English, answer in English. Match the language of your response to the language of the question.
""")

# API Settings
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "your-secret-token")
ENABLE_TOKEN_AUTH = os.getenv("ENABLE_TOKEN_AUTH", "False").lower() == "true"
