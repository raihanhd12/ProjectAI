import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseConfig:
    """Base configuration class with utility methods"""

    @staticmethod
    def get_env(key: str, default: Any = None) -> Any:
        """Get environment variable with fallback to default"""
        return os.getenv(key, default)

    @staticmethod
    def get_env_list(key: str, default: str = "", delimiter: str = ",") -> List[str]:
        """Get environment variable as a list with fallback to default"""
        value = os.getenv(key, default)
        if not value:
            return []
        return [item.strip() for item in value.split(delimiter)]

    @staticmethod
    def get_env_int(key: str, default: int = 0) -> int:
        """Get environment variable as integer with fallback to default"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def get_env_float(key: str, default: float = 0.0) -> float:
        """Get environment variable as float with fallback to default"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def get_env_bool(key: str, default: bool = False) -> bool:
        """Get environment variable as boolean with fallback to default"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', 'yes', '1', 'y')

    @staticmethod
    def ensure_dir(path: str) -> None:
        """Ensure directory exists"""
        os.makedirs(path, exist_ok=True)


class RAGConfig(BaseConfig):
    """Configuration for the RAG module"""

    # Database settings
    VECTORDB_PATH = BaseConfig.get_env("RAG_VECTORDB_PATH", "./vectordb")

    # Default models
    DEFAULT_LLM_MODEL = BaseConfig.get_env("RAG_DEFAULT_LLM_MODEL", "llama3")
    DEFAULT_EMBEDDING_MODEL = BaseConfig.get_env(
        "RAG_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text:latest")

    # Available models
    AVAILABLE_LLM_MODELS = BaseConfig.get_env_list(
        "RAG_AVAILABLE_LLM_MODELS", "llama3,qwen2.5")
    AVAILABLE_EMBEDDING_MODELS = BaseConfig.get_env_list(
        "RAG_AVAILABLE_EMBEDDING_MODELS", "nomic-embed-text:latest,nomic-embed-text,all-MiniLM-L6-v2"
    )

    # Chunking defaults
    DEFAULT_CHUNK_SIZE = BaseConfig.get_env_int("RAG_DEFAULT_CHUNK_SIZE", 400)
    DEFAULT_CHUNK_OVERLAP = BaseConfig.get_env_int(
        "RAG_DEFAULT_CHUNK_OVERLAP", 100)

    # Ollama API endpoint
    OLLAMA_API_URL = BaseConfig.get_env(
        "RAG_OLLAMA_API_URL", "http://localhost:11434/api/embeddings")

    # Collection name
    COLLECTION_NAME = BaseConfig.get_env("RAG_COLLECTION_NAME", "rag_app")

    @classmethod
    def initialize(cls):
        """Initialize RAG configuration (create directories, etc.)"""
        BaseConfig.ensure_dir(cls.VECTORDB_PATH)
        BaseConfig.ensure_dir("./data")


class GenerationConfig(BaseConfig):
    """Configuration for the Text Generation module"""

    DEFAULT_MODEL = BaseConfig.get_env("GEN_DEFAULT_MODEL", "llama3")
    AVAILABLE_MODELS = BaseConfig.get_env_list(
        "GEN_AVAILABLE_MODELS", "llama3,qwen2.5")
    MAX_TOKENS = BaseConfig.get_env_int("GEN_MAX_TOKENS", 2048)
    TEMPERATURE = BaseConfig.get_env_float("GEN_TEMPERATURE", 0.7)


class ImageGenConfig(BaseConfig):
    """Configuration for the Image Generation module"""

    DEFAULT_MODEL = BaseConfig.get_env("IMG_DEFAULT_MODEL", "stable-diffusion")
    AVAILABLE_MODELS = BaseConfig.get_env_list(
        "IMG_AVAILABLE_MODELS", "stable-diffusion,dalle-mini")
    DEFAULT_SIZE = BaseConfig.get_env("IMG_DEFAULT_SIZE", "512x512")
    AVAILABLE_SIZES = BaseConfig.get_env_list(
        "IMG_AVAILABLE_SIZES", "256x256,512x512,1024x1024")


# Initialize configurations
RAGConfig.initialize()
