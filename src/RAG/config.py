"""
Configuration settings for the RAG module.
Supports loading from .env file for sensitive data.
"""
import os
import dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

# Get the absolute path of the project root directory
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(current_script_path)))

# Model configuration
AVAILABLE_LLM_MODELS = os.getenv(
    "AVAILABLE_LLM_MODELS", "").split(",")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "")
AVAILABLE_EMBEDDING_MODELS = os.getenv(
    "AVAILABLE_EMBEDDING_MODELS", "").split(",")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "DEFAULT_EMBEDDING_MODEL", "")

# File paths
VECTORDB_PATH = os.getenv(
    "VECTORDB_PATH", os.path.join(project_root, "db", "vector"))
DB_PATH = os.getenv("DB_PATH", os.path.join(
    project_root, "db", "chat-history", "chat_db.sqlite"))

# Chunking parameters
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", ""))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", ""))

# API configuration
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "")
DIGITAL_OCEAN_API_URL = os.getenv("DIGITAL_OCEAN_API_URL",
                                  "")
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
