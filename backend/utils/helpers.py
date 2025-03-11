"""
Utility functions for the backend.
"""
import os
import datetime
from typing import List


def ensure_directories(directories: List[str]) -> bool:
    """
    Ensure all specified directories exist.

    Args:
        directories (List[str]): List of directory paths to create

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        for directory in directories:
            # If the path points to a file, get its parent directory
            if '.' in os.path.basename(directory):
                directory = os.path.dirname(directory)

            # Create the directory if it doesn't exist
            if directory:  # Make sure we have a non-empty directory path
                os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False


def normalize_filename(filename: str) -> str:
    """
    Normalize a filename for use in the vector store.

    Args:
        filename (str): Original filename

    Returns:
        str: Normalized filename
    """
    return filename.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))


def format_timestamp(timestamp_str: str) -> str:
    """
    Format a timestamp string for display.

    Args:
        timestamp_str (str): ISO format timestamp string

    Returns:
        str: Formatted timestamp string
    """
    try:
        timestamp = datetime.datetime.fromisoformat(timestamp_str)
        return timestamp.strftime("%d %b, %H:%M")
    except (ValueError, TypeError):
        return "Unknown time"


def truncate_text(text: str, max_length: int = 40) -> str:
    """
    Truncate text to specified length with ellipsis.

    Args:
        text (str): Text to truncate
        max_length (int): Maximum length before truncation

    Returns:
        str: Truncated text
    """
    if not text:
        return "New conversation"
    return text if len(text) <= max_length else text[:max_length-3] + "..."