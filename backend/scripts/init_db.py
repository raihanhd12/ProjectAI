"""
Script to initialize the database and create tables.
"""
import sys
import os

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Print for debugging
print(f"Added to path: {parent_dir}")
print(f"Current path: {sys.path}")

# Now import application modules
try:
    import config
    from db import Base, engine
    from sqlalchemy import create_engine, text
    import pymysql
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Failed to import. Current path: {sys.path}")
    sys.exit(1)


def create_database():
    """Create the database if it doesn't exist."""
    # Create connection string without database
    conn_string = f"mysql+pymysql://{config.DB_CONFIG['user']}:{config.DB_CONFIG['password']}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}"

    # Connect to MySQL server
    temp_engine = create_engine(conn_string)

    # Create database
    with temp_engine.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS {config.DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        print(
            f"Database '{config.DB_CONFIG['database']}' created or already exists")


def create_tables():
    """Create all database tables."""
    # Import all models to ensure they're registered with Base
    from db.models import Document, ChatSession, ChatMessage

    # Create tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created")


def main():
    try:
        print("Initializing database...")
        create_database()
        create_tables()
        print("Database initialization complete")
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
