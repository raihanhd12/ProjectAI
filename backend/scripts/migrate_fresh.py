"""
Script to completely reset all databases (MySQL, Qdrant, Elasticsearch, MinIO).
Similar to Laravel's migrate:fresh but also handles vector stores and object storage.
"""
import sys
import os
import time
import requests
import subprocess
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)  # Parent of scripts is backend
sys.path.insert(0, backend_dir)

try:
    from db import Base, engine
    from db.models import Document, ChatSession, ChatMessage, User, TokenBlacklist
    from utils.documents_storage import DocumentStorage
    import config
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Path: {sys.path}")
    sys.exit(1)

# Constants for services
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
doc_storage = DocumentStorage()

#############################################
# MySQL Database Reset Functions
#############################################


def drop_all_tables():
    """Drop all tables in the database."""
    print("\n🗑️ Dropping all MySQL tables...")
    try:
        # Get all table names
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        if not table_names:
            print("✅ No tables to drop")
            return True

        # Drop all tables
        with engine.connect() as conn:
            # Disable foreign key checks temporarily to avoid constraint errors
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))

            # Drop each table
            for table in table_names:
                print(f"  📌 Dropping table: {table}")
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))

            # Re-enable foreign key checks
            conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
            conn.commit()

        print("✅ All MySQL tables dropped successfully")
        return True
    except SQLAlchemyError as e:
        print(f"❌ Error dropping tables: {e}")
        return False


def create_tables():
    """Create all database tables."""
    print("\n🔄 Creating all MySQL tables...")
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        return True
    except SQLAlchemyError as e:
        print(f"❌ Error creating tables: {e}")
        return False

#############################################
# Qdrant Reset Functions
#############################################


def reset_qdrant():
    """Reset all collections in Qdrant."""
    print("\n🗑️ Resetting Qdrant vector database...")
    try:
        # List all collections
        response = requests.get(f"{QDRANT_URL}/collections")
        if response.status_code != 200:
            print(
                f"⚠️ Failed to get Qdrant collections: {response.status_code}")
            return False

        collections = response.json().get("result", {}).get("collections", [])
        if not collections:
            print("✅ No Qdrant collections to delete")
            return True

        # Delete each collection
        for collection in collections:
            collection_name = collection.get("name")
            print(f"  📌 Deleting Qdrant collection: {collection_name}")
            delete_response = requests.delete(
                f"{QDRANT_URL}/collections/{collection_name}")
            if delete_response.status_code not in (200, 204):
                print(
                    f"⚠️ Failed to delete collection {collection_name}: {delete_response.status_code}")

        print("✅ All Qdrant collections reset successfully")
        return True
    except requests.RequestException as e:
        print(f"❌ Error resetting Qdrant: {e}")
        return False

#############################################
# Elasticsearch Reset Functions
#############################################


def reset_elasticsearch():
    """Reset all indices in Elasticsearch."""
    print("\n🗑️ Resetting Elasticsearch...")
    try:
        # List all indices
        response = requests.get(
            f"{ELASTICSEARCH_URL}/_cat/indices?format=json")
        if response.status_code != 200:
            print(
                f"⚠️ Failed to get Elasticsearch indices: {response.status_code}")
            return False

        indices = response.json()
        if not indices:
            print("✅ No Elasticsearch indices to delete")
            return True

        # Delete each index (except system indices)
        for index in indices:
            index_name = index.get("index")
            if index_name and not index_name.startswith("."):
                print(f"  📌 Deleting Elasticsearch index: {index_name}")
                delete_response = requests.delete(
                    f"{ELASTICSEARCH_URL}/{index_name}")
                if delete_response.status_code not in (200, 204):
                    print(
                        f"⚠️ Failed to delete index {index_name}: {delete_response.status_code}")

        print("✅ All Elasticsearch indices reset successfully")
        return True
    except requests.RequestException as e:
        print(f"❌ Error resetting Elasticsearch: {e}")
        return False

#############################################
# MinIO Reset Functions
#############################################


def reset_minio():
    """Reset all objects in MinIO."""
    print("\n🗑️ Resetting MinIO object storage...")
    try:
        # List all objects
        objects = doc_storage.list_objects()
        if not objects:
            print("✅ No objects to delete in MinIO")
            return True

        # Delete each object
        for obj in objects:
            object_name = obj.get("name")
            print(f"  📌 Deleting MinIO object: {object_name}")
            if not doc_storage.delete_file(object_name):
                print(f"⚠️ Failed to delete object {object_name}")

        print("✅ All MinIO objects reset successfully")
        return True
    except Exception as e:
        print(f"❌ Error resetting MinIO: {e}")
        return False

#############################################
# Main Function
#############################################


def main():
    """Main function to perform a complete reset."""
    print("🔄 Starting complete system reset (MySQL, Qdrant, Elasticsearch, MinIO)")

    try:
        # Reset MySQL database
        if not drop_all_tables() or not create_tables():
            print("⚠️ MySQL reset had issues but continuing...")

        # Reset Qdrant
        if not reset_qdrant():
            print("⚠️ Qdrant reset had issues but continuing...")

        # Reset Elasticsearch
        if not reset_elasticsearch():
            print("⚠️ Elasticsearch reset had issues but continuing...")

        # Reset MinIO
        if not reset_minio():
            print("⚠️ MinIO reset had issues but continuing...")

        print("\n🎉 Complete system reset finished!")
        print("   You can now run your application with 'uvicorn app:app --reload --host 0.0.0.0 --port 8080'")

    except Exception as e:
        print(f"❌ Error during system reset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
