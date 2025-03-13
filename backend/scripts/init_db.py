"""
Comprehensive database initialization script for the ToolXpert.
Initializes MySQL database, vector stores (Qdrant and Elasticsearch), and object storage (MinIO).
"""
import sys
import os
import time
import subprocess
import requests
import json
from pathlib import Path

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
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Failed to import. Current path: {sys.path}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Define data directories
BACKEND_DIR = os.path.abspath(parent_dir)
DB_DIR = os.path.join(BACKEND_DIR, "db")

# Constants for Docker containers
QDRANT_CONTAINER_NAME = "qdrant"
ELASTICSEARCH_CONTAINER_NAME = "elasticsearch"
KIBANA_CONTAINER_NAME = "kibana"
MINIO_CONTAINER_NAME = "minio"

# Data directories
QDRANT_DATA_DIR = os.path.join(DB_DIR, "qdrant_data")
ELASTICSEARCH_DATA_DIR = os.path.join(DB_DIR, "elasticsearch_data")
MINIO_DATA_DIR = os.path.join(DB_DIR, "document_storage")

# Ports
QDRANT_PORT = 6333
ELASTICSEARCH_PORT = 9200
KIBANA_PORT = 5601
MINIO_API_PORT = 9000
MINIO_CONSOLE_PORT = 9001

# MinIO credentials
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
DOCUMENT_BUCKET_NAME = "documents"

#############################################
# MySQL Database Initialization Functions
#############################################


def create_database():
    """Create the MySQL database if it doesn't exist."""
    print("\n🔄 Initializing MySQL Database")

    # Create connection string without database
    conn_string = f"mysql+pymysql://{config.DB_CONFIG['user']}:{config.DB_CONFIG['password']}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}"

    try:
        # Connect to MySQL server
        temp_engine = create_engine(conn_string)

        # Create database
        with temp_engine.connect() as conn:
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS {config.DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
            print(
                f"✅ Database '{config.DB_CONFIG['database']}' created or already exists")
        return True
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False


def create_tables():
    """Create all database tables."""
    try:
        # Import all models to ensure they're registered with Base
        from db.models import Document, ChatSession, ChatMessage, User, TokenBlacklist

        # Create tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

#############################################
# Docker and Container Management Functions
#############################################


def check_docker_installed():
    """Check if Docker is installed and running."""
    try:
        subprocess.run(["docker", "--version"],
                       check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_container_running(container_name):
    """Check if a Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            check=True,
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


#############################################
# Qdrant and Elasticsearch Functions
#############################################

def start_qdrant():
    """Start Qdrant Docker container."""
    if check_container_running(QDRANT_CONTAINER_NAME):
        print(
            f"✅ Qdrant container '{QDRANT_CONTAINER_NAME}' is already running")
        return True

    print(f"🚀 Starting Qdrant container...")
    os.makedirs(QDRANT_DATA_DIR, exist_ok=True)

    try:
        # Check if container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f",
                f"name={QDRANT_CONTAINER_NAME}"],
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            # Container exists, start it
            subprocess.run(
                ["docker", "start", QDRANT_CONTAINER_NAME],
                check=True
            )
        else:
            # Container doesn't exist, create and run it
            subprocess.run([
                "docker", "run", "-d",
                "--name", QDRANT_CONTAINER_NAME,
                "-p", f"{QDRANT_PORT}:{QDRANT_PORT}",
                "-p", "6334:6334",
                "-v", f"{QDRANT_DATA_DIR}:/qdrant/storage",
                "qdrant/qdrant"
            ], check=True)

        print(f"✅ Qdrant container started successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Qdrant container: {e}")
        return False


def start_elasticsearch():
    """Start Elasticsearch Docker container."""
    if check_container_running(ELASTICSEARCH_CONTAINER_NAME):
        print(
            f"✅ Elasticsearch container '{ELASTICSEARCH_CONTAINER_NAME}' is already running")
        return True

    print(f"🚀 Starting Elasticsearch container...")
    os.makedirs(ELASTICSEARCH_DATA_DIR, exist_ok=True)

    try:
        # Check if container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f",
                f"name={ELASTICSEARCH_CONTAINER_NAME}"],
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            # Container exists, start it
            subprocess.run(
                ["docker", "start", ELASTICSEARCH_CONTAINER_NAME],
                check=True
            )
        else:
            # Container doesn't exist, create and run it
            subprocess.run([
                "docker", "run", "-d",
                "--name", ELASTICSEARCH_CONTAINER_NAME,
                "-p", f"{ELASTICSEARCH_PORT}:{ELASTICSEARCH_PORT}",
                "-p", "9300:9300",
                "-e", "discovery.type=single-node",
                "-e", "xpack.security.enabled=false",
                "-v", f"{ELASTICSEARCH_DATA_DIR}:/usr/share/elasticsearch/data",
                "elasticsearch:8.6.2"
            ], check=True)

        print(f"✅ Elasticsearch container started successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Elasticsearch container: {e}")
        return False


def start_kibana():
    """Start Kibana Docker container for Elasticsearch UI."""
    if check_container_running(KIBANA_CONTAINER_NAME):
        print(
            f"✅ Kibana container '{KIBANA_CONTAINER_NAME}' is already running")
        return True

    print(f"🚀 Starting Kibana container for Elasticsearch UI...")

    try:
        # Check if container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f",
                f"name={KIBANA_CONTAINER_NAME}"],
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            # Container exists, start it
            subprocess.run(
                ["docker", "start", KIBANA_CONTAINER_NAME],
                check=True
            )
        else:
            # Container doesn't exist, create and run it
            subprocess.run([
                "docker", "run", "-d",
                "--name", KIBANA_CONTAINER_NAME,
                "--link", f"{ELASTICSEARCH_CONTAINER_NAME}:elasticsearch",
                "-p", f"{KIBANA_PORT}:{KIBANA_PORT}",
                "-e", f"ELASTICSEARCH_HOSTS=http://{ELASTICSEARCH_CONTAINER_NAME}:9200",
                "kibana:8.6.2"
            ], check=True)

        print(f"✅ Kibana container started successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Kibana container: {e}")
        return False


def wait_for_qdrant():
    """Wait until Qdrant is ready to accept connections."""
    print("⏳ Waiting for Qdrant to be ready...")
    max_retries = 30
    retry_interval = 2

    for i in range(max_retries):
        try:
            response = requests.get(
                f"http://localhost:{QDRANT_PORT}/collections")
            if response.status_code == 200:
                print("✅ Qdrant is ready!")
                return True
        except requests.RequestException:
            pass

        print(f"⏳ Waiting for Qdrant to start ({i+1}/{max_retries})...")
        time.sleep(retry_interval)

    print("❌ Timed out waiting for Qdrant")
    return False


def wait_for_elasticsearch():
    """Wait until Elasticsearch is ready to accept connections."""
    print("⏳ Waiting for Elasticsearch to be ready...")
    max_retries = 60  # Elasticsearch can take longer to start
    retry_interval = 2

    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{ELASTICSEARCH_PORT}")
            if response.status_code == 200:
                print("✅ Elasticsearch is ready!")
                return True
        except requests.RequestException:
            pass

        print(f"⏳ Waiting for Elasticsearch to start ({i+1}/{max_retries})...")
        time.sleep(retry_interval)

    print("❌ Timed out waiting for Elasticsearch")
    return False


def wait_for_kibana():
    """Wait until Kibana is ready to accept connections."""
    print("⏳ Waiting for Kibana to be ready...")
    max_retries = 60  # Kibana can take longer to start
    retry_interval = 2

    for i in range(max_retries):
        try:
            response = requests.get(
                f"http://localhost:{KIBANA_PORT}/api/status")
            if response.status_code == 200 or response.status_code == 302:
                print("✅ Kibana is ready!")
                return True
        except requests.RequestException:
            pass

        print(f"⏳ Waiting for Kibana to start ({i+1}/{max_retries})...")
        time.sleep(retry_interval)

    print("❌ Timed out waiting for Kibana")
    return False


def create_test_collections():
    """Create test collections to verify everything works."""
    print("\n🧪 Creating test collections...")

    # Create test collection in Qdrant
    try:
        response = requests.put(
            f"http://localhost:{QDRANT_PORT}/collections/test_collection",
            json={
                "vectors": {
                    "size": 384,
                    "distance": "Cosine"
                }
            }
        )

        if response.status_code in (200, 201):
            print("✅ Created test collection in Qdrant")
        else:
            print(
                f"⚠️ Failed to create test collection in Qdrant: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to create test collection in Qdrant: {e}")

    # Create test index in Elasticsearch
    try:
        response = requests.put(
            f"http://localhost:{ELASTICSEARCH_PORT}/test_index",
            json={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text"}
                    }
                }
            }
        )

        if response.status_code in (200, 201):
            print("✅ Created test index in Elasticsearch")
        else:
            print(
                f"⚠️ Failed to create test index in Elasticsearch: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to create test index in Elasticsearch: {e}")


#############################################
# MinIO Object Storage Functions
#############################################

def start_minio():
    """Start MinIO Docker container."""
    if check_container_running(MINIO_CONTAINER_NAME):
        print(
            f"✅ MinIO container '{MINIO_CONTAINER_NAME}' is already running")
        return True

    print(f"🚀 Starting MinIO object storage container...")
    os.makedirs(MINIO_DATA_DIR, exist_ok=True)

    try:
        # Check if container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f",
                f"name={MINIO_CONTAINER_NAME}"],
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            # Container exists, start it
            subprocess.run(
                ["docker", "start", MINIO_CONTAINER_NAME],
                check=True
            )
        else:
            # Container doesn't exist, create and run it
            subprocess.run([
                "docker", "run", "-d",
                "--name", MINIO_CONTAINER_NAME,
                "-p", f"{MINIO_API_PORT}:{MINIO_API_PORT}",
                "-p", f"{MINIO_CONSOLE_PORT}:{MINIO_CONSOLE_PORT}",
                "-e", f"MINIO_ROOT_USER={MINIO_ACCESS_KEY}",
                "-e", f"MINIO_ROOT_PASSWORD={MINIO_SECRET_KEY}",
                "-v", f"{MINIO_DATA_DIR}:/data",
                "minio/minio", "server", "/data",
                "--console-address", f":{MINIO_CONSOLE_PORT}"
            ], check=True)

        print(f"✅ MinIO container started successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start MinIO container: {e}")
        return False


def wait_for_minio():
    """Wait until MinIO is ready to accept connections."""
    print("⏳ Waiting for MinIO to be ready...")
    max_retries = 30
    retry_interval = 2

    for i in range(max_retries):
        try:
            # Check MinIO API
            response = requests.get(
                f"http://localhost:{MINIO_API_PORT}/minio/health/live")
            if response.status_code == 200:
                print("✅ MinIO API is ready!")
                return True
        except requests.RequestException:
            pass

        try:
            # Check Console as fallback
            response = requests.get(f"http://localhost:{MINIO_CONSOLE_PORT}")
            if response.status_code == 200:
                print("✅ MinIO Console is ready!")
                return True
        except requests.RequestException:
            pass

        print(f"⏳ Waiting for MinIO to start ({i+1}/{max_retries})...")
        time.sleep(retry_interval)

    print("❌ Timed out waiting for MinIO")
    return False


def create_minio_bucket():
    """Create a bucket in MinIO for document storage using mc client."""
    print("📦 Creating document storage bucket...")

    try:
        # Run the mc (MinIO Client) in a Docker container to create a bucket
        # First, add the MinIO server as a host
        subprocess.run([
            "docker", "run", "--rm", "--network=host",
            "minio/mc", "alias", "set", "myminio",
            f"http://localhost:{MINIO_API_PORT}",
            MINIO_ACCESS_KEY, MINIO_SECRET_KEY
        ], check=True)

        # Then create the bucket
        subprocess.run([
            "docker", "run", "--rm", "--network=host",
            "minio/mc", "mb", f"myminio/{DOCUMENT_BUCKET_NAME}"
        ], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        print(
            f"✅ Created bucket '{DOCUMENT_BUCKET_NAME}' for document storage")
        return True
    except subprocess.CalledProcessError as e:
        # Check if the error is just that the bucket already exists
        error_output = e.stderr.decode() if e.stderr else ""
        if "already exists" in error_output:
            print(f"ℹ️ Bucket '{DOCUMENT_BUCKET_NAME}' already exists")
            return True
        else:
            print(f"❌ Failed to create bucket: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr.decode()}")
            return False


#############################################
# Environment Configuration
#############################################

def update_env_file():
    """Update .env file with all database configurations."""
    env_path = os.path.join(parent_dir, ".env")

    # Load existing .env
    env_content = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_content[key] = value

    # Update vector database settings
    updates = {
        # Vector DB configuration
        "QDRANT_URL": f"http://localhost:{QDRANT_PORT}",
        "ELASTICSEARCH_URL": f"http://localhost:{ELASTICSEARCH_PORT}",
        "ELASTICSEARCH_USER": "",
        "ELASTICSEARCH_PASSWORD": "",
        "ELASTICSEARCH_VERIFY_CERTS": "False",
        "ENABLE_HYBRID_SEARCH": "True",
        "VECTOR_WEIGHT": "0.7",
        "KEYWORD_WEIGHT": "0.3",
        "EMBEDDING_DIMENSION": "384",

        # MinIO configuration
        "MINIO_ENDPOINT": f"http://localhost:{MINIO_API_PORT}",
        "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
        "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
        "MINIO_BUCKET_NAME": DOCUMENT_BUCKET_NAME,
        "MINIO_SECURE": "False",
        "STORAGE_TYPE": "minio",  # Options: local, minio, s3

        # Set explicit path for legacy vector DB to avoid creating in root
        "VECTORDB_PATH": os.path.join(DB_DIR, "vector")
    }

    # Merge with existing env content
    env_content.update(updates)

    # Write back to .env file
    with open(env_path, "w") as f:
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")

    print("✅ Updated .env file with database configurations")


#############################################
# System Status
#############################################

def check_system_status():
    """Check and display system status."""
    print("\n🔍 System Status:")

    # Check MySQL
    try:
        # Create connection string with database
        conn_string = f"mysql+pymysql://{config.DB_CONFIG['user']}:{config.DB_CONFIG['password']}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['database']}"

        # Connect to MySQL server
        temp_engine = create_engine(conn_string)
        with temp_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ MySQL database is running")
    except Exception as e:
        print(f"❌ MySQL database error: {e}")

    # Check Qdrant
    try:
        response = requests.get(f"http://localhost:{QDRANT_PORT}/collections")
        if response.status_code == 200:
            collections = response.json().get("result", {}).get("collections", [])
            print(f"✅ Qdrant is running with {len(collections)} collections")
            print(f"  🌐 Qdrant UI: http://localhost:{QDRANT_PORT}/dashboard")
        else:
            print(f"⚠️ Qdrant returned status code {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to connect to Qdrant: {e}")

    # Check Elasticsearch
    try:
        response = requests.get(f"http://localhost:{ELASTICSEARCH_PORT}")
        if response.status_code == 200:
            version = response.json().get("version", {}).get("number", "unknown")
            print(f"✅ Elasticsearch v{version} is running")
            print(
                f"  🌐 Elasticsearch API: http://localhost:{ELASTICSEARCH_PORT}")
        else:
            print(
                f"⚠️ Elasticsearch returned status code {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to connect to Elasticsearch: {e}")

    # Check Kibana
    try:
        response = requests.get(f"http://localhost:{KIBANA_PORT}/api/status")
        if response.status_code in [200, 302]:
            print(f"✅ Kibana is running")
            print(f"  🌐 Kibana UI: http://localhost:{KIBANA_PORT}")
        else:
            print(f"⚠️ Kibana returned status code {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to connect to Kibana: {e}")

    # Check MinIO
    try:
        response = requests.get(
            f"http://localhost:{MINIO_API_PORT}/minio/health/live")
        if response.status_code == 200:
            print(f"✅ MinIO is running")
            print(f"  🌐 MinIO Console: http://localhost:{MINIO_CONSOLE_PORT}")
            print(
                f"     - Login with: {MINIO_ACCESS_KEY} / {MINIO_SECRET_KEY}")
        else:
            print(f"⚠️ MinIO returned status code {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to connect to MinIO: {e}")


def print_management_ui_info():
    """Print information about all management UIs."""
    print("\n📊 Database Management Interfaces:")

    print(f"\n  🔍 Qdrant Dashboard: http://localhost:{QDRANT_PORT}/dashboard")
    print(f"     - View collections and settings")
    print(f"     - Run queries")
    print(f"     - Monitor performance")

    print(f"\n  🔍 Kibana (Elasticsearch UI): http://localhost:{KIBANA_PORT}")
    print(f"     - Explore and query data")
    print(f"     - Create visualizations")
    print(f"     - Manage indices and mappings")
    print(f"     - Note: First-time setup may require configuration")

    print(f"\n  🔍 MinIO Console: http://localhost:{MINIO_CONSOLE_PORT}")
    print(f"     - Login with: {MINIO_ACCESS_KEY} / {MINIO_SECRET_KEY}")
    print(f"     - Browse and manage buckets and files")
    print(f"     - Monitor storage usage")
    print(f"     - Set access policies")


#############################################
# Main Function
#############################################

def main():
    """Main function to initialize all databases."""
    print("🔄 Initializing ToolXpert Databases\n")

    try:
        # 1. Initialize MySQL Database
        if not create_database() or not create_tables():
            print("⚠️ MySQL database initialization had some issues")

        # 2. Check if Docker is installed (required for vector DBs and MinIO)
        if not check_docker_installed():
            print(
                "❌ Docker is not installed or not running. Please install Docker first.")
            print("   You can install Docker by running: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh")
            sys.exit(1)

        print("✅ Docker is installed and running")

        # 3. Start Vector Databases
        print("\n🔄 Initializing Vector Databases (Qdrant and Elasticsearch)")

        # Start Qdrant
        if not start_qdrant():
            print("❌ Failed to start Qdrant container")
            sys.exit(1)

        # Start Elasticsearch
        if not start_elasticsearch():
            print("❌ Failed to start Elasticsearch container")
            sys.exit(1)

        # Start Kibana for Elasticsearch UI
        if not start_kibana():
            print("⚠️ Failed to start Kibana container (Elasticsearch UI)")
            print("   You can still use Elasticsearch API directly")

        # Wait for vector DBs to be ready
        if not wait_for_qdrant() or not wait_for_elasticsearch():
            print("❌ Failed to initialize vector databases")
            sys.exit(1)

        # If Kibana was started, wait for it
        if check_container_running(KIBANA_CONTAINER_NAME):
            wait_for_kibana()

        # Create test collections
        create_test_collections()

        # 4. Start MinIO Object Storage
        print("\n🔄 Initializing MinIO Object Storage")

        # Start MinIO
        if not start_minio():
            print("❌ Failed to start MinIO container")
            sys.exit(1)

        # Wait for MinIO to be ready
        if not wait_for_minio():
            print("❌ Failed to initialize MinIO")
            sys.exit(1)

        # Create bucket for document storage
        if not create_minio_bucket():
            print("⚠️ Failed to create MinIO bucket")

        # 5. Update environment variables
        update_env_file()

        # 6. Show system status
        check_system_status()

        # 7. Print management UI information
        print_management_ui_info()

        print("\n🎉 All databases initialized successfully!")
        print("   You can now run your application with 'uvicorn app:app --reload --host 0.0.0.0 --port 8080'")

    except Exception as e:
        print(f"❌ Error initializing databases: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
