"""
Script to initialize Qdrant and Elasticsearch for vector and keyword search.
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

# Import application modules
try:
    import config
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Constants
QDRANT_CONTAINER_NAME = "qdrant"
ELASTICSEARCH_CONTAINER_NAME = "elasticsearch"
QDRANT_DATA_DIR = os.path.join(parent_dir, "..", "qdrant_data")
ELASTICSEARCH_DATA_DIR = os.path.join(parent_dir, "..", "elasticsearch_data")
QDRANT_PORT = 6333
ELASTICSEARCH_PORT = 9200


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


def update_env_file():
    """Update .env file with Qdrant and Elasticsearch configuration."""
    env_path = os.path.join(parent_dir, "..", ".env")

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
        "QDRANT_URL": f"http://localhost:{QDRANT_PORT}",
        "ELASTICSEARCH_URL": f"http://localhost:{ELASTICSEARCH_PORT}",
        "ELASTICSEARCH_USER": "",
        "ELASTICSEARCH_PASSWORD": "",
        "ELASTICSEARCH_VERIFY_CERTS": "False",
        "ENABLE_HYBRID_SEARCH": "True",
        "VECTOR_WEIGHT": "0.7",
        "KEYWORD_WEIGHT": "0.3",
        "EMBEDDING_DIMENSION": "384"
    }

    # Merge with existing env content
    env_content.update(updates)

    # Write back to .env file
    with open(env_path, "w") as f:
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")

    print("✅ Updated .env file with vector database configuration")


def check_system_status():
    """Check and display system status."""
    print("\n🔍 System Status:")

    # Check Qdrant
    try:
        response = requests.get(f"http://localhost:{QDRANT_PORT}/collections")
        if response.status_code == 200:
            collections = response.json().get("result", {}).get("collections", [])
            print(f"✅ Qdrant is running with {len(collections)} collections")
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
        else:
            print(
                f"⚠️ Elasticsearch returned status code {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to connect to Elasticsearch: {e}")


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


def main():
    """Main function to initialize vector databases."""
    print("🔄 Initializing Vector Databases (Qdrant and Elasticsearch)\n")

    # Check if Docker is installed
    if not check_docker_installed():
        print("❌ Docker is not installed or not running. Please install Docker first.")
        print("   You can install Docker by running: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh")
        sys.exit(1)

    print("✅ Docker is installed and running")

    # Start Qdrant
    if not start_qdrant():
        print("❌ Failed to start Qdrant container")
        sys.exit(1)

    # Start Elasticsearch
    if not start_elasticsearch():
        print("❌ Failed to start Elasticsearch container")
        sys.exit(1)

    # Wait for services to be ready
    if not wait_for_qdrant() or not wait_for_elasticsearch():
        print("❌ Failed to initialize vector databases")
        sys.exit(1)

    # Update environment file
    update_env_file()

    # Create test collections
    create_test_collections()

    # Display system status
    check_system_status()

    print("\n🎉 Vector databases initialized successfully!")
    print("   You can now run your application with 'python app.py'")


if __name__ == "__main__":
    main()
