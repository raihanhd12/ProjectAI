"""
Script to check status of Qdrant and Elasticsearch services.
"""
import sys
import os
import subprocess
import requests
import json

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Constants
QDRANT_CONTAINER_NAME = "qdrant"
ELASTICSEARCH_CONTAINER_NAME = "elasticsearch"
QDRANT_PORT = 6333
ELASTICSEARCH_PORT = 9200


def check_container_status(container_name):
    """
    Check if Docker container is running, stopped, or doesn't exist.
    Returns: 'running', 'stopped', or 'not_found'
    """
    try:
        # Check if container exists and is running
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={container_name}"],
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return 'running'

        # Check if container exists but is stopped
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "-f", f"name={container_name}"],
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return 'stopped'

        return 'not_found'
    except subprocess.CalledProcessError:
        return 'error'


def get_container_info(container_name):
    """Get detailed information about a container."""
    try:
        if check_container_status(container_name) == 'not_found':
            return None

        result = subprocess.run(
            ["docker", "inspect", container_name],
            check=True,
            capture_output=True,
            text=True
        )
        info = json.loads(result.stdout)
        return info[0] if info else None
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def check_qdrant_status():
    """Check Qdrant service status."""
    status = check_container_status(QDRANT_CONTAINER_NAME)
    info = get_container_info(QDRANT_CONTAINER_NAME)

    print(f"\n📊 Qdrant Status:")

    # Container status
    if status == 'running':
        print(f"  Container: ✅ Running")

        # Check API
        try:
            response = requests.get(
                f"http://localhost:{QDRANT_PORT}/collections")
            if response.status_code == 200:
                collections = response.json().get("result", {}).get("collections", [])
                collection_names = [c['name'] for c in collections]
                print(
                    f"  API: ✅ Responding (found {len(collections)} collections)")
                if collection_names:
                    print(f"  Collections: {', '.join(collection_names)}")
            else:
                print(
                    f"  API: ⚠️ Responding with error code {response.status_code}")
        except requests.RequestException as e:
            print(f"  API: ❌ Not responding ({e})")

    elif status == 'stopped':
        print(f"  Container: ⚠️ Stopped")
        if info:
            print(f"  Created: {info.get('Created', 'Unknown')}")

    elif status == 'not_found':
        print(f"  Container: ❌ Not found")

    else:
        print(f"  Container: ❓ Status check failed")

    # Display volume info if available
    if info and 'Mounts' in info:
        for mount in info.get('Mounts', []):
            if mount.get('Destination', '') == '/qdrant/storage':
                print(f"  Data directory: {mount.get('Source', 'Unknown')}")
                break


def check_elasticsearch_status():
    """Check Elasticsearch service status."""
    status = check_container_status(ELASTICSEARCH_CONTAINER_NAME)
    info = get_container_info(ELASTICSEARCH_CONTAINER_NAME)

    print(f"\n📊 Elasticsearch Status:")

    # Container status
    if status == 'running':
        print(f"  Container: ✅ Running")

        # Check API
        try:
            response = requests.get(f"http://localhost:{ELASTICSEARCH_PORT}")
            if response.status_code == 200:
                version = response.json().get("version", {}).get("number", "unknown")
                name = response.json().get("name", "unknown")
                print(
                    f"  API: ✅ Responding (version: {version}, node: {name})")

                # Check indices
                try:
                    indices_response = requests.get(
                        f"http://localhost:{ELASTICSEARCH_PORT}/_cat/indices?format=json")
                    if indices_response.status_code == 200:
                        indices = indices_response.json()
                        if indices:
                            index_names = [idx.get('index') for idx in indices]
                            print(f"  Indices: {', '.join(index_names)}")
                        else:
                            print(f"  Indices: None")
                except:
                    pass
            else:
                print(
                    f"  API: ⚠️ Responding with error code {response.status_code}")
        except requests.RequestException as e:
            print(f"  API: ❌ Not responding ({e})")

    elif status == 'stopped':
        print(f"  Container: ⚠️ Stopped")
        if info:
            print(f"  Created: {info.get('Created', 'Unknown')}")

    elif status == 'not_found':
        print(f"  Container: ❌ Not found")

    else:
        print(f"  Container: ❓ Status check failed")

    # Display volume info if available
    if info and 'Mounts' in info:
        for mount in info.get('Mounts', []):
            if mount.get('Destination', '') == '/usr/share/elasticsearch/data':
                print(f"  Data directory: {mount.get('Source', 'Unknown')}")
                break


def main():
    """Main function to check vector database status."""
    print("🔍 Checking Vector Database Status")

    # Check Qdrant status
    check_qdrant_status()

    # Check Elasticsearch status
    check_elasticsearch_status()

    print("\n💡 Tips:")
    print("  - To start services: python scripts/init_vector_db.py")
    print("  - To stop services: python scripts/stop_vector_db.py")


if __name__ == "__main__":
    main()
