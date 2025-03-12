"""
Script to stop Qdrant and Elasticsearch containers.
"""
import sys
import os
import subprocess
import time

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Constants
QDRANT_CONTAINER_NAME = "qdrant"
ELASTICSEARCH_CONTAINER_NAME = "elasticsearch"


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


def stop_container(container_name):
    """Stop a Docker container if it's running."""
    status = check_container_status(container_name)

    if status == 'not_found':
        print(f"✅ Container '{container_name}' not found, nothing to stop")
        return True

    if status == 'stopped':
        print(f"✅ Container '{container_name}' already stopped")
        return True

    if status == 'running':
        try:
            print(f"🛑 Stopping '{container_name}' container...")
            subprocess.run(["docker", "stop", container_name], check=True)
            print(f"✅ Container '{container_name}' stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to stop '{container_name}' container: {e}")
            return False

    print(f"❌ Failed to determine status of '{container_name}' container")
    return False


def wait_for_container_stop(container_name, max_retries=10, retry_interval=1):
    """Wait for container to be fully stopped."""
    for i in range(max_retries):
        status = check_container_status(container_name)
        if status in ('stopped', 'not_found'):
            return True
        print(
            f"⏳ Waiting for '{container_name}' to stop ({i+1}/{max_retries})...")
        time.sleep(retry_interval)
    return False


def main():
    """Main function to stop vector database containers."""
    print("🛑 Stopping Vector Database Containers\n")

    # Stop Elasticsearch (stop this first since it's more resource-intensive)
    es_stopped = stop_container(ELASTICSEARCH_CONTAINER_NAME)
    if es_stopped:
        wait_for_container_stop(ELASTICSEARCH_CONTAINER_NAME)

    # Stop Qdrant
    qdrant_stopped = stop_container(QDRANT_CONTAINER_NAME)
    if qdrant_stopped:
        wait_for_container_stop(QDRANT_CONTAINER_NAME)

    if es_stopped and qdrant_stopped:
        print("\n✅ Successfully stopped all vector database containers")
    else:
        print("\n⚠️ Some containers may still be running")
        print("   Check status with: python scripts/vector_db_status.py")


if __name__ == "__main__":
    main()
