"""
Script to stop all database services for the ToolXpert.
Stops Qdrant, Elasticsearch, Kibana, and MinIO containers.
"""
import sys
import os
import time
import subprocess
from typing import List

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Constants for Docker containers
CONTAINER_NAMES = [
    "minio",      # Document Storage
    "kibana",     # Elasticsearch UI
    "elasticsearch",  # Keyword Search
    "qdrant"      # Vector Database
]


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


def stop_all_containers():
    """Stop all database containers in the correct order."""
    print("🛑 Stopping all database containers")

    # Stop containers in a specific order (reverse of startup)
    # This ensures dependencies are respected
    stopped_containers = []
    failed_containers = []

    for container in CONTAINER_NAMES:
        print(f"\n📦 Processing container: {container}")
        if stop_container(container):
            wait_for_container_stop(container)
            stopped_containers.append(container)
        else:
            failed_containers.append(container)

    # Print summary
    if not failed_containers:
        print("\n✅ All containers stopped successfully")
    else:
        print(
            f"\n⚠️ Stopped {len(stopped_containers)} containers, but {len(failed_containers)} failed")
        print(f"Failed containers: {', '.join(failed_containers)}")


def print_container_status():
    """Print the status of all database containers."""
    print("\n📊 Current Container Status:")

    for container in CONTAINER_NAMES:
        status = check_container_status(container)
        if status == 'running':
            print(f"  • {container}: ✅ Running")
        elif status == 'stopped':
            print(f"  • {container}: ⏹️ Stopped")
        elif status == 'not_found':
            print(f"  • {container}: ❓ Not found")
        else:
            print(f"  • {container}: ❌ Error checking status")


def main():
    """Main function to stop all database services."""
    print("🔄 ToolXpert Database Shutdown")

    try:
        # First show current status
        print_container_status()

        # Stop all containers
        stop_all_containers()

        # Show final status
        print_container_status()

    except Exception as e:
        print(f"❌ Error during shutdown: {e}")
        sys.exit(1)

    print("\n💾 Database services shutdown complete")
    print("To restart the databases, run: python scripts/init_db.py")


if __name__ == "__main__":
    main()
