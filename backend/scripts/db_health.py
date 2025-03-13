"""
Script to check the health status of all database systems for the ToolXpert.
Checks MySQL, Qdrant, Elasticsearch, Kibana, and MinIO status.
"""
import sys
import os
import subprocess
import requests
import json
from sqlalchemy import create_engine, text

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import application modules
try:
    import config
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Failed to import. Current path: {sys.path}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Constants for Docker containers
QDRANT_CONTAINER_NAME = "qdrant"
ELASTICSEARCH_CONTAINER_NAME = "elasticsearch"
KIBANA_CONTAINER_NAME = "kibana"
MINIO_CONTAINER_NAME = "minio"

# Ports
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
KIBANA_PORT = int(os.getenv("KIBANA_PORT", "5601"))
MINIO_API_PORT = int(os.getenv("MINIO_API_PORT", "9000"))
MINIO_CONSOLE_PORT = int(os.getenv("MINIO_CONSOLE_PORT", "9001"))

# MinIO credentials
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
DOCUMENT_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "documents")


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


def check_mysql_status():
    """Check MySQL database status."""
    print("📋 Checking MySQL Database:")

    try:
        # Create connection string with database
        conn_string = f"mysql+pymysql://{config.DB_CONFIG['user']}:{config.DB_CONFIG['password']}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['database']}"

        # Connect to MySQL server
        engine = create_engine(conn_string)

        with engine.connect() as conn:
            # Check if connection works
            result = conn.execute(text("SELECT 1"))
            print(f"  • Connection: ✅ Connected successfully")

            # Get server info
            result = conn.execute(text("SELECT VERSION()"))
            version = result.scalar()
            print(f"  • Version: {version}")

            # Get database size
            result = conn.execute(text(
                f"SELECT table_schema, ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) as size_mb FROM information_schema.tables WHERE table_schema = '{config.DB_CONFIG['database']}' GROUP BY table_schema"))
            row = result.fetchone()
            if row:
                print(f"  • Database Size: {row[1]} MB")

            # Get table count
            result = conn.execute(text(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{config.DB_CONFIG['database']}'"))
            table_count = result.scalar()
            print(f"  • Tables: {table_count}")

            # List tables and row counts
            result = conn.execute(text(f"""
                SELECT table_name, table_rows
                FROM information_schema.tables
                WHERE table_schema = '{config.DB_CONFIG['database']}'
                ORDER BY table_rows DESC
            """))
            tables = result.fetchall()

            if tables:
                print("  • Table Statistics:")
                for table_name, row_count in tables:
                    print(f"    - {table_name}: ~{row_count} rows")

            return True

    except Exception as e:
        print(f"  • Connection: ❌ Failed - {str(e)}")
        return False


def check_qdrant_status():
    """Check Qdrant vector database status."""
    print("\n📋 Checking Qdrant Vector Database:")

    # Check container status first
    container_status = check_container_status(QDRANT_CONTAINER_NAME)
    if container_status == 'running':
        print(f"  • Container: ✅ Running")
    elif container_status == 'stopped':
        print(f"  • Container: ⏹️ Stopped")
        return False
    else:
        print(f"  • Container: ❓ Not found")
        return False

    # Check API connection
    try:
        # Basic health check
        response = requests.get(f"http://localhost:{QDRANT_PORT}/healthz")
        if response.status_code == 200:
            print(f"  • Health Check: ✅ Healthy")
        else:
            print(
                f"  • Health Check: ⚠️ Returned status {response.status_code}")

        # Get collections
        response = requests.get(f"http://localhost:{QDRANT_PORT}/collections")
        if response.status_code == 200:
            collections = response.json().get("result", {}).get("collections", [])
            collection_names = [c["name"] for c in collections]
            print(f"  • Collections: {len(collections)}")
            if collection_names:
                print(f"    - Names: {', '.join(collection_names)}")

            # For each collection, get more info
            for collection_name in collection_names:
                try:
                    response = requests.get(
                        f"http://localhost:{QDRANT_PORT}/collections/{collection_name}")
                    if response.status_code == 200:
                        collection_info = response.json().get("result", {})
                        vector_size = collection_info.get("config", {}).get(
                            "params", {}).get("vectors", {}).get("size")
                        vector_distance = collection_info.get("config", {}).get(
                            "params", {}).get("vectors", {}).get("distance")
                        print(
                            f"    - {collection_name}: {vector_size}d vectors, {vector_distance} distance")
                except:
                    pass

        # Print UI URL
        print(f"  • Dashboard URL: http://localhost:{QDRANT_PORT}/dashboard")
        return True

    except requests.RequestException as e:
        print(f"  • API Connection: ❌ Failed - {str(e)}")
        return False


def check_elasticsearch_status():
    """Check Elasticsearch database status."""
    print("\n📋 Checking Elasticsearch:")

    # Check container status first
    container_status = check_container_status(ELASTICSEARCH_CONTAINER_NAME)
    if container_status == 'running':
        print(f"  • Container: ✅ Running")
    elif container_status == 'stopped':
        print(f"  • Container: ⏹️ Stopped")
        return False
    else:
        print(f"  • Container: ❓ Not found")
        return False

    # Check API connection
    try:
        # Basic info
        response = requests.get(f"http://localhost:{ELASTICSEARCH_PORT}")
        if response.status_code == 200:
            es_info = response.json()
            version = es_info.get("version", {}).get("number", "unknown")
            cluster_name = es_info.get("cluster_name", "unknown")
            print(f"  • Version: {version}")
            print(f"  • Cluster: {cluster_name}")
        else:
            print(
                f"  • API Connection: ⚠️ Returned status {response.status_code}")
            return False

        # Check cluster health
        response = requests.get(
            f"http://localhost:{ELASTICSEARCH_PORT}/_cluster/health")
        if response.status_code == 200:
            health = response.json()
            status = health.get("status", "unknown")
            status_icon = "✅" if status == "green" else "⚠️" if status == "yellow" else "❌"
            print(f"  • Cluster Health: {status_icon} {status}")
            print(f"    - Nodes: {health.get('number_of_nodes', 0)}")
            print(f"    - Data Nodes: {health.get('number_of_data_nodes', 0)}")

        # Get indices
        response = requests.get(
            f"http://localhost:{ELASTICSEARCH_PORT}/_cat/indices?format=json")
        if response.status_code == 200:
            indices = response.json()
            print(f"  • Indices: {len(indices)}")
            for idx in indices:
                health = idx.get("health", "")
                status_icon = "✅" if health == "green" else "⚠️" if health == "yellow" else "❌"
                print(
                    f"    - {idx.get('index', 'unknown')}: {status_icon} {idx.get('docs.count', '0')} docs, {idx.get('store.size', '0')} size")

        # Print API and Kibana URLs
        print(f"  • API URL: http://localhost:{ELASTICSEARCH_PORT}")

        # Check Kibana
        kibana_status = check_container_status(KIBANA_CONTAINER_NAME)
        if kibana_status == 'running':
            print(f"  • Kibana: ✅ Running (http://localhost:{KIBANA_PORT})")
        elif kibana_status == 'stopped':
            print(f"  • Kibana: ⏹️ Stopped")
        else:
            print(f"  • Kibana: ❓ Not found")

        return True

    except requests.RequestException as e:
        print(f"  • API Connection: ❌ Failed - {str(e)}")
        return False


def check_minio_status():
    """Check MinIO object storage status."""
    print("\n📋 Checking MinIO Object Storage:")

    # Check container status first
    container_status = check_container_status(MINIO_CONTAINER_NAME)
    if container_status == 'running':
        print(f"  • Container: ✅ Running")
    elif container_status == 'stopped':
        print(f"  • Container: ⏹️ Stopped")
        return False
    else:
        print(f"  • Container: ❓ Not found")
        return False

    # Check API connection
    try:
        # Health check
        response = requests.get(
            f"http://localhost:{MINIO_API_PORT}/minio/health/live")
        if response.status_code == 200:
            print(f"  • Health Check: ✅ Healthy")
        else:
            print(
                f"  • Health Check: ⚠️ Returned status {response.status_code}")

        # Check MinIO buckets using mc (MinIO Client)
        try:
            # Add alias
            subprocess.run([
                "docker", "run", "--rm", "--network=host",
                "minio/mc", "alias", "set", "myminio",
                f"http://localhost:{MINIO_API_PORT}",
                MINIO_ACCESS_KEY, MINIO_SECRET_KEY
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # List buckets
            result = subprocess.run([
                "docker", "run", "--rm", "--network=host",
                "minio/mc", "ls", "myminio"
            ], check=True, stdout=subprocess.PIPE, text=True)

            buckets = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        buckets.append(parts[-1])

            print(f"  • Buckets: {len(buckets)}")
            if buckets:
                print(f"    - Names: {', '.join(buckets)}")

                # Check documents bucket specifically
                if DOCUMENT_BUCKET_NAME in buckets:
                    # Get count of objects in documents bucket
                    result = subprocess.run([
                        "docker", "run", "--rm", "--network=host",
                        "minio/mc", "stat", f"myminio/{DOCUMENT_BUCKET_NAME}"
                    ], check=True, stdout=subprocess.PIPE, text=True)

                    # Try to extract object count
                    for line in result.stdout.strip().split('\n'):
                        if "counts" in line.lower():
                            print(
                                f"    - {DOCUMENT_BUCKET_NAME}: {line.strip()}")
                            break
        except Exception as e:
            print(f"  • Bucket Info: ⚠️ Error - {str(e)}")

        # Print URLs
        print(f"  • API URL: http://localhost:{MINIO_API_PORT}")
        print(f"  • Console URL: http://localhost:{MINIO_CONSOLE_PORT}")
        print(f"    - Login with: {MINIO_ACCESS_KEY} / {MINIO_SECRET_KEY}")

        return True

    except requests.RequestException as e:
        print(f"  • API Connection: ❌ Failed - {str(e)}")
        return False


def print_system_overview():
    """Print overall system status overview."""
    print("\n📊 System Overview:")

    # Check each component status
    components = [
        ("MySQL Database", check_container_status("mysql") == "running"),
        ("Qdrant Vector DB", check_container_status(
            QDRANT_CONTAINER_NAME) == "running"),
        ("Elasticsearch", check_container_status(
            ELASTICSEARCH_CONTAINER_NAME) == "running"),
        ("Kibana UI", check_container_status(KIBANA_CONTAINER_NAME) == "running"),
        ("MinIO Storage", check_container_status(
            MINIO_CONTAINER_NAME) == "running")
    ]

    # Print status table
    for component, status in components:
        status_icon = "✅" if status else "❌"
        print(f"  • {component}: {status_icon}")

    # Print overall status
    all_running = all(status for _, status in components)
    if all_running:
        print("\n🟢 All systems operational")
    else:
        print("\n🟠 Some systems are not running")
        print("  Run 'python scripts/init_db.py' to start all components")


def main():
    """Main function to check database health."""
    print("🔍 ToolXpert Database Health Check")

    try:
        # First show overview
        print_system_overview()

        # Check each component in detail
        check_mysql_status()
        check_qdrant_status()
        check_elasticsearch_status()
        check_minio_status()

        print("\n✅ Health check complete")
        print("  • To start databases: python scripts/init_db.py")
        print("  • To stop databases: python scripts/stop_db.py")

    except Exception as e:
        print(f"\n❌ Error during health check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
