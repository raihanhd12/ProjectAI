"""
Document storage utility for storing and retrieving documents using MinIO.
"""
import os
import io
import uuid
import tempfile
from typing import BinaryIO, List, Dict, Any, Optional, Tuple
from fastapi import UploadFile
from minio import Minio
from minio.error import S3Error
import mimetypes

import config


class DocumentStorage:
    """Document storage manager using MinIO."""

    def __init__(self,
                 endpoint: str = None,
                 access_key: str = None,
                 secret_key: str = None,
                 secure: bool = False,
                 bucket_name: str = None):
        """
        Initialize document storage connection.

        Args:
            endpoint: MinIO endpoint (e.g., "localhost:9000")
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Use HTTPS instead of HTTP
            bucket_name: Name of the bucket to use
        """
        # Get configuration from environment or parameters
        self.endpoint = endpoint or os.getenv(
            "MINIO_ENDPOINT", "localhost:9000").replace("http://", "")
        self.access_key = access_key or os.getenv(
            "MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv(
            "MINIO_SECRET_KEY", "minioadmin")
        self.secure = secure or os.getenv(
            "MINIO_SECURE", "False").lower() == "true"
        self.bucket_name = bucket_name or os.getenv(
            "MINIO_BUCKET_NAME", "documents")

        # Initialize MinIO client
        self.client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

        # Ensure bucket exists
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> bool:
        """
        Ensure the bucket exists, create it if it doesn't.

        Returns:
            bool: True if successful
        """
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Created bucket: {self.bucket_name}")
            return True
        except S3Error as e:
            print(f"Error ensuring bucket exists: {e}")
            return False

    def upload_file(self, file_path: str, object_name: str = None, metadata: Dict = None) -> Tuple[bool, str]:
        """
        Upload a file to MinIO.

        Args:
            file_path: Path to the file
            object_name: Custom object name (defaults to filename)
            metadata: Optional metadata for the object

        Returns:
            Tuple: (Success flag, object name)
        """
        try:
            if object_name is None:
                object_name = os.path.basename(file_path)

            # Get file size
            file_size = os.path.getsize(file_path)

            # Guess content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = 'application/octet-stream'

            # Upload the file
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=file_path,
                content_type=content_type,
                metadata=metadata
            )
            return True, object_name
        except S3Error as e:
            print(f"Error uploading file: {e}")
            return False, None

    def upload_fileobj(self, file_obj: BinaryIO, object_name: str,
                       content_type: str = None, metadata: Dict = None) -> Tuple[bool, str]:
        """
        Upload a file-like object to MinIO.

        Args:
            file_obj: File-like object
            object_name: Object name
            content_type: Content type of the file
            metadata: Optional metadata for the object

        Returns:
            Tuple: (Success flag, object name)
        """
        try:
            # Default content type if not provided
            if content_type is None:
                content_type = 'application/octet-stream'

            # Get length of the file object
            file_obj.seek(0, os.SEEK_END)
            file_size = file_obj.tell()
            file_obj.seek(0)

            # Upload the file
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file_obj,
                length=file_size,
                content_type=content_type,
                metadata=metadata
            )
            return True, object_name
        except S3Error as e:
            print(f"Error uploading file object: {e}")
            return False, None

    async def upload_fastapi_file(self, upload_file: UploadFile,
                                  object_name: str = None,
                                  metadata: Dict = None) -> Tuple[bool, str]:
        """
        Upload a FastAPI UploadFile to MinIO.

        Args:
            upload_file: FastAPI UploadFile
            object_name: Custom object name (defaults to unique name)
            metadata: Optional metadata for the object

        Returns:
            Tuple: (Success flag, object name)
        """
        try:
            # Generate unique name if not provided
            if object_name is None:
                file_ext = os.path.splitext(upload_file.filename)[1]
                object_name = f"{str(uuid.uuid4())}{file_ext}"

            # Get content type from the upload file
            content_type = upload_file.content_type

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # Write uploaded file content to the temp file
                contents = await upload_file.read()
                temp_file.write(contents)
                temp_file_path = temp_file.name

            # Upload using the temporary file
            success, obj_name = self.upload_file(
                file_path=temp_file_path,
                object_name=object_name,
                metadata=metadata
            )

            # Remove the temporary file
            os.unlink(temp_file_path)

            return success, obj_name
        except Exception as e:
            print(f"Error uploading FastAPI file: {e}")
            return False, None

    def download_file(self, object_name: str, file_path: str) -> bool:
        """
        Download a file from MinIO.

        Args:
            object_name: Name of the object
            file_path: Path to save the file

        Returns:
            bool: True if successful
        """
        try:
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=file_path
            )
            return True
        except S3Error as e:
            print(f"Error downloading file: {e}")
            return False

    def get_file_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for object access.

        Args:
            object_name: Name of the object
            expires: Expiration time in seconds

        Returns:
            str: Presigned URL or None if failed
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=expires
            )
            return url
        except S3Error as e:
            print(f"Error generating presigned URL: {e}")
            return None

    def delete_file(self, object_name: str) -> bool:
        """
        Delete a file from MinIO.

        Args:
            object_name: Name of the object

        Returns:
            bool: True if successful
        """
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            return True
        except S3Error as e:
            print(f"Error deleting file: {e}")
            return False

    def list_files(self, prefix: str = "", recursive: bool = True) -> List[Dict]:
        """
        List files in the bucket.

        Args:
            prefix: Prefix to filter objects
            recursive: If true, recursively list objects

        Returns:
            List[Dict]: List of objects with metadata
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=recursive
            )

            result = []
            for obj in objects:
                # Get object metadata
                try:
                    stat = self.client.stat_object(
                        bucket_name=self.bucket_name,
                        object_name=obj.object_name
                    )

                    result.append({
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified,
                        "content_type": stat.content_type,
                        "metadata": stat.metadata
                    })
                except:
                    # If stat fails, just add basic info
                    result.append({
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified
                    })

            return result
        except S3Error as e:
            print(f"Error listing files: {e}")
            return []

    def get_file_content(self, object_name: str) -> Optional[bytes]:
        """
        Get file content as bytes.

        Args:
            object_name: Name of the object

        Returns:
            bytes: File content or None if failed
        """
        try:
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            print(f"Error getting file content: {e}")
            return None
