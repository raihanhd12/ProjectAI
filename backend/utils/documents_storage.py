"""
Document storage utility for storing and retrieving documents using MinIO.
"""
import os
import io
import uuid
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
        self.endpoint = endpoint or config.MINIO_ENDPOINT.replace(
            "http://", "")
        self.access_key = access_key or config.MINIO_ACCESS_KEY
        self.secret_key = secret_key or config.MINIO_SECRET_KEY
        self.secure = secure or config.MINIO_SECURE
        self.bucket_name = bucket_name or config.MINIO_BUCKET_NAME

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

    async def upload_file(self, file: UploadFile, metadata: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Upload a file to MinIO.

        Args:
            file: The uploaded file
            metadata: Optional metadata for the object

        Returns:
            Tuple[bool, str]: Success flag and object name
        """
        try:
            # Generate unique object name
            ext = os.path.splitext(file.filename)[1]
            object_name = f"{uuid.uuid4()}{ext}"

            # Get content type
            content_type = file.content_type or "application/octet-stream"

            # Read file content
            content = await file.read()

            # Upload to MinIO
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=io.BytesIO(content),
                length=len(content),
                content_type=content_type,
                metadata=metadata
            )

            return True, object_name
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False, ""

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
