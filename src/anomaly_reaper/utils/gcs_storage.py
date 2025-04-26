"""
Google Cloud Storage utility for Anomaly Reaper.

This module provides functions to interact with GCS buckets for image storage and retrieval.
"""

import os
import tempfile
from typing import Tuple
from google.cloud import storage
from google.oauth2 import service_account

from anomaly_reaper.config import settings, logger


def validate_gcs_config() -> bool:
    """
    Validate that the required GCS configuration is present.

    Returns
    -------
    bool
        True if configuration is valid, False otherwise
    """
    if not settings.project_id:
        logger.error(
            "GCS project_id not configured. Set ANOMALY_REAPER_PROJECT_ID in .env"
        )
        return False

    if not settings.gcs_bucket_name:
        logger.error(
            "GCS bucket_name not configured. Set ANOMALY_REAPER_GCS_BUCKET_NAME in .env"
        )
        return False

    return True


# Initialize GCS client
def get_gcs_client() -> storage.Client:
    """
    Get a Google Cloud Storage client instance.

    Returns
    -------
    storage.Client
        Authenticated GCS client

    Raises
    ------
    ValueError
        If required GCS configuration is missing
    """
    if not validate_gcs_config():
        raise ValueError(
            "Missing GCS configuration. Check .env file for PROJECT_ID and GCS_BUCKET_NAME."
        )

    # Check if service account credentials are provided
    if settings.gcp_service_account_path and os.path.exists(
        settings.gcp_service_account_path
    ):
        logger.info(
            f"Using service account credentials from {settings.gcp_service_account_path}"
        )
        credentials = service_account.Credentials.from_service_account_file(
            settings.gcp_service_account_path
        )
        return storage.Client(credentials=credentials, project=settings.project_id)

    # Otherwise use default authentication (ADC)
    logger.info("Using application default credentials for GCS")
    return storage.Client(project=settings.project_id)


def upload_image_to_gcs(
    file_content: bytes, filename: str, content_type: str = "image/jpeg"
) -> str:
    """
    Upload an image to the GCS bucket.

    Parameters
    ----------
    file_content : bytes
        The image file content
    filename : str
        The name to give the file in GCS
    content_type : str
        The MIME type of the image

    Returns
    -------
    str
        The public URL of the uploaded image

    Raises
    ------
    ValueError
        If required GCS configuration is missing
    """
    if not validate_gcs_config():
        raise ValueError(
            "Missing GCS configuration. Check .env file for PROJECT_ID and GCS_BUCKET_NAME."
        )

    try:
        # Get client and bucket
        client = get_gcs_client()
        bucket = client.bucket(settings.gcs_bucket_name)

        # Create blob with full path in the bucket
        blob_name = f"{settings.gcs_images_prefix}/{filename}"
        blob = bucket.blob(blob_name)

        # Upload the file
        blob.upload_from_string(file_content, content_type=content_type)

        # If bucket is public, we could use public URL
        if settings.gcs_use_public_urls:
            return (
                f"https://storage.googleapis.com/{settings.gcs_bucket_name}/{blob_name}"
            )

        # Otherwise use the GCS path format
        return f"gs://{settings.gcs_bucket_name}/{blob_name}"

    except Exception as e:
        logger.error(f"Error uploading image to GCS: {str(e)}")
        raise


def download_image_from_gcs(gcs_path: str) -> Tuple[bytes, str]:
    """
    Download an image from GCS.

    Parameters
    ----------
    gcs_path : str
        The GCS path or URL of the image

    Returns
    -------
    Tuple[bytes, str]
        The image data and its content type

    Raises
    ------
    ValueError
        If required GCS configuration is missing for default bucket
    """
    try:
        # Parse the GCS path to extract bucket and blob name
        if gcs_path.startswith("gs://"):
            # Format: gs://bucket-name/path/to/file
            bucket_name = gcs_path.replace("gs://", "").split("/")[0]
            blob_name = "/".join(
                gcs_path.replace(f"gs://{bucket_name}/", "").split("/")
            )
        elif gcs_path.startswith("https://storage.googleapis.com/"):
            # Format: https://storage.googleapis.com/bucket-name/path/to/file
            path_parts = gcs_path.replace("https://storage.googleapis.com/", "").split(
                "/"
            )
            bucket_name = path_parts[0]
            blob_name = "/".join(path_parts[1:])
        else:
            # Assume it's already a blob name in our default bucket
            if not validate_gcs_config():
                raise ValueError(
                    "Missing GCS configuration. Check .env file for PROJECT_ID and GCS_BUCKET_NAME."
                )
            bucket_name = settings.gcs_bucket_name
            blob_name = gcs_path

        # Get client and blob
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download content
        content = blob.download_as_bytes()

        # Determine content type based on file extension
        content_type = "image/jpeg"  # Default
        lower_name = blob_name.lower()
        if lower_name.endswith(".png"):
            content_type = "image/png"
        elif lower_name.endswith(".gif"):
            content_type = "image/gif"
        elif lower_name.endswith(".fits") or lower_name.endswith(".fit"):
            content_type = "image/fits"

        return content, content_type

    except Exception as e:
        logger.error(f"Error downloading image from GCS: {str(e)}")
        raise


def is_gcs_path(path: str) -> bool:
    """
    Check if a path is a GCS path.

    Parameters
    ----------
    path : str
        The path to check

    Returns
    -------
    bool
        True if the path is a GCS path, False otherwise
    """
    return path.startswith("gs://") or path.startswith(
        "https://storage.googleapis.com/"
    )


def get_local_copy(gcs_path: str) -> str:
    """
    Download a GCS file to a local temporary file.

    Parameters
    ----------
    gcs_path : str
        The GCS path to download

    Returns
    -------
    str
        Path to the local temporary file
    """
    content, content_type = download_image_from_gcs(gcs_path)

    # Determine file extension from content type
    extension = ".jpg"
    if content_type == "image/png":
        extension = ".png"
    elif content_type == "image/gif":
        extension = ".gif"
    elif content_type == "image/fits":
        extension = ".fits"

    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix=extension)
    with os.fdopen(fd, "wb") as temp_file:
        temp_file.write(content)

    return temp_path
