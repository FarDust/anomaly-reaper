"""
Data Transfer Objects (DTOs) for the anomaly_reaper application.

This module contains all the Pydantic models used for API request/response
validation and serialization in the anomaly detection and classification system.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
import datetime
from enum import Enum


class ImageResponse(BaseModel):
    """
    Pydantic model for image response data.

    This model defines the structure of the API response when returning
    image anomaly detection results.

    Attributes
    ----------
    id : str
        Unique identifier for the image
    filename : str
        Original filename of the image
    timestamp : datetime
        When the image was processed
    reconstruction_error : float
        PCA reconstruction error for the image
    is_anomaly : bool
        Whether the image was classified as an anomaly
    anomaly_score : float
        Normalized anomaly score
    path : str
        Path where the image is stored

    Examples
    --------
    >>> image_response = ImageResponse(
    ...     id="550e8400-e29b-41d4-a716-446655440000",
    ...     filename="example.jpg",
    ...     timestamp=datetime.datetime.now(),
    ...     reconstruction_error=0.123,
    ...     is_anomaly=True,
    ...     anomaly_score=2.5,
    ...     path="/uploads/example.jpg"
    ... )
    >>> print(image_response.is_anomaly)
    True
    """

    id: str
    filename: str
    timestamp: datetime.datetime
    reconstruction_error: float
    is_anomaly: bool
    anomaly_score: float
    path: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "filename": "example.jpg",
                    "timestamp": "2025-04-25T12:34:56.789Z",
                    "reconstruction_error": 0.123,
                    "is_anomaly": True,
                    "anomaly_score": 2.5,
                    "path": "/uploads/550e8400-e29b-41d4-a716-446655440000.jpg",
                }
            ]
        }
    )


class ClassificationRequest(BaseModel):
    """
    Pydantic model for classification request data.

    This model defines the structure of the API request when submitting
    a classification for an image.

    Attributes
    ----------
    user_classification : bool
        User's classification (True for anomaly, False for normal)
    comment : str, optional
        Optional comment about the classification

    Examples
    --------
    >>> classification = ClassificationRequest(
    ...     user_classification=True,
    ...     comment="This image shows an unusual pattern in the top right corner."
    ... )
    >>> print(classification.model_dump())
    {'user_classification': True, 'comment': 'This image shows an unusual pattern in the top right corner.'}
    """

    user_classification: bool = Field(
        ..., description="True if image is anomalous, False if normal"
    )
    comment: Optional[str] = Field(
        None, description="Optional comment about the classification"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "user_classification": True,
                    "comment": "This image shows an unusual pattern in the top right corner.",
                }
            ]
        }
    )


class ClassificationResponse(BaseModel):
    """
    Pydantic model for classification response data.

    This model defines the structure of the API response when returning
    information about an image classification.

    Attributes
    ----------
    id : str
        Unique identifier for the classification
    image_id : str
        ID of the classified image
    user_classification : bool
        User's classification (True for anomaly, False for normal)
    comment : str, optional
        Optional comment about the classification
    timestamp : datetime
        When the classification was made

    Examples
    --------
    >>> classification_response = ClassificationResponse(
    ...     id="550e8400-e29b-41d4-a716-446655440000",
    ...     image_id="550e8400-e29b-41d4-a716-446655440000",
    ...     user_classification=True,
    ...     comment="This image shows an unusual pattern in the top right corner.",
    ...     timestamp=datetime.datetime.now()
    ... )
    >>> print(classification_response.user_classification)
    True
    """

    id: str
    image_id: str
    user_classification: bool
    comment: Optional[str]
    timestamp: datetime.datetime

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "image_id": "550e8400-e29b-41d4-a716-446655440000",
                    "user_classification": True,
                    "comment": "This image shows an unusual pattern in the top right corner.",
                    "timestamp": "2025-04-25T12:34:56.789Z",
                }
            ]
        }
    )


class StorageConfigResponse(BaseModel):
    """
    Pydantic model for storage configuration response.

    Attributes
    ----------
    storage_type : str
        Type of storage being used (e.g., "gcs" or "local")
    storage_location : str
        Location where images are being stored
    use_cloud_storage: bool
        Whether cloud storage is enabled
    gcs_bucket_name: Optional[str]
        Name of the GCS bucket if cloud storage is enabled
    """

    storage_type: str
    storage_location: str
    use_cloud_storage: bool
    gcs_bucket_name: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "storage_type": "gcs",
                    "storage_location": "gs://my-bucket/anomaly_reaper/images",
                    "use_cloud_storage": True,
                    "gcs_bucket_name": "my-bucket",
                },
                {
                    "storage_type": "local",
                    "storage_location": "/home/user/anomaly_reaper/uploads",
                    "use_cloud_storage": False,
                    "gcs_bucket_name": None,
                },
            ]
        }
    )


class ImageUrlResponse(BaseModel):
    """
    Pydantic model for image URL response.

    Attributes
    ----------
    id : str
        Image ID
    url : str
        URL to access the image
    is_temporary: bool
        Whether the URL is temporary (will expire)
    expires_at: Optional[datetime.datetime]
        When the URL will expire, if temporary
    """

    id: str
    url: str
    is_temporary: bool = False
    expires_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "url": "https://storage.googleapis.com/my-bucket/anomaly_reaper/images/550e8400-e29b-41d4-a716-446655440000.jpg",
                    "is_temporary": True,
                    "expires_at": "2025-04-25T13:34:56.789Z",
                }
            ]
        }
    )


class BatchProcessRequest(BaseModel):
    """
    Pydantic model for batch processing request.

    Attributes
    ----------
    file_paths: List[str]
        List of file paths to process
    anomaly_threshold: Optional[float]
        Custom threshold for anomaly detection
    """

    file_paths: List[str]
    anomaly_threshold: Optional[float] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "file_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
                    "anomaly_threshold": 0.1,
                }
            ]
        }
    )


class BatchProcessResponse(BaseModel):
    """
    Pydantic model for batch processing response.

    Attributes
    ----------
    processed_count: int
        Number of images processed
    anomaly_count: int
        Number of anomalies detected
    results: List[ImageResponse]
        Processing results for each image
    """

    processed_count: int
    anomaly_count: int
    results: List[ImageResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "processed_count": 2,
                    "anomaly_count": 1,
                    "results": [
                        {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "filename": "image1.jpg",
                            "timestamp": "2025-04-25T12:34:56.789Z",
                            "reconstruction_error": 0.123,
                            "is_anomaly": True,
                            "anomaly_score": 2.5,
                            "path": "/uploads/550e8400-e29b-41d4-a716-446655440000.jpg",
                        }
                    ],
                }
            ]
        }
    )


class CloudStorageConfigRequest(BaseModel):
    """
    Pydantic model for configuring cloud storage settings.

    Attributes
    ----------
    use_cloud_storage: bool
        Whether to use cloud storage
    gcs_bucket_name: str
        GCS bucket name
    gcs_use_public_urls: bool
        Whether to use public URLs for GCS images
    """

    use_cloud_storage: bool
    gcs_bucket_name: str
    gcs_use_public_urls: bool = True

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "use_cloud_storage": True,
                    "gcs_bucket_name": "my-bucket",
                    "gcs_use_public_urls": True,
                }
            ]
        }
    )


class ImageClassificationRequest(BaseModel):
    """
    Pydantic model for image classification request.

    Attributes
    ----------
    is_anomaly : bool
        Whether the image is an anomaly (True) or normal (False)
    comment : str, optional
        Optional comment about the classification
    """

    is_anomaly: bool = Field(
        ..., description="True if image is anomalous, False otherwise"
    )
    comment: Optional[str] = Field(
        None, description="Optional comment about the classification"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "is_anomaly": True,
                    "comment": "This appears to be an anomaly due to the unusual brightness pattern.",
                }
            ]
        }
    )


class ImageClassificationResponse(BaseModel):
    """
    Pydantic model for image classification response.

    Attributes
    ----------
    id : str
        ID of the classified image
    user_classification : bool
        Whether the image was classified as an anomaly by the user
    comment : str, optional
        Optional comment about the classification
    """

    id: str
    user_classification: bool
    comment: Optional[str]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "user_classification": True,
                    "comment": "This appears to be an anomaly due to the unusual brightness pattern.",
                }
            ]
        }
    )


class HealthCheckResponse(BaseModel):
    """
    Pydantic model for health check response.

    Attributes
    ----------
    status : str
        Status of the API (e.g., "ok")
    timestamp : datetime.datetime
        Current timestamp
    version : str
        API version
    """

    status: str
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    version: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "timestamp": "2025-04-25T12:34:56.789Z",
                    "version": "1.0.0",
                }
            ]
        }
    )


class StatisticsResponse(BaseModel):
    """
    Pydantic model for statistics response.

    Attributes
    ----------
    total_images : int
        Total number of images processed
    anomalies_detected : int
        Number of anomalies detected
    classified_images : int
        Number of images that have been classified by users
    average_anomaly_score : float
        Average anomaly score across all images
    storage_type : str
        Type of storage being used (e.g., "Google Cloud Storage" or "Local")
    storage_location : str
        Location where images are being stored
    """

    total_images: int
    anomalies_detected: int
    classified_images: int
    average_anomaly_score: float
    storage_type: str
    storage_location: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "total_images": 100,
                    "anomalies_detected": 10,
                    "classified_images": 50,
                    "average_anomaly_score": 0.3,
                    "storage_type": "Google Cloud Storage",
                    "storage_location": "my-bucket",
                }
            ]
        }
    )


class SyncAnomalyDataResponse(BaseModel):
    """
    Pydantic model for synchronizing anomaly data response.

    Attributes
    ----------
    message : str
        Summary message about the synchronization
    source : str
        Source of the anomaly data
    imported_count : int
        Number of new records imported
    anomaly_count : int
        Number of anomalies among imported records
    skipped_count : int
        Number of records skipped (already in DB)
    error_count : int
        Number of records that had errors
    total_records : int
        Total number of records after synchronization
    """

    message: str
    source: str
    imported_count: int
    anomaly_count: int
    skipped_count: int
    error_count: int
    total_records: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "Successfully synchronized anomaly data",
                    "source": "gs://my-bucket/anomaly_reaper/embeddings_df.csv",
                    "imported_count": 50,
                    "anomaly_count": 5,
                    "skipped_count": 10,
                    "error_count": 0,
                    "total_records": 150,
                }
            ]
        }
    )


class DateRangeFilterRequest(BaseModel):
    """
    Pydantic model for date range filtering.

    Attributes
    ----------
    start_date : Optional[datetime.datetime]
        Start date for filtering (inclusive)
    end_date : Optional[datetime.datetime]
        End date for filtering (inclusive)
    """

    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "start_date": "2025-03-25T00:00:00.000Z",
                    "end_date": "2025-04-25T23:59:59.999Z",
                }
            ]
        }
    )


class AnomalyScoreFilterRequest(BaseModel):
    """
    Pydantic model for anomaly score range filtering.

    Attributes
    ----------
    min_score : Optional[float]
        Minimum anomaly score (inclusive)
    max_score : Optional[float]
        Maximum anomaly score (inclusive)
    """

    min_score: Optional[float] = None
    max_score: Optional[float] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "min_score": 0.5,
                    "max_score": 1.0,
                }
            ]
        }
    )


class ClassificationFilterRequest(BaseModel):
    """
    Pydantic model for classification status filtering.

    Attributes
    ----------
    is_classified : Optional[bool]
        Whether the image has been classified by a user
    user_classification : Optional[bool]
        Filter by user classification (True for anomaly, False for normal)
    """

    is_classified: Optional[bool] = None
    user_classification: Optional[bool] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "is_classified": True,
                    "user_classification": True,
                }
            ]
        }
    )


class AdvancedFilterRequest(BaseModel):
    """
    Pydantic model for advanced image filtering.

    Attributes
    ----------
    date_range : Optional[DateRangeFilterRequest]
        Filter by date range
    anomaly_score : Optional[AnomalyScoreFilterRequest]
        Filter by anomaly score range
    classification : Optional[ClassificationFilterRequest]
        Filter by classification status
    is_anomaly : Optional[bool]
        Filter by model-detected anomaly status
    page : int
        Page number for pagination
    page_size : int
        Number of results per page
    sort_by : str
        Field to sort results by
    sort_order : str
        Sort order ("asc" or "desc")
    """

    date_range: Optional[DateRangeFilterRequest] = None
    anomaly_score: Optional[AnomalyScoreFilterRequest] = None
    classification: Optional[ClassificationFilterRequest] = None
    is_anomaly: Optional[bool] = None
    page: int = 1
    page_size: int = 50
    sort_by: str = "processed_at"
    sort_order: str = "desc"

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "date_range": {
                        "start_date": "2025-03-25T00:00:00.000Z",
                        "end_date": "2025-04-25T23:59:59.999Z",
                    },
                    "anomaly_score": {
                        "min_score": 0.5,
                        "max_score": 1.0,
                    },
                    "classification": {
                        "is_classified": True,
                        "user_classification": True,
                    },
                    "is_anomaly": True,
                    "page": 1,
                    "page_size": 50,
                    "sort_by": "anomaly_score",
                    "sort_order": "desc",
                }
            ]
        }
    )


class AdvancedFilterResponse(BaseModel):
    """
    Pydantic model for advanced filter response.

    Attributes
    ----------
    results : List[ImageResponse]
        List of image records matching the filter criteria
    total_count : int
        Total number of records matching the criteria
    page : int
        Current page number
    page_size : int
        Number of results per page
    total_pages : int
        Total number of pages
    """

    results: List[ImageResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "results": [
                        {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "filename": "example.jpg",
                            "timestamp": "2025-04-25T12:34:56.789Z",
                            "reconstruction_error": 0.123,
                            "is_anomaly": True,
                            "anomaly_score": 2.5,
                            "path": "/uploads/550e8400-e29b-41d4-a716-446655440000.jpg",
                        }
                    ],
                    "total_count": 150,
                    "page": 1,
                    "page_size": 50,
                    "total_pages": 3,
                }
            ]
        }
    )


class BatchClassificationRequest(BaseModel):
    """
    Pydantic model for batch classification request.

    Attributes
    ----------
    image_ids : List[str]
        List of image IDs to classify
    is_anomaly : bool
        Whether the images are anomalous (True) or normal (False)
    comment : Optional[str]
        Optional comment to apply to all classifications
    """

    image_ids: List[str]
    is_anomaly: bool
    comment: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "image_ids": [
                        "550e8400-e29b-41d4-a716-446655440000",
                        "650e8400-e29b-41d4-a716-446655440001",
                    ],
                    "is_anomaly": True,
                    "comment": "Batch classification of multiple similar anomalies",
                }
            ]
        }
    )


class BatchClassificationResponse(BaseModel):
    """
    Pydantic model for batch classification response.

    Attributes
    ----------
    total : int
        Total number of images in the request
    successful : int
        Number of images successfully classified
    failed : int
        Number of images that failed to be classified
    failed_ids : List[str]
        List of image IDs that failed to be classified
    """

    total: int
    successful: int
    failed: int
    failed_ids: List[str]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "total": 2,
                    "successful": 2,
                    "failed": 0,
                    "failed_ids": [],
                }
            ]
        }
    )


class DashboardStatsResponse(BaseModel):
    """
    Pydantic model for dashboard statistics response.

    Attributes
    ----------
    total_images : int
        Total number of images in the system
    total_anomalies : int
        Total number of detected anomalies
    user_confirmed_anomalies : int
        Number of user-confirmed anomalies
    unclassified_anomalies : int
        Number of anomalies not yet classified by users
    false_positives : int
        Number of model anomalies classified as normal by users
    false_negatives : int
        Number of model normal classified as anomalies by users
    recent_activity : List[Dict]
        Recent classification activity
    """

    total_images: int
    total_anomalies: int
    user_confirmed_anomalies: int
    unclassified_anomalies: int
    false_positives: int
    false_negatives: int
    recent_activity: List[Dict]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "total_images": 150,
                    "total_anomalies": 25,
                    "user_confirmed_anomalies": 15,
                    "unclassified_anomalies": 10,
                    "false_positives": 5,
                    "false_negatives": 3,
                    "recent_activity": [
                        {
                            "timestamp": "2025-04-25T12:34:56.789Z",
                            "action": "classification",
                            "is_anomaly": True,
                        }
                    ],
                }
            ]
        }
    )


class ClassificationHistoryResponse(BaseModel):
    """
    Pydantic model for image classification history.

    Attributes
    ----------
    image_id : str
        ID of the image
    classifications : List[ClassificationResponse]
        List of all classifications for this image
    """

    image_id: str
    classifications: List[ClassificationResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "image_id": "550e8400-e29b-41d4-a716-446655440000",
                    "classifications": [
                        {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "image_id": "550e8400-e29b-41d4-a716-446655440000",
                            "user_classification": True,
                            "comment": "Initial classification",
                            "timestamp": "2025-04-25T12:34:56.789Z",
                        },
                        {
                            "id": "650e8400-e29b-41d4-a716-446655440000",
                            "image_id": "550e8400-e29b-41d4-a716-446655440000",
                            "user_classification": False,
                            "comment": "After further review, this is not an anomaly",
                            "timestamp": "2025-04-26T09:12:34.567Z",
                        }
                    ]
                }
            ]
        }
    )


class ExportFormat(str, Enum):
    """
    Enum for export format options.
    """

    JSON = "json"
    CSV = "csv"


class ExportRequest(BaseModel):
    """
    Pydantic model for exporting data.

    Attributes
    ----------
    format : ExportFormat
        Format for export (JSON or CSV)
    filter : Optional[AdvancedFilterRequest]
        Optional filter to apply before export
    include_classifications : bool
        Whether to include user classifications in export
    """

    format: ExportFormat = ExportFormat.JSON
    filter: Optional[AdvancedFilterRequest] = None
    include_classifications: bool = True

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "format": "csv",
                    "filter": {
                        "is_anomaly": True,
                        "date_range": {
                            "start_date": "2025-03-25T00:00:00.000Z",
                        }
                    },
                    "include_classifications": True
                }
            ]
        }
    )


class SimilarAnomaliesRequest(BaseModel):
    """
    Pydantic model for finding similar anomalies.

    Attributes
    ----------
    image_id : str
        Reference image ID to find similar anomalies for
    limit : int
        Maximum number of similar anomalies to return
    min_score : float
        Minimum similarity score threshold (0-1)
    """

    image_id: str
    limit: int = 10
    min_score: float = 0.5

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "image_id": "550e8400-e29b-41d4-a716-446655440000",
                    "limit": 5,
                    "min_score": 0.7
                }
            ]
        }
    )


class SimilarityResult(BaseModel):
    """
    Pydantic model for a single similarity result.

    Attributes
    ----------
    image : ImageResponse
        The similar image
    similarity_score : float
        Score indicating similarity to the reference image (0-1)
    """

    image: ImageResponse
    similarity_score: float

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "image": {
                        "id": "650e8400-e29b-41d4-a716-446655440000",
                        "filename": "similar.jpg",
                        "timestamp": "2025-04-25T12:34:56.789Z",
                        "reconstruction_error": 0.125,
                        "is_anomaly": True,
                        "anomaly_score": 2.3,
                        "path": "/uploads/650e8400-e29b-41d4-a716-446655440000.jpg",
                    },
                    "similarity_score": 0.92
                }
            ]
        }
    )


class SimilarAnomaliesResponse(BaseModel):
    """
    Pydantic model for similar anomalies response.

    Attributes
    ----------
    reference_image_id : str
        ID of the reference image
    similar_images : List[SimilarityResult]
        List of similar images with similarity scores
    """

    reference_image_id: str
    similar_images: List[SimilarityResult]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "reference_image_id": "550e8400-e29b-41d4-a716-446655440000",
                    "similar_images": [
                        {
                            "image": {
                                "id": "650e8400-e29b-41d4-a716-446655440000",
                                "filename": "similar1.jpg",
                                "timestamp": "2025-04-25T12:34:56.789Z",
                                "reconstruction_error": 0.125,
                                "is_anomaly": True,
                                "anomaly_score": 2.3,
                                "path": "/uploads/650e8400-e29b-41d4-a716-446655440000.jpg",
                            },
                            "similarity_score": 0.92
                        }
                    ]
                }
            ]
        }
    )
