from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import joblib
import datetime
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, Optional, AsyncGenerator, List
import logging
from tempfile import NamedTemporaryFile

from typer import Typer
import pandas as pd
import tempfile

# Import models and settings
from anomaly_reaper.infrastructure.database.connector import query_images
from anomaly_reaper.infrastructure.database.models import (
    get_db,
    create_tables,
    ImageRecord,
    Classification,
)
from anomaly_reaper.config import settings, logger
from anomaly_reaper.processing.anomaly_detection import process_image

# Import GCS utilities
from anomaly_reaper.schemas import (
    ClassificationResponse,
    HealthCheckResponse,
    ImageResponse,
    StatisticsResponse,
    ImageClassificationRequest,
    ImageClassificationResponse,
    SyncAnomalyDataResponse,
    # Add the new schemas we'll define
    AdvancedFilterRequest,
    AdvancedFilterResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    DashboardStatsResponse,
    # Add the new schemas for the additional endpoints
    ClassificationHistoryResponse,
    ExportRequest,
    ExportFormat,
    SimilarAnomaliesRequest,
    SimilarAnomaliesResponse,
    SimilarityResult,
    # Import the visualization schemas and utilities
    ImageVisualizationType,
    ImageVisualizationResponse,
    AnomalyVisualizationResponse,
    PCAProjectionRequest,
    PCAProjectionResponse,
    PaginatedImagesResponse,
)
from anomaly_reaper.utils.gcs_storage import (
    get_gcs_client,
    upload_image_to_gcs,
    download_image_from_gcs,
    is_gcs_path,
)
from anomaly_reaper.utils.visualization import (
    encode_image_to_base64,
    create_anomaly_heatmap,
    create_bounding_box_visualization,
    generate_pca_projection,
    calculate_anomaly_bbox,
)


os.makedirs(settings.uploads_dir, exist_ok=True)
os.makedirs(settings.models_dir, exist_ok=True)


# Load models on startup
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifecycle manager for the FastAPI application.

    This async context manager handles the startup and shutdown events
    for the FastAPI application. On startup, it loads the PCA model and
    scaler. On shutdown, it performs any necessary cleanup.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance

    Yields
    ------
    None
        Yields control back to FastAPI after startup tasks are complete

    Examples
    --------
    >>> app = FastAPI(lifespan=lifespan)
    """
    create_tables()

    # Load the ML models
    app.state.pca_model = None

    try:
        # Log GCS configuration
        if settings.use_cloud_storage:
            logger.info("Using Google Cloud Storage for image storage")
            logger.info(f"GCS bucket: {settings.gcs_bucket_name}")
            logger.info(f"GCS images prefix: {settings.gcs_images_prefix}")
        else:
            logger.info("Using local storage for images")

        model_dir_path = Path(settings.models_dir)

        model_dir_path.mkdir(parents=True, exist_ok=True)
        pca_model_path = model_dir_path / "pca_model.pkl"

        if not pca_model_path.exists() and settings.use_cloud_storage:
            # Attempt to download the model from GCS
            try:
                client = get_gcs_client()
                bucket = client.bucket(settings.gcs_bucket_name)
                blob = bucket.blob("anomaly_reaper/pca_model.pkl")

                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = temp_file.name

                # Download the file
                blob.download_to_filename(temp_path)
                logger.info(f"Downloaded PCA model to {temp_path}")

                # Move it to the models directory
                os.rename(temp_path, pca_model_path)
            except Exception as e:
                logger.error(f"Error downloading PCA model from GCS: {str(e)}")

        app.state.pca_model = joblib.load(str(pca_model_path))

        logger.info("Models and configuration loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
    except (IOError, OSError) as e:
        logger.error(f"I/O error when loading models: {e}")
    except (joblib.JoblibException, AttributeError) as e:
        logger.error(f"Error in model format or content: {e}")

    yield

    # Cleanup code (if needed)
    logger.info("Shutting down application")


# Create the FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="API for detecting anomalies in images using PCA",
    version="1.0.0",
    lifespan=lifespan,
    # Define tags with descriptions for API documentation
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check endpoint for monitoring system status",
        },
        {
            "name": "images",
            "description": "Operations with images including upload, retrieval, and classification",
        },
        {
            "name": "statistics",
            "description": "Statistical information about processed images and anomalies",
        },
        {
            "name": "visualizations",
            "description": "Generate visualizations for anomalies and data projections",
        },
        {
            "name": "export",
            "description": "Export data in various formats",
        },
        {
            "name": "classifications",
            "description": "Classify images as anomalous or normal",
        },
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint to verify the API is running.

    Returns
    -------
    HealthCheckResponse
        Status of the API
    """
    return HealthCheckResponse(status="ok", version="1.0.0")


# Use plural resource name consistently for collection 
@app.post("/images", tags=["images"])
async def create_image(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> ImageResponse:
    """
    Create a new image record by processing an uploaded image file.
    
    Parameters
    ----------
    file : UploadFile
        The image file to process

    Returns
    -------
    ImageResponse
        The processing results with status code 201
    """
    try:
        # Create a unique filename for the uploaded image
        filename = f"{uuid.uuid4()}_{file.filename}"

        # Read the file content
        contents = await file.read()

        # Store the image based on configuration
        if settings.use_cloud_storage:
            try:
                # Determine content type based on file extension
                content_type = "image/jpeg"  # Default
                if file.filename.lower().endswith(".png"):
                    content_type = "image/png"
                elif file.filename.lower().endswith(".gif"):
                    content_type = "image/gif"
                elif file.filename.lower().endswith((".fits", ".fit")):
                    content_type = "image/fits"

                # Upload to GCS and get the path
                filepath = upload_image_to_gcs(contents, filename, content_type)
                logger.info(f"Uploaded image to GCS: {filepath}")

                # For processing, we need a local copy temporarily
                temp_local_path = os.path.join(settings.uploads_dir, filename)
                with open(temp_local_path, "wb") as f:
                    f.write(contents)

                # Process the image
                result = process_image(temp_local_path, settings.models_dir)

                # Clean up the temporary file
                os.unlink(temp_local_path)
            except ValueError as e:
                # Handle missing GCP configuration by falling back to local storage
                logger.warning(
                    f"Cloud storage configuration issue: {str(e)}. Falling back to local storage."
                )

                # Save locally instead
                filepath = os.path.join(settings.uploads_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(contents)

                # Process the image
                result = process_image(filepath, settings.models_dir)
        else:
            # Save locally
            filepath = os.path.join(settings.uploads_dir, filename)
            with open(filepath, "wb") as f:
                f.write(contents)

            # Process the image
            result = process_image(filepath, settings.models_dir)

        # Create record
        record = ImageRecord(
            id=str(uuid.uuid4()),
            filename=filename,
            path=filepath,
            reconstruction_error=result["reconstruction_error"],
            is_anomaly=result["is_anomaly"],
            anomaly_score=result["anomaly_score"],
            processed_at=datetime.datetime.now(
                datetime.timezone.utc
            ),  # Use actual datetime object with timezone
        )

        # Save to database
        db.add(record)
        db.commit()
        db.refresh(record)

        # Return response
        return ImageResponse(
            id=record.id,
            filename=record.filename,
            timestamp=record.processed_at,
            reconstruction_error=record.reconstruction_error,
            is_anomaly=record.is_anomaly,
            anomaly_score=record.anomaly_score,
            path=record.path,
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Get list of processed images
@app.get("/images/", tags=["images"])
async def get_images(
    anomalies_only: bool = False, 
    page: int = 1,
    page_size: int = 9,
    sort_by: str = "processed_at",
    sort_order: str = "desc"
) -> PaginatedImagesResponse:
    """
    Get list of processed images from the database with pagination.

    Parameters
    ----------
    anomalies_only : bool
        If True, only return anomalous images
    page : int
        Page number (1-based)
    page_size : int
        Number of images per page
    sort_by : str
        Field to sort images by
    sort_order : str
        Sort order ("asc" or "desc")

    Returns
    -------
    PaginatedImagesResponse
        Paginated list of image records
    """
    try:
        results, total_count, total_pages = query_images(
            anomalies_only=anomalies_only,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Convert to response objects
        image_responses = [
            ImageResponse(
                id=result.id,
                filename=result.filename,
                timestamp=result.processed_at,
                reconstruction_error=result.reconstruction_error,
                is_anomaly=result.is_anomaly,
                anomaly_score=result.anomaly_score,
                path=result.path,
            )
            for result in results
        ]
        
        # Return paginated response
        return PaginatedImagesResponse(
            results=image_responses,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    except Exception as e:
        logger.error(f"Error retrieving images: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving images: {str(e)}"
        )


# Search images with filtering criteria.
@app.get("/images/search", tags=["images"])
async def search_images(
    is_anomaly: Optional[bool] = None,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    is_classified: Optional[bool] = None,
    user_classification: Optional[bool] = None,
    page: int = 1,
    page_size: int = 10,
    sort_by: str = "processed_at",
    sort_order: str = "desc",
    db: Session = Depends(get_db)
) -> PaginatedImagesResponse:
    """
    Search images with filtering criteria.

    Parameters
    ----------
    is_anomaly : Optional[bool]
        Filter by model-detected anomaly status
    start_date : Optional[datetime]
        Filter by start date (inclusive)
    end_date : Optional[datetime]
        Filter by end date (inclusive)
    min_score : Optional[float]
        Minimum anomaly score
    max_score : Optional[float]
        Maximum anomaly score
    is_classified : Optional[bool]
        Whether the image has been classified by a user
    user_classification : Optional[bool]
        Filter by user classification (True for anomaly, False for normal)
    page : int
        Page number (1-based)
    page_size : int
        Number of results per page
    sort_by : str
        Field to sort by
    sort_order : str
        Sort order ("asc" or "desc")

    Returns
    -------
    PaginatedImagesResponse
        Filtered and paginated image records
    """
    try:
        # Start with a base query
        query = db.query(ImageRecord)

        # Apply filters
        if is_anomaly is not None:
            query = query.filter(ImageRecord.is_anomaly == is_anomaly)

        # Date range filter
        if start_date:
            query = query.filter(ImageRecord.processed_at >= start_date)
        if end_date:
            query = query.filter(ImageRecord.processed_at <= end_date)

        # Anomaly score filter
        if min_score is not None:
            query = query.filter(ImageRecord.anomaly_score >= min_score)
        if max_score is not None:
            query = query.filter(ImageRecord.anomaly_score <= max_score)

        # Classification filters are more complex as they involve joins
        if is_classified is not None:
            if is_classified:
                # Images that have classifications
                query = query.join(
                    Classification, ImageRecord.id == Classification.image_id
                ).distinct()
            else:
                # Images that don't have classifications
                # This requires a subquery to find images without classifications
                classified_image_ids = (
                    db.query(Classification.image_id).distinct().subquery()
                )
                query = query.filter(~ImageRecord.id.in_(classified_image_ids))

        # Filter by specific user classification (True/False)
        if user_classification is not None and is_classified:
            query = (
                query.join(
                    Classification, ImageRecord.id == Classification.image_id
                )
                .filter(
                    Classification.user_classification == user_classification
                )
                .distinct()
            )

        # Count total matching records before pagination
        total_count = query.count()

        # Apply sorting
        sort_column = getattr(ImageRecord, sort_by, ImageRecord.processed_at)
        if sort_order and sort_order.lower() == "asc":
            query = query.order_by(sort_column.asc())
        else:
            query = query.order_by(sort_column.desc())

        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)

        # Execute query
        records = query.all()

        # Calculate total pages
        total_pages = (
            (total_count + page_size - 1) // page_size
            if total_count > 0
            else 1
        )

        # Convert to response objects
        results = [
            ImageResponse(
                id=record.id,
                filename=record.filename,
                timestamp=record.processed_at,
                reconstruction_error=record.reconstruction_error,
                is_anomaly=record.is_anomaly,
                anomaly_score=record.anomaly_score,
                path=record.path,
            )
            for record in records
        ]

        # Return paginated response
        return PaginatedImagesResponse(
            results=results,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    except Exception as e:
        logger.error(f"Error searching images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching images: {str(e)}")


# Get image details by ID
@app.get("/images/{image_id}", tags=["images"])
async def get_image_details(
    image_id: str, db: Session = Depends(get_db)
) -> ImageResponse:
    """
    Get details for a specific image.

    Parameters
    ----------
    image_id : str
        ID of the image to retrieve

    Returns
    -------
    ImageResponse
        Image details
    """
    try:
        # Query the database for the image
        image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()

        if not image:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )

        return ImageResponse(
            id=image.id,
            filename=image.filename,
            timestamp=image.processed_at,
            reconstruction_error=image.reconstruction_error,
            is_anomaly=image.is_anomaly,
            anomaly_score=image.anomaly_score,
            path=image.path,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image details: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving image details: {str(e)}"
        )


# Get image file
@app.get("/images/{image_id}/file", tags=["images"])
async def get_image_file(image_id: str) -> FileResponse:
    """
    Retrieve the image file for a specific image.

    Parameters
    ----------
    image_id : str
        ID of the image to retrieve

    Returns
    -------
    FileResponse
        The image file
    """
    image_data = get_image_by_id(image_id)

    if not image_data:
        raise HTTPException(
            status_code=404, detail=f"Image file with ID {image_id} not found"
        )

    # Create a temporary file to serve
    with NamedTemporaryFile(
        delete=False, suffix=".jpg"
    ) as temp_file:
        temp_file.write(image_data["image_data"])
        temp_file_path = temp_file.name

    return FileResponse(
        path=temp_file_path,
        headers={"Content-Disposition": f"inline; filename={image_id}.jpg"},
        media_type=image_data["content_type"],
    )


# Classify an image
@app.post("/images/{image_id}/classify", tags=["classifications"])
async def api_classify_image(
    image_id: str, request: ImageClassificationRequest
) -> ImageClassificationResponse:
    """
    Classify an image as anomalous or normal.

    Parameters
    ----------
    image_id : str
        ID of the image to classify
    request : ImageClassificationRequest
        Request containing is_anomaly and optional comment fields

    Returns
    -------
    ImageClassificationResponse
        Classification result
    """
    try:
        is_anomaly = request.is_anomaly
        comment = request.comment

        result = classify_image(image_id, is_anomaly, comment)

        if not result:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )

        return ImageClassificationResponse(
            id=image_id, user_classification=is_anomaly, comment=comment
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error classifying image: {str(e)}"
        )


# Get statistics about processed images
@app.get("/statistics", tags=["statistics"])
async def api_statistics() -> StatisticsResponse:
    """
    Get statistics about processed images.

    Returns
    -------
    StatisticsResponse
        Statistics about processed images
    """
    try:
        stats = get_statistics()
        return StatisticsResponse(**stats)
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving statistics: {str(e)}"
        )


# Get an image by ID
def get_image_by_id(image_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve an image file by its ID.

    Parameters
    ----------
    image_id : str
        ID of the image to retrieve

    Returns
    -------
    Optional[Dict]
        Dictionary containing image data and content type, or None if not found
    """
    try:
        # Query the database for the image record
        db = next(get_db())
        record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()

        if not record or not record.path:
            return None

        # Check if the path is a GCS path
        if is_gcs_path(record.path):
            if settings.use_cloud_storage:
                try:
                    # Download from GCS
                    image_data, content_type = download_image_from_gcs(record.path)
                    return {"image_data": image_data, "content_type": content_type}
                except Exception as e:
                    logger.error(f"Error downloading image from GCS: {str(e)}")
                    # Try to find a local copy with same filename
                    filename = os.path.basename(record.path)
                    local_path = os.path.join(settings.uploads_dir, filename)
                    if os.path.exists(local_path):
                        logger.info(f"Found local copy of GCS image at {local_path}")
                        with open(local_path, "rb") as f:
                            image_data = f.read()

                        # Determine content type based on file extension
                        content_type = "image/jpeg"  # Default
                        if local_path.lower().endswith(".png"):
                            content_type = "image/png"
                        elif local_path.lower().endswith(".gif"):
                            content_type = "image/gif"
                        elif local_path.lower().endswith((".fits", ".fit")):
                            content_type = "image/fits"

                        return {"image_data": image_data, "content_type": content_type}
                    return None
            else:
                # GCS is disabled but path is GCS - try to find a local copy
                logger.warning(
                    f"GCS path found ({record.path}) but cloud storage is disabled. Trying to find local copy."
                )
                filename = os.path.basename(record.path)
                local_path = os.path.join(settings.uploads_dir, filename)

                if os.path.exists(local_path):
                    logger.info(f"Found local copy of GCS image at {local_path}")
                    with open(local_path, "rb") as f:
                        image_data = f.read()

                    # Determine content type based on file extension
                    content_type = "image/jpeg"  # Default
                    if local_path.lower().endswith(".png"):
                        content_type = "image/png"
                    elif local_path.lower().endswith(".gif"):
                        content_type = "image/gif"
                    elif local_path.lower().endswith((".fits", ".fit")):
                        content_type = "image/fits"

                    return {"image_data": image_data, "content_type": content_type}

                # If we can't find a local copy, check the original data folder
                original_image_path = None
                for root, _, files in os.walk(settings.data_dir):
                    for file in files:
                        if file == filename:
                            original_image_path = os.path.join(root, file)
                            break
                    if original_image_path:
                        break

                if original_image_path and os.path.exists(original_image_path):
                    logger.info(f"Found original image at {original_image_path}")
                    with open(original_image_path, "rb") as f:
                        image_data = f.read()

                    # Determine content type based on file extension
                    content_type = "image/jpeg"  # Default
                    if original_image_path.lower().endswith(".png"):
                        content_type = "image/png"
                    elif original_image_path.lower().endswith(".gif"):
                        content_type = "image/gif"
                    elif original_image_path.lower().endswith((".fits", ".fit")):
                        content_type = "image/fits"

                    return {"image_data": image_data, "content_type": content_type}

                logger.error(
                    f"Cannot find local copy for GCS image with path {record.path}"
                )
                return None
        else:
            # Handle local file
            if not os.path.exists(record.path):
                logger.error(f"Image file not found at path: {record.path}")
                return None

            # Read the image file
            with open(record.path, "rb") as f:
                image_data = f.read()

            # Determine content type based on file extension
            content_type = "image/jpeg"  # Default
            if record.path.lower().endswith(".png"):
                content_type = "image/png"
            elif record.path.lower().endswith(".gif"):
                content_type = "image/gif"
            elif record.path.lower().endswith((".fits", ".fit")):
                content_type = "image/fits"

            return {"image_data": image_data, "content_type": content_type}
    except Exception as e:
        logger.error(f"Error retrieving image by ID: {str(e)}")
        return None


# Get statistics about processed images
def get_statistics() -> Dict[str, Any]:
    """
    Get statistics about processed images.

    Returns
    -------
    dict
        Statistics about processed images
    """
    try:
        db = next(get_db())
        total_count = db.query(func.count(ImageRecord.id)).scalar()
        anomaly_count = (
            db.query(func.count(ImageRecord.id))
            .filter(ImageRecord.is_anomaly == True)  # noqa: E712
            .scalar()
        )
        classified_count = db.query(func.count(Classification.id)).scalar()
        avg_score = db.query(func.avg(ImageRecord.anomaly_score)).scalar() or 0

        return {
            "total_images": total_count,
            "anomalies_detected": anomaly_count,
            "classified_images": classified_count,
            "average_anomaly_score": float(avg_score),
            "storage_type": "Google Cloud Storage"
            if settings.use_cloud_storage
            else "Local",
            "storage_location": settings.gcs_bucket_name
            if settings.use_cloud_storage
            else settings.uploads_dir,
        }
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise


# Classify an image and record the classification
def classify_image(
    image_id: str, is_anomaly: bool, comment: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Classify an image as anomalous or normal and store the classification.

    Parameters
    ----------
    image_id : str
        ID of the image to classify
    is_anomaly : bool
        Whether the image is an anomaly (True) or normal (False)
    comment : str, optional
        Optional comment about the classification

    Returns
    -------
    dict
        Classification result, or None if image not found
    """
    try:
        # Get database session
        db = next(get_db())

        # Find the image record
        record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()

        if not record:
            logger.error(f"Image with ID {image_id} not found")
            return None

        # Create a new classification record
        classification_id = str(uuid.uuid4())
        classification = Classification(
            id=classification_id,
            image_id=image_id,
            user_classification=is_anomaly,
            comment=comment,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )

        # Add to database
        db.add(classification)
        db.commit()

        # Return the result
        return {"id": image_id, "user_classification": is_anomaly, "comment": comment}
    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        return None


# Get visualizations for a specific image
@app.get("/images/{image_id}/visualization", tags=["visualizations"])
async def get_image_visualization(
    image_id: str, db: Session = Depends(get_db)
) -> AnomalyVisualizationResponse:
    """
    Get visualizations for a specific image including:
    - Original image
    - Heatmap visualization of anomaly regions
    - Bounding box visualization of detected anomalies

    Parameters
    ----------
    image_id : str
        ID of the image to visualize

    Returns
    -------
    AnomalyVisualizationResponse
        Multiple visualization types for the image
    """
    try:
        # Verify the image exists
        image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
        if not image:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )

        # Get the image data
        image_data = get_image_by_id(image_id)
        if not image_data:
            raise HTTPException(
                status_code=404, detail=f"Image file with ID {image_id} not found"
            )

        # Create a temporary file from the image data
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_data["image_data"])
            temp_file_path = temp_file.name

        try:
            # Load the image
            from PIL import Image

            original_image = Image.open(temp_file_path)

            # Calculate anomaly threshold as it was used in the model
            # We're using the standard threshold from the settings
            anomaly_threshold = float(settings.anomaly_threshold)
            error_value = float(image.reconstruction_error)
            is_anomaly = bool(image.is_anomaly)

            # 1. Original image - just encode to base64
            original_visualization = ImageVisualizationResponse(
                image_id=image_id,
                visualization_type=ImageVisualizationType.ORIGINAL,
                image_data=encode_image_to_base64(original_image),
            )

            # 2. Anomaly heatmap visualization
            heatmap_image = create_anomaly_heatmap(
                original_image=original_image,
                error_value=error_value,
                threshold=anomaly_threshold,
                colormap="plasma",
            )
            heatmap_visualization = ImageVisualizationResponse(
                image_id=image_id,
                visualization_type=ImageVisualizationType.ANOMALY_HEATMAP,
                image_data=encode_image_to_base64(heatmap_image),
            )

            # 3. Bounding box visualization
            # Calculate a bounding box based on the error value
            bbox = calculate_anomaly_bbox(
                image_shape=original_image.size[
                    ::-1
                ],  # PIL Image.size is (width, height)
                error_value=error_value,
                threshold=anomaly_threshold,
            )

            bbox_image = create_bounding_box_visualization(
                original_image=original_image,
                bbox=bbox,
                is_anomaly=is_anomaly,
                error_value=error_value,
                threshold=anomaly_threshold,
            )

            bbox_visualization = ImageVisualizationResponse(
                image_id=image_id,
                visualization_type=ImageVisualizationType.BOUNDING_BOX,
                image_data=encode_image_to_base64(bbox_image),
            )

            # Combine all visualizations
            visualizations = [
                original_visualization,
                heatmap_visualization,
                bbox_visualization,
            ]

            # Return the visualization response
            return AnomalyVisualizationResponse(
                image_id=image_id,
                visualizations=visualizations,
                reconstruction_error=error_value,
                is_anomaly=is_anomaly,
                anomaly_score=float(image.anomaly_score)
                if image.anomaly_score is not None
                else 0.0,
            )

        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image visualization: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating image visualization: {str(e)}"
        )


# Synchronize anomaly detection data from GCS/local storage
@app.post("/sync-anomaly-data", tags=["statistics"])
async def sync_anomaly_data(db: Session = Depends(get_db)) -> SyncAnomalyDataResponse:
    """
    Synchronize anomaly detection data from GCS bucket or local storage.

    This endpoint loads pre-calculated anomaly detection data

    It checks which images are already in the database and only imports the new ones.
    No request body is needed - the endpoint automatically detects the correct data source.

    Returns
    -------
    SyncAnomalyDataResponse
        Summary of synchronization results
    """
    try:
        # Determine the source path for embeddings_df.csv
        if settings.use_cloud_storage:
            # Use GCS path
            embeddings_df_path = (
                f"gs://{settings.gcs_bucket_name}/anomaly_reaper/embeddings_df.csv"
            )
            logger.info(f"Loading anomaly data from GCS: {embeddings_df_path}")
        else:
            # Use local path
            embeddings_df_path = os.path.join("data", "embeddings_df.csv")
            logger.info(f"Loading anomaly data from local path: {embeddings_df_path}")

        # Load the embeddings DataFrame
        if settings.use_cloud_storage and is_gcs_path(embeddings_df_path):
            # For GCS, we need to download the file first
            try:
                client = get_gcs_client()
                bucket = client.bucket(settings.gcs_bucket_name)
                blob = bucket.blob("anomaly_reaper/embeddings_df.csv")

                # Create a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv"
                ) as temp_file:
                    temp_path = temp_file.name

                # Download the file
                blob.download_to_filename(temp_path)
                logger.info(f"Downloaded embeddings dataframe to {temp_path}")

                # Read the CSV
                embeddings_df = pd.read_csv(temp_path)

                # Clean up
                os.unlink(temp_path)

            except Exception as e:
                logger.error(f"Error loading from GCS, falling back to local: {str(e)}")
                embeddings_df_path = os.path.join("data", "embeddings_df.csv")
                embeddings_df = pd.read_csv(embeddings_df_path)
        else:
            # Load directly from local path
            embeddings_df = pd.read_csv(embeddings_df_path)

        logger.info(f"Loaded data with {len(embeddings_df)} records")

        # Check if required columns exist
        required_columns = ["image_path", "reconstruction_error", "is_anomaly"]
        missing_columns = [
            col for col in required_columns if col not in embeddings_df.columns
        ]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Embeddings data missing required columns: {missing_columns}",
            )

        # Get existing filenames and paths from database for quick lookup
        existing_filenames = set()
        existing_paths = set()

        for record in db.query(ImageRecord.filename, ImageRecord.path).all():
            existing_filenames.add(record.filename)
            existing_paths.add(record.path)

        logger.info(f"Database contains {len(existing_filenames)} existing records")

        # Counters for statistics
        imported_count = 0
        anomaly_count = 0
        skipped_count = 0
        error_count = 0

        # Process each row in the DataFrame
        for idx, row in embeddings_df.iterrows():
            try:
                # Extract image path and normalize it
                image_path = row["image_path"]
                filename = os.path.basename(image_path)

                # Prepare paths for checking against database
                local_path = image_path
                gcs_path = None

                # Format GCS path for comparison with database records
                if settings.use_cloud_storage:
                    gcs_prefix = settings.gcs_images_prefix or ""
                    if not gcs_prefix.endswith("/"):
                        gcs_prefix += "/"
                    gcs_path = f"gs://{settings.gcs_bucket_name}/{gcs_prefix}{os.path.basename(image_path)}"

                # Check if image already exists in the database
                if (
                    filename in existing_filenames
                    or image_path in existing_paths
                    or (gcs_path and gcs_path in existing_paths)
                ):
                    skipped_count += 1
                    continue

                # Use the GCS path if in cloud storage mode
                final_path = gcs_path if settings.use_cloud_storage else local_path

                # Extract/compute values from the dataframe
                reconstruction_error = float(row["reconstruction_error"])
                is_anomaly = bool(row["is_anomaly"])

                # Get anomaly score if available, otherwise compute it
                if "anomaly_score" in row:
                    anomaly_score = float(row["anomaly_score"])
                else:
                    # Compute a simple anomaly score based on reconstruction error
                    # Higher reconstruction error means higher anomaly score
                    mean_error = embeddings_df["reconstruction_error"].mean()
                    if mean_error > 0:
                        anomaly_score = min(reconstruction_error / mean_error, 1.0)
                    else:
                        anomaly_score = 0.5 if is_anomaly else 0.0

                # Create a new record
                new_record = ImageRecord(
                    id=str(uuid.uuid4()),
                    filename=filename,
                    path=final_path,
                    reconstruction_error=reconstruction_error,
                    is_anomaly=is_anomaly,
                    anomaly_score=anomaly_score,
                    processed_at=datetime.datetime.now(datetime.timezone.utc),
                )

                # Add to database
                db.add(new_record)
                imported_count += 1

                # Count anomalies
                if is_anomaly:
                    anomaly_count += 1

                # Commit in batches to avoid memory issues
                if imported_count % 100 == 0:
                    db.commit()
                    logger.info(
                        f"Imported {imported_count} records so far ({anomaly_count} anomalies)"
                    )

            except Exception as e:
                logger.error(f"Error processing record {idx}: {str(e)}")
                error_count += 1

        # Final commit
        db.commit()

        # Return summary
        return SyncAnomalyDataResponse(
            message="Successfully synchronized anomaly data",
            source=embeddings_df_path,
            imported_count=imported_count,
            anomaly_count=anomaly_count,
            skipped_count=skipped_count,
            error_count=error_count,
            total_records=len(existing_filenames) + imported_count,
        )

    except Exception as e:
        logger.error(f"Error synchronizing anomaly data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error synchronizing anomaly data: {str(e)}"
        )


# Find images with similar anomaly patterns to a reference image.
@app.get("/images/{image_id}/similarities", tags=["images"])
async def get_similar_images(
    image_id: str,
    limit: int = 10,
    min_score: float = 0.5,
    db: Session = Depends(get_db)
) -> SimilarAnomaliesResponse:
    """
    Find images with similar anomaly patterns to a reference image.

    Parameters
    ----------
    image_id : str
        ID of the reference image
    limit : int
        Maximum number of similar images to return
    min_score : float
        Minimum similarity score threshold (0-1)

    Returns
    -------
    SimilarAnomaliesResponse
        List of similar images with similarity scores
    """
    try:
        # Get the reference image
        reference_image = (
            db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
        )
        if not reference_image:
            raise HTTPException(
                status_code=404, detail=f"Reference image with ID {image_id} not found"
            )

        # Find potential similar images
        # First, get images with similar anomaly characteristics (score and error)
        # Focus on images that were detected as anomalies
        query = (
            db.query(ImageRecord)
            .filter(ImageRecord.id != image_id)  # Exclude reference image
            .filter(ImageRecord.is_anomaly == True)  # noqa: E712
        )

        # Get all potential matches
        potential_matches = query.all()

        # Calculate similarity scores based on a combination of factors
        similarity_results = []
        for image in potential_matches:
            # Calculate score based on similarity in anomaly score and reconstruction error
            # Convert to numeric values to ensure we can do math with them
            ref_score = (
                float(reference_image.anomaly_score)
                if reference_image.anomaly_score is not None
                else 0.0
            )
            img_score = (
                float(image.anomaly_score) if image.anomaly_score is not None else 0.0
            )

            ref_error = (
                float(reference_image.reconstruction_error)
                if reference_image.reconstruction_error is not None
                else 0.0
            )
            img_error = (
                float(image.reconstruction_error)
                if image.reconstruction_error is not None
                else 0.0
            )

            # Score based on how close these values are (inversely proportional to difference)
            score_diff = abs(ref_score - img_score)
            error_diff = abs(ref_error - img_error)

            # Normalize differences relative to the reference values to get values between 0-1
            # (where 1 is most similar)
            max_score_diff = max(ref_score, 1.0)  # Avoid division by zero
            max_error_diff = max(ref_error, 1.0)  # Avoid division by zero

            normalized_score_similarity = 1.0 - min(score_diff / max_score_diff, 1.0)
            normalized_error_similarity = 1.0 - min(error_diff / max_error_diff, 1.0)

            # Combine the similarity scores (equal weight to both factors)
            combined_similarity = (
                normalized_score_similarity + normalized_error_similarity
            ) / 2

            # Only include if above minimum score threshold
            if combined_similarity >= min_score:
                similarity_results.append(
                    {"image": image, "similarity_score": combined_similarity}
                )

        # Sort by similarity score (highest first) and limit results
        similarity_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        limited_results = similarity_results[: limit]

        # Convert to response objects
        response_results = [
            SimilarityResult(
                image=ImageResponse(
                    id=result["image"].id,
                    filename=result["image"].filename,
                    timestamp=result["image"].processed_at,
                    reconstruction_error=result["image"].reconstruction_error,
                    is_anomaly=result["image"].is_anomaly,
                    anomaly_score=result["image"].anomaly_score,
                    path=result["image"].path,
                ),
                similarity_score=result["similarity_score"],
            )
            for result in limited_results
        ]

        # Return the results
        return SimilarAnomaliesResponse(
            reference_image_id=image_id, similar_images=response_results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar anomalies: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error finding similar anomalies: {str(e)}"
        )


# For visualization, use a more specific resource name
@app.post("/visualizations/pca", tags=["visualizations"])
async def create_pca_visualization(
    request: PCAProjectionRequest = None, db: Session = Depends(get_db)
) -> PCAProjectionResponse:
    """
    Create a PCA projection visualization of images.
    
    Parameters
    ----------
    request : PCAProjectionRequest, optional
        Optional request parameters for customizing the visualization

    Returns
    -------
    PCAProjectionResponse
        Visualization and related data
    """
    try:
        # Default request if none provided
        if request is None:
            request = PCAProjectionRequest()

        # Load the saved embeddings DataFrame or recompute it if not available
        embeddings_path = os.path.join(settings.data_dir, "embeddings_df.csv")

        if not os.path.exists(embeddings_path):
            # If we don't have the embeddings file locally, try to download it from GCS
            if settings.use_cloud_storage:
                try:
                    client = get_gcs_client()
                    bucket = client.bucket(settings.gcs_bucket_name)
                    blob = bucket.blob("anomaly_reaper/embeddings_df.csv")

                    # Create the data directory if it doesn't exist
                    os.makedirs(settings.data_dir, exist_ok=True)

                    # Download the file
                    blob.download_to_filename(embeddings_path)
                    logger.info("Downloaded embeddings dataframe from GCS")
                except Exception as e:
                    logger.error(f"Error downloading embeddings from GCS: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail="PCA projection visualization not available - embeddings data not found",
                    )
            else:
                raise HTTPException(
                    status_code=404,
                    detail="PCA projection visualization not available - embeddings data not found",
                )

        # Load the embeddings DataFrame
        import pandas as pd

        try:
            embeddings_df = pd.read_csv(embeddings_path)
            logger.info(f"Loaded embeddings DataFrame with {len(embeddings_df)} rows")
        except Exception as e:
            logger.error(f"Error loading embeddings DataFrame: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error loading embeddings data: {str(e)}"
            )

        # Apply any filters from the request
        filtered_df = embeddings_df

        # Apply filter if provided
        if request.filter:
            # Filter by anomaly status
            if request.filter.is_anomaly is not None:
                filtered_df = filtered_df[
                    filtered_df["is_anomaly"] == request.filter.is_anomaly
                ]

            # TODO: Add more filters based on embeddings_df columns and request.filter

        # Generate the PCA projection
        visualization, projection_data, threshold = generate_pca_projection(
            embeddings_df=filtered_df,
            highlight_anomalies=request.highlight_anomalies,
            use_interactive=request.use_interactive,
            include_paths=request.include_image_paths,
        )

        # Count the anomalies in the filtered data
        anomaly_count = (
            filtered_df["is_anomaly"].sum()
            if "is_anomaly" in filtered_df.columns
            else 0
        )

        # Return the visualization response
        return PCAProjectionResponse(
            visualization=visualization,
            projection_data=projection_data if request.include_image_paths else None,
            anomaly_threshold=threshold,
            total_points=len(filtered_df),
            anomaly_count=int(anomaly_count),
            is_interactive=request.use_interactive,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating PCA projection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating PCA projection: {str(e)}"
        )


typer_app = Typer(
    name=settings.app_name,
    help="Command-line interface for Anomaly Reaper",
    add_completion=False,
    no_args_is_help=True,
)


@typer_app.command()
def run(
    host: str = settings.host,
    port: int = settings.port,
    log_level: str = settings.log_level,
    reload: bool = False,
):
    """
    Run the Anomaly Reaper API server.

    Parameters
    ----------
    host : str
        Host to bind the server to
    port : int
        Port to bind the server to
    log_level : str
        Logging level (debug, info, warning, error, critical)
    reload : bool
        Enable auto-reload for development
    """
    import uvicorn

    # Set the log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start the server
    logger.info(f"Starting {settings.app_name} API server at http://{host}:{port}")
    uvicorn.run(
        "anomaly_reaper.interfaces.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    # Run the CLI app
    typer_app()
