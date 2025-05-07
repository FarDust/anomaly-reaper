from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import joblib
import datetime
import pandas as pd
import tempfile
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, Optional, AsyncGenerator
from tempfile import NamedTemporaryFile
import logging

from typer import Typer

# Import models and settings
from anomaly_reaper.constants import NOT_PROCESSABLE_RECONSTRUCTION_THRESHOLD
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
    HealthCheckResponse,
    ImageResponse,
    StatisticsResponse,
    ImageClassificationRequest,
    ImageClassificationResponse,
    SyncAnomalyDataResponse,
    # Add the new schemas we'll define
    AdvancedFilterResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    DashboardStatsResponse,
    # Add the new schemas for the additional endpoints
    ClassificationHistoryResponse,
    ExportFormat,
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

from anomaly_reaper.infrastructure.vector_database.images_store import ImagesVectorStore
from anomaly_reaper.infrastructure.vector_database.image_embeddings import VertexAIMultimodalImageEmbeddings


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
        local_filepath = os.path.join(settings.uploads_dir, filename)

        # Read the file content
        contents = await file.read()

        # Always save a local copy regardless of storage settings
        # This ensures there's always a local backup even when using GCS
        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
        with open(local_filepath, "wb") as f:
            f.write(contents)

        logger.info(f"Saved local copy at {local_filepath}")

        # Determine the path to store in the database
        filepath = local_filepath

        # If cloud storage is enabled, also upload to GCS and update the path
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
                elif file.filename.lower().endswith(".webp"):
                    content_type = "image/webp"

                # Upload to GCS and get the path
                gcs_path = upload_image_to_gcs(contents, filename, content_type)
                logger.info(f"Uploaded image to GCS: {gcs_path}")

                # Update the filepath to the GCS path
                filepath = gcs_path
            except Exception as e:
                # Handle missing GCP configuration by keeping the local path
                logger.warning(
                    f"Could not upload to cloud storage: {str(e)}. Using local storage only."
                )
                # Keep using the local path

        # Process the image - always use the local path for processing
        result = process_image(local_filepath, settings.models_dir)

        if result["reconstruction_error"] >= NOT_PROCESSABLE_RECONSTRUCTION_THRESHOLD:
            logger.info(
                f"Reconstruction error {result['reconstruction_error']} exceeds threshold {NOT_PROCESSABLE_RECONSTRUCTION_THRESHOLD}"
            )
            logger.warning(f"Image {filename} is not processable")
            unprocessed_record = ImageResponse(
                id="",
                filename=filename,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                reconstruction_error=result["reconstruction_error"],
                is_anomaly=result["is_anomaly"],
                anomaly_score=result["anomaly_score"],
                path=filepath,
            )
            return JSONResponse(
                status_code=422,
                content=unprocessed_record.model_dump(mode="json", fallback=str),
            )
        else:
            logger.info(
                f"Image {filename} processed successfully with reconstruction error {result['reconstruction_error']}"
            )
            logger.info(
                f"The image is classified was accepted as {'anomaly' if result['is_anomaly'] else 'normal'}"
            )

            image_id = str(uuid.uuid4())

            logger.info(f"Storing image {image_id} to the database")

            # Create record with the appropriate filepath (GCS or local)
            record = ImageRecord(
                id=image_id,
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
    sort_order: str = "desc",
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
            sort_order=sort_order,
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
    db: Session = Depends(get_db),
) -> AdvancedFilterResponse:
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
    AdvancedFilterResponse
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
                query.join(Classification, ImageRecord.id == Classification.image_id)
                .filter(Classification.user_classification == user_classification)
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
            (total_count + page_size - 1) // page_size if total_count > 0 else 1
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

        # Return advanced filter response
        return AdvancedFilterResponse(
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
async def get_image_file(image_id: str, db: Session = Depends(get_db)) -> FileResponse:
    """
    Retrieve the image file for a specific image.

    Parameters
    ----------
    image_id : str
        ID of the image to retrieve
    db : Session
        Database session

    Returns
    -------
    FileResponse
        The image file
    """
    try:
        logger.info(f"Attempting to retrieve image file for ID: {image_id}")

        # First, check if the image record exists in the database
        record = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()

        if not record:
            logger.error(f"Image record with ID {image_id} not found in database")
            raise HTTPException(
                status_code=404, detail=f"Image record with ID {image_id} not found"
            )

        logger.info(f"Found image record: path={record.path}")

        # Get the file path from the record
        file_path = record.path

        # Check if the file exists
        if is_gcs_path(file_path):
            # Handle GCS path
            try:
                image_data = get_image_by_id(image_id)
                if not image_data:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Image file with ID {image_id} not found in GCS",
                    )

                # Create a temporary file to serve
                with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(image_data["image_data"])
                    temp_file_path = temp_file.name

                return FileResponse(
                    path=temp_file_path,
                    filename=os.path.basename(file_path),
                    media_type=image_data["content_type"],
                    background=None,  # Let FastAPI handle cleanup
                )
            except Exception as e:
                logger.error(f"Error retrieving file from GCS: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Error retrieving file from GCS: {str(e)}"
                )
        else:
            # Handle local path
            if not os.path.exists(file_path):
                # Try alternative locations
                file_name = os.path.basename(file_path)
                alternative_path = os.path.join(settings.uploads_dir, file_name)

                if os.path.exists(alternative_path):
                    file_path = alternative_path
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Image file with ID {image_id} not found at {file_path}",
                    )

            # Determine content type
            content_type = get_content_type_from_filename(file_path)

            # Return the file
            return FileResponse(
                path=file_path,
                filename=os.path.basename(file_path),
                media_type=content_type,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error serving image file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error serving image file: {str(e)}"
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


# Batch classify multiple images
@app.patch("/images/classifications", tags=["classifications"])
async def batch_update_classifications(
    request: BatchClassificationRequest, db: Session = Depends(get_db)
) -> BatchClassificationResponse:
    """
    Update classifications for multiple images at once.

    Parameters
    ----------
    request : BatchClassificationRequest
        The batch classification request containing image IDs, classification, and optional comment

    Returns
    -------
    BatchClassificationResponse
        Summary of batch classification results
    """
    try:
        image_ids = request.image_ids
        is_anomaly = request.is_anomaly
        comment = request.comment

        # Validate that image_ids is not empty
        if not image_ids:
            raise HTTPException(
                status_code=422, detail="Image IDs list cannot be empty"
            )

        # Track results
        successful_count = 0
        failed_ids = []

        # Process each image ID
        for image_id in image_ids:
            try:
                # Check if the image exists
                image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
                if not image:
                    logger.warning(f"Image with ID {image_id} not found")
                    failed_ids.append(image_id)
                    continue

                # Create a classification record
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
                successful_count += 1

            except Exception as e:
                logger.error(f"Error classifying image {image_id}: {str(e)}")
                failed_ids.append(image_id)

        # Commit all successful classifications
        db.commit()

        # Return the summary
        return BatchClassificationResponse(
            total=len(image_ids),
            successful=successful_count,
            failed=len(failed_ids),
            failed_ids=failed_ids,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch classification: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in batch classification: {str(e)}"
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


# Get dashboard statistics
@app.get("/statistics/dashboard", tags=["statistics"])
async def get_dashboard_statistics(
    db: Session = Depends(get_db),
) -> DashboardStatsResponse:
    """
    Get comprehensive statistics for dashboard visualization.

    Returns
    -------
    DashboardStatsResponse
        Detailed dashboard statistics
    """
    try:
        # Get total number of images
        total_images = db.query(func.count(ImageRecord.id)).scalar()

        # Get number of anomalies detected by the model
        total_anomalies = (
            db.query(func.count(ImageRecord.id))
            .filter(ImageRecord.is_anomaly == True)  # noqa: E712
            .scalar()
        )

        # Get number of user-confirmed anomalies
        # These are images marked as anomalies by the model that users also classified as anomalies
        user_confirmed_anomalies = (
            db.query(func.count(ImageRecord.id))
            .join(Classification, ImageRecord.id == Classification.image_id)
            .filter(ImageRecord.is_anomaly == True)  # noqa: E712
            .filter(Classification.user_classification == True)  # noqa: E712
            .distinct()
            .scalar()
        )

        # Get number of unclassified anomalies
        # These are images marked as anomalies by the model that have not been classified by users yet
        # First find all classified image IDs
        classified_image_ids = db.query(Classification.image_id).distinct().subquery()

        # Then count anomalous images that aren't in the classified set
        unclassified_anomalies = (
            db.query(func.count(ImageRecord.id))
            .filter(ImageRecord.is_anomaly == True)  # noqa: E712
            .filter(~ImageRecord.id.in_(classified_image_ids))
            .scalar()
        )

        # Get false positives
        # These are images marked as anomalies by the model but classified as normal by users
        false_positives = (
            db.query(func.count(ImageRecord.id))
            .join(Classification, ImageRecord.id == Classification.image_id)
            .filter(ImageRecord.is_anomaly == True)  # noqa: E712
            .filter(Classification.user_classification == False)  # noqa: E712
            .distinct()
            .scalar()
        )

        # Get false negatives
        # These are images marked as normal by the model but classified as anomalies by users
        false_negatives = (
            db.query(func.count(ImageRecord.id))
            .join(Classification, ImageRecord.id == Classification.image_id)
            .filter(ImageRecord.is_anomaly == False)  # noqa: E712
            .filter(Classification.user_classification == True)  # noqa: E712
            .distinct()
            .scalar()
        )

        # Get recent activity (10 most recent classifications)
        recent_activity_records = (
            db.query(Classification)
            .order_by(Classification.timestamp.desc())
            .limit(10)
            .all()
        )

        # Format the activity for the response
        recent_activity = []
        for record in recent_activity_records:
            # Get the image details
            image = (
                db.query(ImageRecord).filter(ImageRecord.id == record.image_id).first()
            )

            activity_item = {
                "timestamp": record.timestamp,
                "image_id": record.image_id,
                "action": "classification",
                "is_anomaly": record.user_classification,
                "comment": record.comment,
                "model_prediction": image.is_anomaly if image else None,
                "anomaly_score": float(image.anomaly_score)
                if image and image.anomaly_score is not None
                else None,
            }
            recent_activity.append(activity_item)

        # Return the dashboard statistics
        return DashboardStatsResponse(
            total_images=total_images,
            total_anomalies=total_anomalies,
            user_confirmed_anomalies=user_confirmed_anomalies,
            unclassified_anomalies=unclassified_anomalies,
            false_positives=false_positives,
            false_negatives=false_negatives,
            recent_activity=recent_activity,
        )
    except Exception as e:
        logger.error(f"Error retrieving dashboard statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving dashboard statistics: {str(e)}"
        )


# Get image by ID
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
            logger.error(f"No record found for image ID {image_id} or path is empty")
            return None

        logger.info(f"Found image record: id={image_id}, path={record.path}")

        # Extract the filename from the path
        filename = os.path.basename(record.path)

        # Check if the path is a GCS path
        if is_gcs_path(record.path):
            if settings.use_cloud_storage:
                try:
                    # Download from GCS
                    image_data, content_type = download_image_from_gcs(record.path)
                    return {"image_data": image_data, "content_type": content_type}
                except Exception as e:
                    logger.error(f"Error downloading image from GCS: {str(e)}")
                    # Continue to fallback mechanisms below
            else:
                # GCS is disabled but path is GCS - try to find a local copy
                logger.warning(
                    f"GCS path found ({record.path}) but cloud storage is disabled. Trying to find local copy."
                )

            # Try local fallbacks:

            # 1. First check in uploads directory
            local_path = os.path.join(settings.uploads_dir, filename)
            if os.path.exists(local_path):
                logger.info(f"Found local copy of GCS image at {local_path}")
                with open(local_path, "rb") as f:
                    image_data = f.read()

                # Determine content type based on file extension
                content_type = get_content_type_from_filename(local_path)
                return {"image_data": image_data, "content_type": content_type}

            # 2. Check in data directory for astronomical images (FITS files, etc.)
            potential_data_paths = []

            # Specifically check the data/s0027/cam1-ccd1 directory for TESS images
            tess_data_dir = os.path.join(settings.data_dir, "s0027", "cam1-ccd1")
            if os.path.exists(tess_data_dir):
                for root, _, files in os.walk(tess_data_dir):
                    for file in files:
                        if (
                            filename in file
                        ):  # If filename is part of any file in the TESS data directory
                            potential_data_paths.append(os.path.join(root, file))

            # 3. Search all of data directory recursively for the exact filename
            for root, _, files in os.walk(settings.data_dir):
                if filename in files:
                    found_path = os.path.join(root, filename)
                    logger.info(f"Found image in data directory: {found_path}")
                    potential_data_paths.append(found_path)

            # Use the first found path
            if potential_data_paths:
                found_path = potential_data_paths[0]
                logger.info(f"Using found image at: {found_path}")
                with open(found_path, "rb") as f:
                    image_data = f.read()

                content_type = get_content_type_from_filename(found_path)
                return {"image_data": image_data, "content_type": content_type}

            # 4. If still not found, try to download from the GCS URL directly
            # even though GCS integration is disabled
            try:
                logger.info(
                    f"Attempting to download from GCS URL directly: {record.path}"
                )
                # Extract bucket and blob path from gs:// URL
                gcs_path = record.path.replace("gs://", "")
                parts = gcs_path.split("/", 1)
                if len(parts) < 2:
                    logger.error(f"Invalid GCS path format: {record.path}")
                    return None

                bucket_name = parts[0]
                blob_path = parts[1]

                # Create a public HTTP URL
                http_url = f"https://storage.googleapis.com/{bucket_name}/{blob_path}"

                # Download using requests
                import requests

                logger.info(f"Attempting to download from public URL: {http_url}")
                response = requests.get(http_url, timeout=10)

                if response.status_code == 200:
                    # Save a local copy for future use
                    local_save_path = os.path.join(settings.uploads_dir, filename)
                    os.makedirs(os.path.dirname(local_save_path), exist_ok=True)

                    with open(local_save_path, "wb") as f:
                        f.write(response.content)

                    logger.info(
                        f"Successfully downloaded and saved image to {local_save_path}"
                    )
                    return {
                        "image_data": response.content,
                        "content_type": response.headers.get(
                            "Content-Type", get_content_type_from_filename(filename)
                        ),
                    }
                else:
                    logger.error(
                        f"Failed to download from public URL. Status code: {response.status_code}"
                    )
            except Exception as e:
                logger.error(f"Error downloading from public URL: {str(e)}")

            logger.error(
                f"Cannot find local copy for GCS image with path {record.path}"
            )
            return None
        else:
            # Handle local file path
            # If the path doesn't exist as is, try to find it
            if not os.path.exists(record.path):
                logger.warning(
                    f"Image file not found at path: {record.path}. Trying alternative locations."
                )

                # Try finding the file in uploads directory
                upload_path = os.path.join(settings.uploads_dir, filename)
                if os.path.exists(upload_path):
                    logger.info(f"Found image in uploads directory: {upload_path}")
                    with open(upload_path, "rb") as f:
                        image_data = f.read()
                    content_type = get_content_type_from_filename(upload_path)
                    return {"image_data": image_data, "content_type": content_type}

                # Try recursively searching the data directory for the file
                for root, _, files in os.walk(settings.data_dir):
                    if filename in files:
                        found_path = os.path.join(root, filename)
                        logger.info(f"Found image in data directory: {found_path}")
                        with open(found_path, "rb") as f:
                            image_data = f.read()
                        content_type = get_content_type_from_filename(found_path)
                        return {"image_data": image_data, "content_type": content_type}

                logger.error(f"Image file not found in any location: {filename}")
                return None

            # Read the image file from the original path
            with open(record.path, "rb") as f:
                image_data = f.read()

            # Determine content type based on file extension
            content_type = get_content_type_from_filename(record.path)
            return {"image_data": image_data, "content_type": content_type}
    except Exception as e:
        logger.error(f"Error retrieving image by ID: {str(e)}")
        return None


# Helper function to determine content type from filename
def get_content_type_from_filename(filepath: str) -> str:
    """
    Determine content type based on file extension.

    Parameters
    ----------
    filepath : str
        Path to the file

    Returns
    -------
    str
        MIME content type for the file
    """
    content_type = "application/octet-stream"  # Default
    if filepath.lower().endswith(".jpg") or filepath.lower().endswith(".jpeg"):
        content_type = "image/jpeg"
    elif filepath.lower().endswith(".png"):
        content_type = "image/png"
    elif filepath.lower().endswith(".gif"):
        content_type = "image/gif"
    elif filepath.lower().endswith((".fits", ".fit")):
        content_type = "image/fits"
    elif filepath.lower().endswith(".webp"):
        content_type = "image/webp"
    return content_type


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
    db: Session = Depends(get_db),
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

        # Get the image data for the reference image
        reference_image_data = get_image_by_id(image_id)
        if not reference_image_data:
            raise HTTPException(
                status_code=404, detail=f"Image file with ID {image_id} not found"
            )
            
        
        # Create a temporary file from the reference image data for embedding
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(reference_image_data["image_data"])
            reference_temp_path = temp_file.name
        
        try:
            # Initialize the vertex AI embedding model
            embedding_model = VertexAIMultimodalImageEmbeddings()
            
            # Get all anomalous images (excluding the reference image)
            query = (
                db.query(ImageRecord)
                .filter(ImageRecord.id != image_id)  # Exclude reference image
                .filter(ImageRecord.is_anomaly == True)  # noqa: E712
            )
            
            # Get all potential matches
            potential_matches = query.all()
            
            # Create list of image paths and metadata for vector store
            image_paths = []
            metadatas = []
            
            for image in potential_matches:
                image_data = get_image_by_id(image.id)
                if image_data:
                    # Create a temporary file for this image
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_temp_file:
                        img_temp_file.write(image_data["image_data"])
                        image_paths.append(img_temp_file.name)
                        
                        # Store metadata with each image
                        metadatas.append({
                            "id": image.id,
                            "filename": image.filename,
                            "timestamp": image.processed_at,
                            "reconstruction_error": image.reconstruction_error,
                            "is_anomaly": image.is_anomaly,
                            "anomaly_score": image.anomaly_score,
                            "path": image.path
                        })
            
            # If we have potential matches, perform vector similarity search
            if image_paths:
                # Create vector store from the images
                vector_store = ImagesVectorStore.from_texts(
                    texts=image_paths,
                    embedding=embedding_model,
                    metadatas=metadatas
                )
                
                # Perform similarity search using vector embeddings
                search_results = vector_store.similarity_search_by_vector_with_score(
                    embedding=embedding_model.embed_query(reference_temp_path),
                    k=limit
                )
                
                # Convert results to response format
                response_results = []
                for doc, score in search_results:
                    if score >= min_score:
                        metadata = doc.metadata
                        response_results.append(
                            SimilarityResult(
                                image=ImageResponse(
                                    id=metadata["id"],
                                    filename=metadata["filename"],
                                    timestamp=metadata["timestamp"],
                                    reconstruction_error=metadata["reconstruction_error"],
                                    is_anomaly=metadata["is_anomaly"],
                                    anomaly_score=metadata["anomaly_score"],
                                    path=metadata["path"],
                                ),
                                similarity_score=score,
                            )
                        )
            else:
                # No potential matches found
                response_results = []
                
            # Clean up temporary files
            try:
                os.unlink(reference_temp_path)
                for path in image_paths:
                    if os.path.exists(path):
                        os.unlink(path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")
                
            # Return the results
            return SimilarAnomaliesResponse(
                reference_image_id=image_id, similar_images=response_results
            )
            
        finally:
            # Ensure cleanup of temporary files
            try:
                if os.path.exists(reference_temp_path):
                    os.unlink(reference_temp_path)
            except Exception as e:
                logger.warning(f"Error cleaning up reference image temporary file: {str(e)}")
            
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


# Export image data in CSV or JSON format
@app.get("/images/export", tags=["export"])
async def export_images(
    format: ExportFormat = ExportFormat.JSON,
    is_anomaly: Optional[bool] = None,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    is_classified: Optional[bool] = None,
    user_classification: Optional[bool] = None,
    include_classifications: bool = True,
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Export image data in CSV or JSON format with optional filtering.

    Parameters
    ----------
    format : ExportFormat
        Format for export (JSON or CSV)
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
    include_classifications : bool
        Whether to include classification details in the export
    sort_by : Optional[str]
        Field to sort by
    sort_order : str
        Sort order ("asc" or "desc")

    Returns
    -------
    FileResponse
        File containing the exported data
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

        # Classification filters
        if is_classified is not None:
            if is_classified:
                # Images that have classifications
                query = query.join(
                    Classification, ImageRecord.id == Classification.image_id
                ).distinct()
            else:
                # Images that don't have classifications
                classified_image_ids = (
                    db.query(Classification.image_id).distinct().subquery()
                )
                query = query.filter(~ImageRecord.id.in_(classified_image_ids))

        # Filter by specific user classification (True/False)
        if user_classification is not None and is_classified:
            query = (
                query.join(Classification, ImageRecord.id == Classification.image_id)
                .filter(Classification.user_classification == user_classification)
                .distinct()
            )

        # Apply sorting if specified
        if sort_by:
            sort_column = getattr(ImageRecord, sort_by, ImageRecord.processed_at)
            if sort_order.lower() == "asc":
                query = query.order_by(sort_column.asc())
            else:
                query = query.order_by(sort_column.desc())
        else:
            # Default sort by processed_at
            if sort_order.lower() == "asc":
                query = query.order_by(ImageRecord.processed_at.asc())
            else:
                query = query.order_by(ImageRecord.processed_at.desc())

        # Execute query
        records = query.all()

        # Prepare data for export
        export_data = []
        for record in records:
            image_data = {
                "id": record.id,
                "filename": record.filename,
                "timestamp": record.processed_at.isoformat(),
                "reconstruction_error": float(record.reconstruction_error)
                if record.reconstruction_error is not None
                else None,
                "is_anomaly": bool(record.is_anomaly),
                "anomaly_score": float(record.anomaly_score)
                if record.anomaly_score is not None
                else None,
                "path": record.path,
            }

            # Include classifications if requested
            if include_classifications:
                # Get the most recent classification for this image
                classification = (
                    db.query(Classification)
                    .filter(Classification.image_id == record.id)
                    .order_by(Classification.timestamp.desc())
                    .first()
                )

                if classification:
                    image_data["user_classified"] = True
                    image_data["user_classification"] = bool(
                        classification.user_classification
                    )
                    image_data["classification_comment"] = classification.comment
                    image_data["classification_timestamp"] = (
                        classification.timestamp.isoformat()
                    )
                else:
                    image_data["user_classified"] = False
                    image_data["user_classification"] = None
                    image_data["classification_comment"] = None
                    image_data["classification_timestamp"] = None

            export_data.append(image_data)

        # Create a file in the requested format
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == ExportFormat.CSV:
            import csv

            # Create a temporary CSV file
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".csv", newline=""
            ) as temp_file:
                temp_file_path = temp_file.name

                # Determine field names - use first record as template
                if export_data:
                    fieldnames = export_data[0].keys()
                    writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(export_data)
                else:
                    # No data, just create empty file with headers
                    fieldnames = [
                        "id",
                        "filename",
                        "timestamp",
                        "reconstruction_error",
                        "is_anomaly",
                        "anomaly_score",
                        "path",
                    ]
                    if include_classifications:
                        fieldnames.extend(
                            [
                                "user_classified",
                                "user_classification",
                                "classification_comment",
                                "classification_timestamp",
                            ]
                        )
                    writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
                    writer.writeheader()

            filename = f"anomaly_export_{current_timestamp}.csv"
            media_type = "text/csv"

        else:  # JSON format
            import json

            # Create a temporary JSON file
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as temp_file:
                temp_file_path = temp_file.name
                json.dump(export_data, temp_file, indent=2, default=str)

            filename = f"anomaly_export_{current_timestamp}.json"
            media_type = "application/json"

        # Return the file
        return FileResponse(
            path=temp_file_path,
            filename=filename,
            media_type=media_type,
            background=None,  # Let FastAPI handle cleanup
        )

    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


# Get classification history for a specific image
@app.get("/images/{image_id}/classifications", tags=["classifications"])
async def get_classification_history(
    image_id: str, db: Session = Depends(get_db)
) -> ClassificationHistoryResponse:
    """
    Get classification history for a specific image.

    This endpoint returns all user classifications made for a specific image,
    ordered by timestamp (most recent first).

    Parameters
    ----------
    image_id : str
        The ID of the image to get classification history for

    Returns
    -------
    ClassificationHistoryResponse
        Classification history for the image
    """
    try:
        # Verify the image exists
        image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
        if not image:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )

        # Get all classifications for this image, ordered by timestamp (newest first)
        classifications = (
            db.query(Classification)
            .filter(Classification.image_id == image_id)
            .order_by(Classification.timestamp.desc())
            .all()
        )

        # Convert to response objects
        classification_responses = [
            {
                "id": classification.id,
                "timestamp": classification.timestamp.isoformat()
                if hasattr(classification.timestamp, "isoformat")
                else classification.timestamp,
                "user_classification": classification.user_classification,
                "comment": classification.comment,
            }
            for classification in classifications
        ]

        # Return the classification history
        return ClassificationHistoryResponse(
            image_id=image_id, classifications=classification_responses
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving classification history: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving classification history: {str(e)}"
        )


def image_to_dict(image: ImageRecord) -> dict:
    """Convert an ImageRecord model instance to a dictionary.

    Parameters
    ----------
    image : ImageRecord
        The image record to convert

    Returns
    -------
    dict
        Dictionary representation of the image
    """
    return {
        "id": image.id,
        "filename": image.filename,
        "anomaly_score": image.anomaly_score,
        "is_anomaly": image.is_anomaly,
        "reconstruction_error": image.reconstruction_error,
        "timestamp": image.processed_at.isoformat() if image.processed_at else None,
    }


typer_app = Typer(
    name=settings.app_name,
    help="Command-line interface for Anomaly Reaper",
    add_completion=False,
    no_args_is_help=False,
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
