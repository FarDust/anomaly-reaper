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
    DateRangeFilterRequest,
    AnomalyScoreFilterRequest,
    ClassificationFilterRequest,
    DashboardStatsResponse,
    # Add the new schemas for the additional endpoints
    ClassificationHistoryResponse,
    ExportRequest,
    ExportFormat,
    SimilarAnomaliesRequest,
    SimilarAnomaliesResponse,
    SimilarityResult,
)
from anomaly_reaper.utils.gcs_storage import (
    get_gcs_client,
    upload_image_to_gcs,
    download_image_from_gcs,
    is_gcs_path,
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
@app.get("/health")
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint to verify the API is running.

    Returns
    -------
    HealthCheckResponse
        Status of the API
    """
    return HealthCheckResponse(status="ok", version="1.0.0")


# Process a single image
@app.post("/process")
async def process_single_image(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> ImageResponse:
    """
    Process a single image for anomaly detection.

    Parameters
    ----------
    file : UploadFile
        The image file to process

    Returns
    -------
    ImageResponse
        The processing results
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
            processed_at=datetime.datetime.now().isoformat(),
            user_classified=False,
            user_classification=None,
            user_comment=None,
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
@app.get("/images/")
async def get_images(anomalies_only: bool = False) -> List[ImageResponse]:
    """
    Get list of processed images from the database.

    Parameters
    ----------
    anomalies_only : bool
        If True, only return anomalous images

    Returns
    -------
    List[ImageResponse]
        List of image records
    """
    try:
        results = query_images(anomalies_only=anomalies_only)
        return [
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
    except Exception as e:
        logger.error(f"Error retrieving images: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving images: {str(e)}"
        )


# Get image details by ID
@app.get("/images/{image_id}")
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
        results = query_images()
        image = next((img for img in results if img.id == image_id), None)

        if not image:
            raise HTTPException(
                status_code=404, detail=f"Image with ID {image_id} not found"
            )

        return image
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image details: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving image details: {str(e)}"
        )


# Get image file
@app.get("/images/{image_id}/file")
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
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_data["image_data"])
        temp_file_path = temp_file.name

    return FileResponse(
        path=temp_file_path,
        headers={"Content-Disposition": f"inline; filename={image_id}.jpg"},
        media_type=image_data["content_type"],
    )


# Classify an image
@app.post("/images/{image_id}/classify")
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
@app.get("/statistics")
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
        if settings.use_cloud_storage and is_gcs_path(record.path):
            try:
                # Download from GCS
                image_data, content_type = download_image_from_gcs(record.path)
                return {"image_data": image_data, "content_type": content_type}
            except Exception as e:
                logger.error(f"Error downloading image from GCS: {str(e)}")
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
            elif record.path.lower().endswith(".fits") or record.path.lower().endswith(
                ".fit"
            ):
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


# Synchronize anomaly detection data from GCS/local storage
@app.post("/sync-anomaly-data")
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


# Advanced filtering endpoint
@app.post("/images/filter")
async def filter_images(
    request: AdvancedFilterRequest, db: Session = Depends(get_db)
) -> AdvancedFilterResponse:
    """
    Advanced filtering of images with multiple criteria.
    
    This endpoint allows filtering images by date range, anomaly score, 
    classification status, and other criteria, with pagination and sorting.

    Parameters
    ----------
    request : AdvancedFilterRequest
        The filtering criteria
        
    Returns
    -------
    AdvancedFilterResponse
        Filtered and paginated image records
    """
    try:
        # Start with a base query
        query = db.query(ImageRecord)
        
        # Apply filters
        if request.is_anomaly is not None:
            query = query.filter(ImageRecord.is_anomaly == request.is_anomaly)
            
        # Date range filter
        if request.date_range:
            if request.date_range.start_date:
                query = query.filter(
                    ImageRecord.processed_at >= request.date_range.start_date
                )
            if request.date_range.end_date:
                query = query.filter(
                    ImageRecord.processed_at <= request.date_range.end_date
                )
                
        # Anomaly score filter
        if request.anomaly_score:
            if request.anomaly_score.min_score is not None:
                query = query.filter(
                    ImageRecord.anomaly_score >= request.anomaly_score.min_score
                )
            if request.anomaly_score.max_score is not None:
                query = query.filter(
                    ImageRecord.anomaly_score <= request.anomaly_score.max_score
                )
        
        # Classification filters are more complex as they involve joins
        if request.classification:
            # User has classified the image
            if request.classification.is_classified is not None:
                if request.classification.is_classified:
                    # Images that have classifications
                    query = query.join(
                        Classification, 
                        ImageRecord.id == Classification.image_id
                    ).distinct()
                else:
                    # Images that don't have classifications
                    # This requires a subquery to find images without classifications
                    classified_image_ids = db.query(Classification.image_id).distinct().subquery()
                    query = query.filter(~ImageRecord.id.in_(classified_image_ids))
            
            # Filter by specific user classification (True/False)
            if request.classification.user_classification is not None and request.classification.is_classified:
                query = query.join(
                    Classification, 
                    ImageRecord.id == Classification.image_id
                ).filter(
                    Classification.user_classification == request.classification.user_classification
                ).distinct()
        
        # Count total matching records before pagination
        total_count = query.count()
        
        # Apply sorting
        if request.sort_by:
            sort_column = getattr(ImageRecord, request.sort_by, ImageRecord.processed_at)
            if request.sort_order and request.sort_order.lower() == "asc":
                query = query.order_by(sort_column.asc())
            else:
                query = query.order_by(sort_column.desc())
        
        # Apply pagination
        query = query.offset((request.page - 1) * request.page_size).limit(request.page_size)
        
        # Execute query
        records = query.all()
        
        # Calculate total pages
        total_pages = (total_count + request.page_size - 1) // request.page_size if total_count > 0 else 1
        
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
        return AdvancedFilterResponse(
            results=results,
            total_count=total_count,
            page=request.page,
            page_size=request.page_size,
            total_pages=total_pages,
        )
    except Exception as e:
        logger.error(f"Error filtering images: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error filtering images: {str(e)}"
        )


# Batch classify images
@app.post("/images/batch-classify")
async def batch_classify_images(
    request: BatchClassificationRequest, db: Session = Depends(get_db)
) -> BatchClassificationResponse:
    """
    Batch classify multiple images.
    
    This endpoint allows classifying multiple images at once with the same classification
    and optional comment.

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
        successful = 0
        failed_ids = []
        
        # Process each image ID
        for image_id in request.image_ids:
            # Call the existing classify_image function
            result = classify_image(
                image_id=image_id,
                is_anomaly=request.is_anomaly,
                comment=request.comment
            )
            
            if result:
                successful += 1
            else:
                failed_ids.append(image_id)
        
        # Return the results
        return BatchClassificationResponse(
            total=len(request.image_ids),
            successful=successful,
            failed=len(failed_ids),
            failed_ids=failed_ids
        )
    except Exception as e:
        logger.error(f"Error batch classifying images: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error batch classifying images: {str(e)}"
        )


# Dashboard statistics endpoint
@app.get("/dashboard/stats")
async def dashboard_stats(db: Session = Depends(get_db)) -> DashboardStatsResponse:
    """
    Get comprehensive statistics for dashboard visualization.
    
    This endpoint provides detailed statistics about anomaly detection results,
    including model performance metrics based on user feedback.

    Returns
    -------
    DashboardStatsResponse
        Detailed dashboard statistics
    """
    try:
        # Get basic counts
        total_images = db.query(func.count(ImageRecord.id)).scalar() or 0
        total_anomalies = db.query(func.count(ImageRecord.id)).filter(
            ImageRecord.is_anomaly == True  # noqa: E712
        ).scalar() or 0
        
        # Join with classifications to get more advanced metrics
        # First, get a subquery for the latest classification for each image
        latest_classifications = (
            db.query(
                Classification.image_id,
                func.max(Classification.timestamp).label("latest_timestamp")
            )
            .group_by(Classification.image_id)
            .subquery()
        )
        
        # Then join with classifications to get the actual latest classifications
        latest_class_query = (
            db.query(
                Classification,
                ImageRecord.is_anomaly
            )
            .join(
                latest_classifications,
                (Classification.image_id == latest_classifications.c.image_id) &
                (Classification.timestamp == latest_classifications.c.latest_timestamp)
            )
            .join(
                ImageRecord,
                Classification.image_id == ImageRecord.id
            )
        )
        
        # Get classification counts
        latest_class_results = latest_class_query.all()
        
        # Count user-confirmed anomalies (both model and user agree it's an anomaly)
        user_confirmed_anomalies = sum(
            1 for item in latest_class_results
            if item[0].user_classification is True and item[1] is True
        )
        
        # Count unclassified anomalies (model says anomaly, but no user classification)
        classified_image_ids = {item[0].image_id for item in latest_class_results}
        unclassified_anomalies_query = (
            db.query(ImageRecord)
            .filter(
                ImageRecord.is_anomaly == True,  # noqa: E712
                ~ImageRecord.id.in_(classified_image_ids)
            )
        )
        unclassified_anomalies = unclassified_anomalies_query.count() or 0
        
        # Count false positives (model says anomaly, user says normal)
        false_positives = sum(
            1 for item in latest_class_results
            if item[0].user_classification is False and item[1] is True
        )
        
        # Count false negatives (model says normal, user says anomaly)
        false_negatives = sum(
            1 for item in latest_class_results
            if item[0].user_classification is True and item[1] is False
        )
        
        # Get recent activity (limited to last 10)
        recent_activity_query = (
            db.query(Classification)
            .order_by(Classification.timestamp.desc())
            .limit(10)
        )
        recent_activity = [
            {
                "timestamp": classification.timestamp,
                "image_id": classification.image_id,
                "action": "classification",
                "is_anomaly": classification.user_classification
            }
            for classification in recent_activity_query.all()
        ]
        
        # Return the dashboard statistics
        return DashboardStatsResponse(
            total_images=total_images,
            total_anomalies=total_anomalies,
            user_confirmed_anomalies=user_confirmed_anomalies,
            unclassified_anomalies=unclassified_anomalies,
            false_positives=false_positives,
            false_negatives=false_negatives,
            recent_activity=recent_activity
        )
    except Exception as e:
        logger.error(f"Error generating dashboard statistics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating dashboard statistics: {str(e)}"
        )


# Get classification history for an image
@app.get("/images/{image_id}/classifications")
async def get_classification_history(
    image_id: str, db: Session = Depends(get_db)
) -> ClassificationHistoryResponse:
    """
    Get the complete classification history for a specific image.
    
    This endpoint returns all user classifications that have been made for a specific image,
    ordered by timestamp (most recent first).
    
    Parameters
    ----------
    image_id : str
        ID of the image to get classifications for
        
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
            
        # Get all classifications for this image, ordered by timestamp
        classifications = (
            db.query(Classification)
            .filter(Classification.image_id == image_id)
            .order_by(Classification.timestamp.desc())
            .all()
        )
        
        # Convert to response objects
        classification_responses = [
            ClassificationResponse(
                id=classification.id,
                image_id=classification.image_id,
                user_classification=classification.user_classification,
                comment=classification.comment,
                timestamp=classification.timestamp,
            )
            for classification in classifications
        ]
        
        # Return the history
        return ClassificationHistoryResponse(
            image_id=image_id,
            classifications=classification_responses
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving classification history: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving classification history: {str(e)}"
        )


# Export data in CSV or JSON format
@app.post("/export")
async def export_data(
    request: ExportRequest, db: Session = Depends(get_db)
) -> FileResponse:
    """
    Export anomaly detection data in CSV or JSON format.
    
    This endpoint allows exporting filtered data for further analysis or reporting.
    The export can be filtered using the same criteria as the advanced filter endpoint.
    
    Parameters
    ----------
    request : ExportRequest
        Export configuration including format, filters, and options
        
    Returns
    -------
    FileResponse
        File containing the exported data
    """
    try:
        # Start with a base query - include all image records
        query = db.query(ImageRecord)
        
        # Apply filters if provided
        if request.filter:
            if request.filter.is_anomaly is not None:
                query = query.filter(ImageRecord.is_anomaly == request.filter.is_anomaly)
                
            # Date range filter
            if request.filter.date_range:
                if request.filter.date_range.start_date:
                    query = query.filter(
                        ImageRecord.processed_at >= request.filter.date_range.start_date
                    )
                if request.filter.date_range.end_date:
                    query = query.filter(
                        ImageRecord.processed_at <= request.filter.date_range.end_date
                    )
                    
            # Anomaly score filter
            if request.filter.anomaly_score:
                if request.filter.anomaly_score.min_score is not None:
                    query = query.filter(
                        ImageRecord.anomaly_score >= request.filter.anomaly_score.min_score
                    )
                if request.filter.anomaly_score.max_score is not None:
                    query = query.filter(
                        ImageRecord.anomaly_score <= request.filter.anomaly_score.max_score
                    )
            
            # Classification filters
            if request.filter.classification:
                # User has classified the image
                if request.filter.classification.is_classified is not None:
                    if request.filter.classification.is_classified:
                        # Images that have classifications
                        query = query.join(
                            Classification, 
                            ImageRecord.id == Classification.image_id
                        ).distinct()
                    else:
                        # Images that don't have classifications
                        classified_image_ids = db.query(Classification.image_id).distinct().subquery()
                        query = query.filter(~ImageRecord.id.in_(classified_image_ids))
                
                # Filter by specific user classification (True/False)
                if request.filter.classification.user_classification is not None and request.filter.classification.is_classified:
                    query = query.join(
                        Classification, 
                        ImageRecord.id == Classification.image_id
                    ).filter(
                        Classification.user_classification == request.filter.classification.user_classification
                    ).distinct()
                    
        # Apply sorting
        if request.filter and request.filter.sort_by:
            sort_column = getattr(ImageRecord, request.filter.sort_by, ImageRecord.processed_at)
            if request.filter.sort_order and request.filter.sort_order.lower() == "asc":
                query = query.order_by(sort_column.asc())
            else:
                query = query.order_by(sort_column.desc())
        else:
            # Default sorting by processed_at desc
            query = query.order_by(ImageRecord.processed_at.desc())
        
        # Execute query
        records = query.all()
        
        # Prepare data for export
        export_data = []
        for record in records:
            # Basic image data
            image_data = {
                "id": record.id,
                "filename": record.filename,
                "processed_at": record.processed_at.isoformat() if record.processed_at else None,
                "reconstruction_error": record.reconstruction_error,
                "is_anomaly": record.is_anomaly,
                "anomaly_score": record.anomaly_score,
                "path": record.path,
            }
            
            # Include classifications if requested
            if request.include_classifications:
                # Get the latest classification for this image
                latest_classification = (
                    db.query(Classification)
                    .filter(Classification.image_id == record.id)
                    .order_by(Classification.timestamp.desc())
                    .first()
                )
                
                if latest_classification:
                    image_data.update({
                        "user_classification": latest_classification.user_classification,
                        "user_comment": latest_classification.comment,
                        "classification_timestamp": latest_classification.timestamp.isoformat() if latest_classification.timestamp else None,
                    })
                else:
                    image_data.update({
                        "user_classification": None,
                        "user_comment": None,
                        "classification_timestamp": None,
                    })
            
            export_data.append(image_data)
        
        # Create the export file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            if request.format == ExportFormat.CSV:
                # Convert to DataFrame for CSV export
                df = pd.DataFrame(export_data)
                df.to_csv(temp_file.name, index=False)
                media_type = "text/csv"
                filename = f"anomaly_data_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            else:  # Default to JSON
                # Write as JSON
                import json
                with open(temp_file.name, "w") as f:
                    json.dump(export_data, f, indent=2)
                media_type = "application/json"
                filename = f"anomaly_data_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            temp_file_path = temp_file.name
        
        # Return the file
        return FileResponse(
            path=temp_file_path,
            filename=filename,
            media_type=media_type,
        )
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error exporting data: {str(e)}"
        )


# Find similar anomalies to a reference image
@app.post("/images/{image_id}/similar")
async def find_similar_anomalies(
    image_id: str, request: SimilarAnomaliesRequest, db: Session = Depends(get_db)
) -> SimilarAnomaliesResponse:
    """
    Find images with similar anomaly patterns to a reference image.
    
    This endpoint finds images that exhibit similar anomaly patterns to the
    specified reference image, using anomaly score and reconstruction error.
    
    Parameters
    ----------
    image_id : str
        ID of the reference image to find similar anomalies for
    request : SimilarAnomaliesRequest
        Request parameters including limit and minimum similarity score
        
    Returns
    -------
    SimilarAnomaliesResponse
        List of similar images with similarity scores
    """
    try:
        # Get the reference image
        reference_image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
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
            .filter(ImageRecord.is_anomaly == True)  # Only anomalies
        )
        
        # Get all potential matches
        potential_matches = query.all()
        
        # Calculate similarity scores based on a combination of factors
        similarity_results = []
        for image in potential_matches:
            # Calculate score based on similarity in anomaly score and reconstruction error
            # Convert to numeric values to ensure we can do math with them
            ref_score = float(reference_image.anomaly_score) if reference_image.anomaly_score is not None else 0.0
            img_score = float(image.anomaly_score) if image.anomaly_score is not None else 0.0
            
            ref_error = float(reference_image.reconstruction_error) if reference_image.reconstruction_error is not None else 0.0
            img_error = float(image.reconstruction_error) if image.reconstruction_error is not None else 0.0
            
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
            combined_similarity = (normalized_score_similarity + normalized_error_similarity) / 2
            
            # Only include if above minimum score threshold
            if combined_similarity >= request.min_score:
                similarity_results.append({
                    "image": image,
                    "similarity_score": combined_similarity
                })
        
        # Sort by similarity score (highest first) and limit results
        similarity_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        limited_results = similarity_results[:request.limit]
        
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
                similarity_score=result["similarity_score"]
            )
            for result in limited_results
        ]
        
        # Return the results
        return SimilarAnomaliesResponse(
            reference_image_id=image_id,
            similar_images=response_results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar anomalies: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error finding similar anomalies: {str(e)}"
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
