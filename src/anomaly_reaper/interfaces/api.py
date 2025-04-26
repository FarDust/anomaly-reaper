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
