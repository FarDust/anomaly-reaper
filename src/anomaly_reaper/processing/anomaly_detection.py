"""
Image processing and anomaly detection functionality for the Anomaly Reaper.

This module provides functions for detecting anomalies in images using PCA-based
reconstruction error methods, as demonstrated in the Jupyter notebook.
"""

import os
import numpy as np
import joblib
from PIL import Image
import base64
import io
from typing import Tuple, Dict, Any, Union
from anomaly_reaper.config import settings, logger


def encode_image_base64(image):
    """Convert a PIL Image to base64 encoding, which is required by some Vertex AI models.

    Args:
        image (PIL.Image): The image to encode

    Returns:
        str: Base64 encoded image
    """

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image or a NumPy array.")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def get_image_embedding(image, dimension=1408):
    """Generate an embedding for an image using Vertex AI.

    Args:
        image: The image to embed (PIL Image or numpy array)
        dimension: Embedding dimension (default: 1408)

    Returns:
        np.ndarray: Image embedding vector
    """
    if settings.project_id is None:
        logger.warning(
            "GCP project_id not configured, using fallback feature extraction"
        )
        return extract_features_fallback(image)

    try:
        from vertexai.vision_models import MultiModalEmbeddingModel
        from vertexai.vision_models import Image as VertexImage
        import vertexai

        # Initialize Vertex AI with the project ID from settings
        vertexai.init(project=settings.project_id, location=settings.location)

        # Load the multimodal embedding model
        embeddings_model = MultiModalEmbeddingModel.from_pretrained(
            "multimodalembedding@001"
        )

        # Convert the image to a format suitable for Vertex AI
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a PIL Image or numpy array")

        # Encode the image
        encoded_image = encode_image_base64(pil_image)

        # Create Vertex AI image object
        vertex_image = VertexImage(
            image_bytes=base64.decodebytes(encoded_image),
        )

        # Generate the embedding
        embedding = embeddings_model.get_embeddings(
            image=vertex_image,
            dimension=dimension,
            contextual_text="deep space image with an anomaly detected by an automated system",
        ).image_embedding

        return np.array(embedding)
    except Exception as e:
        logger.error(f"Error generating Vertex AI embedding: {str(e)}")
        logger.warning("Falling back to basic feature extraction")
        return extract_features_fallback(image)


def extract_features(image) -> np.ndarray:
    """Extract features from an image.

    This is a wrapper around get_image_embedding to maintain API compatibility.

    Args:
        image: Input image as PIL Image or numpy array

    Returns:
        Feature vector as numpy array
    """
    return get_image_embedding(image)


def extract_features_fallback(image: np.ndarray) -> np.ndarray:
    """Fallback feature extraction when Vertex AI is not available.

    Args:
        image: Input image as numpy array

    Returns:
        Feature vector
    """
    # Basic feature extraction - resize and flatten
    from skimage.transform import resize
    from skimage.feature import hog

    # Convert to array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure consistent size
    image_resized = resize(image, (128, 128), anti_aliasing=True)

    # Convert to grayscale if color
    if len(image_resized.shape) == 3 and image_resized.shape[2] == 3:
        image_gray = np.mean(image_resized, axis=2)
    else:
        image_gray = image_resized

    features = hog(
        image_gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=False,
    )

    return features


def compute_reconstruction_error(
    original: np.ndarray, reconstructed: np.ndarray
) -> float:
    """Compute the reconstruction error between original and reconstructed data.

    Args:
        original: Original feature vector
        reconstructed: Reconstructed feature vector

    Returns:
        Reconstruction error (Euclidean distance)
    """
    return float(np.linalg.norm(original - reconstructed))


def detect_anomaly(reconstruction_error: float, threshold: float) -> Tuple[bool, float]:
    """Determine if the reconstruction error indicates an anomaly.

    Args:
        reconstruction_error: Reconstruction error value
        threshold: Threshold for anomaly detection

    Returns:
        Tuple containing:
        - Boolean indicating if anomaly is detected
        - Anomaly score (error/threshold ratio)
    """
    is_anomaly = reconstruction_error >= threshold
    score = reconstruction_error / threshold if threshold > 0 else float("inf")
    return is_anomaly, score


def process_image(image: Union[str, Image.Image], model_dir: str) -> Dict[str, Any]:
    """Process an image and detect if it contains anomalies.

    Args:
        image: Path to the image file or a PIL Image object
        model_dir: Directory containing the PCA model files

    Returns:
        Dictionary with detection results including:
        - reconstruction_error: The calculated reconstruction error
        - is_anomaly: Boolean indicating if an anomaly was detected
        - anomaly_score: Normalized anomaly score (ratio of error to threshold)
    """
    # Load the model
    pca, scaler, threshold = load_model(model_dir)

    # Handle both file paths and PIL Images
    if isinstance(image, str):
        img = Image.open(image)
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        img = image
        if img.mode != "RGB":
            img = img.convert("RGB")

    # Generate embedding using Vertex AI (or fallback to basic features)
    features = get_image_embedding(img)

    # Reshape if needed for the scaler
    features = features.reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(features)

    # Apply PCA transformation
    pca_features = pca.transform(scaled_features)

    # Reconstruct the data
    reconstructed = pca.inverse_transform(pca_features)

    # Calculate reconstruction error
    error = compute_reconstruction_error(scaled_features[0], reconstructed[0])

    # Determine if anomaly
    is_anomaly, anomaly_score = detect_anomaly(error, threshold)

    return {
        "reconstruction_error": float(error),
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": float(anomaly_score),
        "threshold": float(threshold),
    }


def load_model(model_dir: str) -> Tuple[Any, Any, float]:
    """Load the PCA model and create a scaler from the given directory.

    Args:
        model_dir: Directory containing the model files

    Returns:
        Tuple of (pca_model, scaler, threshold)
    """
    import json

    # Try to load from GCS if cloud storage is enabled
    if settings.use_cloud_storage and settings.project_id and settings.gcs_bucket_name:
        try:
            from google.cloud import storage

            # Initialize GCS
            client = storage.Client(project=settings.project_id)
            bucket = client.bucket(settings.gcs_bucket_name)

            # Use 'anomaly_reaper/' as the base path rather than the images prefix
            # to avoid looking in the wrong location
            gcs_base_path = "anomaly_reaper/"

            # Path in GCS
            gcs_path = f"{gcs_base_path}pca_model.pkl"
            blob = bucket.blob(gcs_path)

            # Download to local temp file
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as temp:
                blob.download_to_filename(temp.name)
                pca_model = joblib.load(temp.name)

            logger.info(
                f"Loaded PCA model from GCS: gs://{settings.gcs_bucket_name}/{gcs_path}"
            )

            # Try to get scaler from GCS too
            try:
                gcs_scaler_path = f"{gcs_base_path}scaler.pkl"
                blob = bucket.blob(gcs_scaler_path)
                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    blob.download_to_filename(temp.name)
                    scaler = joblib.load(temp.name)
                logger.info(
                    f"Loaded scaler from GCS: gs://{settings.gcs_bucket_name}/{gcs_scaler_path}"
                )
            except Exception as e:
                logger.warning(f"Couldn't load scaler from GCS: {str(e)}")
                # Create a fitted scaler instead of failing
                logger.info("Creating pre-fitted scaler instead")
                scaler = create_prefitted_scaler()

            # Get threshold from GCS config if available
            try:
                gcs_config_path = f"{gcs_base_path}config.json"
                blob = bucket.blob(gcs_config_path)
                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    blob.download_to_filename(temp.name)
                    with open(temp.name, "r") as f:
                        config = json.load(f)
                        threshold = float(config.get("threshold", 0.05))
                logger.info(f"Loaded config from GCS with threshold: {threshold}")
            except Exception:
                threshold = 0.05  # Default

            return pca_model, scaler, threshold

        except Exception as e:
            logger.error(
                f"Error loading model from GCS: {str(e)}, falling back to local files"
            )

    # Fall back to local files
    # Load PCA model
    pca_path = os.path.join(model_dir, "pca_model.pkl")
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA model not found at {pca_path}")
    pca_model = joblib.load(pca_path)

    # Try to load scaler from file, create one if not found
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.warning(f"Error loading scaler from {scaler_path}: {str(e)}")
            # Create a fitted scaler as fallback
            logger.info("Creating pre-fitted scaler instead")
            scaler = create_prefitted_scaler()
    else:
        # Create a fitted scaler since we don't have one
        logger.info(f"Scaler not found at {scaler_path}, creating pre-fitted scaler")
        scaler = create_prefitted_scaler()

    # Load configuration (threshold, etc.)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            threshold = float(config.get("threshold", 0.05))
    else:
        threshold = 0.05  # Default threshold

    return pca_model, scaler, threshold


def create_prefitted_scaler():
    """Create a pre-fitted StandardScaler with statistics from the actual embeddings data.

    Instead of loading from a scaler.pkl file, we initialize a scaler with statistics
    calculated from the saved embeddings in the CSV file. We then save this fitted
    scaler for future use.

    Returns:
        StandardScaler: A pre-fitted standard scaler
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import os
    from pathlib import Path

    # First try to load an existing scaler.pkl
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "models",
    )
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if os.path.exists(scaler_path):
        try:
            logger.info(f"Loading existing scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)

            # If cloud storage is enabled, check if we need to upload the existing scaler to GCS
            if (
                settings.use_cloud_storage
                and settings.project_id
                and settings.gcs_bucket_name
            ):
                try:
                    from google.cloud import storage

                    # Initialize GCS client
                    logger.info("Checking if scaler needs to be uploaded to GCS...")
                    client = storage.Client(project=settings.project_id)
                    bucket = client.bucket(settings.gcs_bucket_name)

                    # Define the correct GCS path
                    gcs_base_path = "anomaly_reaper/"
                    gcs_path = f"{gcs_base_path}scaler.pkl"

                    # Check if the scaler already exists in GCS
                    blob = bucket.blob(gcs_path)
                    if not blob.exists():
                        logger.info(
                            f"Scaler not found in GCS at {gcs_path}, uploading local version..."
                        )
                        blob.upload_from_filename(scaler_path)
                        logger.info(
                            f"Successfully uploaded scaler to GCS at gs://{settings.gcs_bucket_name}/{gcs_path}"
                        )
                    else:
                        logger.info(
                            f"Scaler already exists in GCS at gs://{settings.gcs_bucket_name}/{gcs_path}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to check/upload scaler to GCS: {str(e)}")

            return scaler
        except Exception as e:
            logger.warning(f"Failed to load existing scaler: {str(e)}")

    logger.info("Creating new pre-fitted scaler")
    scaler = StandardScaler()

    # Try to load the embeddings data from CSV to get real statistics
    try:
        # Look for embeddings file in data directory
        data_path = Path(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        )

        # Try both potential CSV file names
        embeddings_csv = data_path / "data" / "image_embeddings.csv"
        embeddings_df_csv = data_path / "data" / "embeddings_df.csv"

        if embeddings_csv.exists():
            logger.info(
                f"Loading embeddings from {embeddings_csv} to create pre-fitted scaler"
            )
            # Load the embeddings and calculate real statistics
            embeddings_df = pd.read_csv(embeddings_csv)
            embeddings = embeddings_df.values

            # Calculate mean and std from the real embeddings
            scaler.mean_ = np.mean(embeddings, axis=0)
            scaler.var_ = np.var(embeddings, axis=0)
            scaler.scale_ = np.sqrt(scaler.var_)

            # Mark as fitted with the actual dimension and sample count
            scaler.n_features_in_ = embeddings.shape[1]
            scaler.n_samples_seen_ = embeddings.shape[0]

            logger.info(
                f"Created pre-fitted scaler with statistics from {embeddings.shape[0]} samples of dimension {embeddings.shape[1]}"
            )
        elif embeddings_df_csv.exists():
            logger.info(
                f"Loading embeddings from {embeddings_df_csv} to create pre-fitted scaler"
            )
            # Try the other CSV file format
            embeddings_df = pd.read_csv(embeddings_df_csv)

            # Filter out non-numeric columns
            numeric_cols = embeddings_df.select_dtypes(include=[np.number]).columns

            if "PC1" in embeddings_df.columns and "PC2" in embeddings_df.columns:
                logger.info(
                    "Found PCs in embeddings_df.csv, using original embeddings if available"
                )
                # This is the reduced dimension data, not useful for creating a scaler
                # Try to use image_embeddings.csv instead
                if embeddings_csv.exists():
                    embeddings_df = pd.read_csv(embeddings_csv)
                    embeddings = embeddings_df.values
                else:
                    logger.warning("Using PC data to create scaler (not ideal)")
                    embeddings = embeddings_df[numeric_cols].values
            else:
                embeddings = embeddings_df[numeric_cols].values

            # Calculate mean and std from the real embeddings
            scaler.mean_ = np.mean(embeddings, axis=0)
            scaler.var_ = np.var(embeddings, axis=0)
            scaler.scale_ = np.sqrt(scaler.var_)

            # Mark as fitted with the actual dimension and sample count
            scaler.n_features_in_ = embeddings.shape[1]
            scaler.n_samples_seen_ = embeddings.shape[0]

            logger.info(
                f"Created pre-fitted scaler with statistics from {embeddings.shape[0]} samples of dimension {embeddings.shape[1]}"
            )
        else:
            logger.warning(
                f"No embeddings file found at {embeddings_csv} or {embeddings_df_csv}, using default scaler values"
            )
            # Fallback to default values if loading fails
            embedding_dim = 1408  # Dimension of your image embeddings

            # Initialize with reasonable default values
            scaler.mean_ = np.zeros(embedding_dim)
            scaler.scale_ = np.ones(embedding_dim)
            scaler.var_ = np.ones(embedding_dim)

            # Mark as fitted
            scaler.n_features_in_ = embedding_dim
            scaler.n_samples_seen_ = 100

            logger.info(
                f"Created default pre-fitted scaler with dimension {embedding_dim}"
            )
    except Exception as e:
        logger.warning(
            f"Error loading embeddings for scaler: {str(e)}, using default scaler values"
        )
        # Fallback to default values
        embedding_dim = 1408
        scaler.mean_ = np.zeros(embedding_dim)
        scaler.scale_ = np.ones(embedding_dim)
        scaler.var_ = np.ones(embedding_dim)
        scaler.n_features_in_ = embedding_dim
        scaler.n_samples_seen_ = 100

    # Save the scaler to disk for future use
    try:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        # If we're using GCS, save it there too
        if (
            settings.use_cloud_storage
            and settings.project_id
            and settings.gcs_bucket_name
        ):
            try:
                from google.cloud import storage

                # Initialize GCS client
                client = storage.Client(project=settings.project_id)
                bucket = client.bucket(settings.gcs_bucket_name)

                # Use 'anomaly_reaper/' as the base path (not the images prefix)
                gcs_base_path = "anomaly_reaper/"

                # Path in GCS
                gcs_path = f"{gcs_base_path}scaler.pkl"
                blob = bucket.blob(gcs_path)

                # Save to a temporary file and upload
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    joblib.dump(scaler, temp.name)
                    temp.flush()  # Ensure all data is written to disk
                    blob.upload_from_filename(temp.name)
                    os.unlink(temp.name)  # Clean up the temp file

                logger.info(
                    f"Uploaded scaler to GCS: gs://{settings.gcs_bucket_name}/{gcs_path}"
                )
            except Exception as e:
                logger.warning(f"Couldn't save scaler to GCS: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to save scaler: {str(e)}")

    return scaler
