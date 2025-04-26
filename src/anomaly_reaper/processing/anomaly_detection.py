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
    """Load the PCA model and scaler from the given directory.

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

            # Get GCS path from environment variables
            gcs_prefix = settings.gcs_images_prefix or ""
            if not gcs_prefix.endswith("/"):
                gcs_prefix += "/"

            # Path in GCS
            gcs_path = f"{gcs_prefix}pca_model.pkl"
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
                gcs_scaler_path = f"{gcs_prefix}scaler.pkl"
                blob = bucket.blob(gcs_scaler_path)
                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    blob.download_to_filename(temp.name)
                    scaler = joblib.load(temp.name)
                logger.info(
                    f"Loaded scaler from GCS: gs://{settings.gcs_bucket_name}/{gcs_scaler_path}"
                )
            except Exception as e:
                logger.warning(f"Couldn't load scaler from GCS: {str(e)}")
                # Create a basic scaler as fallback
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()

            # Get threshold from GCS config if available
            try:
                gcs_config_path = f"{gcs_prefix}config.json"
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

    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Load configuration (threshold, etc.)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            threshold = float(config.get("threshold", 0.05))
    else:
        threshold = 0.05  # Default threshold

    return pca_model, scaler, threshold
