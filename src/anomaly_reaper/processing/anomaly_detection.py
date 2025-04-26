"""
Image processing and anomaly detection functionality for the Anomaly Reaper.

This module provides functions for detecting anomalies in images using PCA-based
reconstruction error methods, as demonstrated in the Jupyter notebook.
"""

import os
import numpy as np
import joblib
from PIL import Image
from typing import Tuple, Dict, Any, Union


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
    return np.linalg.norm(original - reconstructed)


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


def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract features from an image for anomaly detection.

    This is a simplified version that flattens and normalizes the image.
    In production, more sophisticated feature extraction could be used.

    Args:
        image: Input image as numpy array

    Returns:
        Feature vector
    """
    # Basic feature extraction - resize and flatten
    # In a real implementation, this could use VertexAI embeddings as in the notebook
    features = np.array(image, dtype=np.float32)

    # Normalize to [0,1] range
    if features.max() > 1.0:
        features = features / 255.0

    # Flatten to a 1D vector
    return features.flatten()


def process_image(image: Union[str, Image.Image], model_dir: str) -> Dict[str, Any]:
    """Process an image and detect if it contains anomalies.

    Args:
        image: Path to the image file or a PIL Image object
        model_dir: Directory containing the PCA model files

    Returns:
        Dictionary with detection results including:
        - reconstruction_error: The calculated reconstruction error
        - is_anomaly: Boolean indicating if an anomaly was detected
        - anomaly_score: Normalized anomaly score (0-1)
    """
    # Load the model
    pca, scaler, threshold = load_model(model_dir)

    # Handle both file paths and PIL Images
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image

    # Convert to numpy array
    img_array = np.array(img)

    # Extract features
    features = extract_features(img_array)

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

    # Normalize the anomaly score to a 0-1 range (optional)
    # This assumes error typically ranges from 0 to 2*threshold
    anomaly_score = min(anomaly_score, 1.0)

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
