"""
Visualization utilities for the anomaly reaper application.

This module provides functions for generating visualizations of
anomaly detection results using matplotlib, plotly, and other libraries.
"""

import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
from typing import Dict, Tuple, Union, Any


def encode_image_to_base64(image_data: Union[np.ndarray, Image.Image]) -> str:
    """
    Encode an image to base64 for transmission over HTTP.

    Parameters
    ----------
    image_data : Union[np.ndarray, Image.Image]
        The image data to encode

    Returns
    -------
    str
        Base64 encoded image data
    """
    if isinstance(image_data, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image_data)
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        raise ValueError("Input must be a PIL Image or a NumPy array")

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode a base64 string to a PIL Image.

    Parameters
    ----------
    base64_string : str
        Base64 encoded image data

    Returns
    -------
    Image.Image
        The decoded image
    """
    img_data = base64.b64decode(base64_string)
    buffer = io.BytesIO(img_data)
    img = Image.open(buffer)
    return img


def create_anomaly_heatmap(
    original_image: Union[np.ndarray, Image.Image],
    error_value: float,
    threshold: float,
    colormap: str = "viridis",
) -> Image.Image:
    """
    Create a heatmap overlay to visualize anomaly intensity.

    Parameters
    ----------
    original_image : Union[np.ndarray, Image.Image]
        The original image
    error_value : float
        Reconstruction error or anomaly score
    threshold : float
        Anomaly threshold
    colormap : str
        Matplotlib colormap name

    Returns
    -------
    Image.Image
        The heatmap visualization
    """
    # Convert to PIL if numpy array
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(original_image)

    # Create a figure with the original image
    fig, ax = plt.subplots(figsize=(10, 10))

    # Convert PIL to numpy for matplotlib
    img_array = np.array(original_image)

    # Display the original image
    ax.imshow(img_array)

    # Determine color intensity based on error value relative to threshold
    normalized_error = min(1.0, error_value / (threshold * 2))

    # Create a semi-transparent overlay based on the error intensity
    overlay = np.zeros((img_array.shape[0], img_array.shape[1], 4))

    # Set higher opacity where the error is higher
    overlay[:, :, 0] = (
        1.0 if error_value > threshold else 0.0
    )  # Red for anomaly, nothing for normal
    overlay[:, :, 3] = (
        normalized_error * 0.7
    )  # Alpha channel, more transparent for lower error

    ax.imshow(overlay, alpha=0.5, cmap=colormap)

    # Add text showing the error value
    status_text = (
        f"ANOMALY (Score: {error_value:.4f})"
        if error_value > threshold
        else f"Normal (Score: {error_value:.4f})"
    )
    ax.text(
        10,
        30,
        status_text,
        fontsize=12,
        color="white",
        bbox=dict(facecolor="black", alpha=0.7),
    )

    # Add a note about the threshold
    ax.text(
        10,
        img_array.shape[0] - 20,
        f"Threshold: {threshold:.4f}",
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )

    # Remove axes
    ax.axis("off")
    plt.tight_layout()

    # Convert matplotlib figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


def create_bounding_box_visualization(
    original_image: Union[np.ndarray, Image.Image],
    bbox: Dict[str, int],
    is_anomaly: bool,
    error_value: float,
    threshold: float,
) -> Image.Image:
    """
    Create a visualization with a bounding box around the anomaly.

    Parameters
    ----------
    original_image : Union[np.ndarray, Image.Image]
        The original image
    bbox : Dict[str, int]
        Bounding box coordinates (x_min, y_min, x_max, y_max)
    is_anomaly : bool
        Whether the image is an anomaly
    error_value : float
        Reconstruction error or anomaly score
    threshold : float
        Anomaly threshold

    Returns
    -------
    Image.Image
        The visualization with bounding box
    """
    # Convert to PIL if numpy array
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(original_image)

    # Create a figure with the original image
    fig, ax = plt.subplots(figsize=(10, 10))

    # Convert PIL to numpy for matplotlib
    img_array = np.array(original_image)

    # Display the original image
    ax.imshow(img_array)

    # Draw bounding box
    bbox_color = "red" if is_anomaly else "green"
    rect = plt.Rectangle(
        (bbox["x_min"], bbox["y_min"]),  # origin
        bbox["x_max"] - bbox["x_min"],  # width
        bbox["y_max"] - bbox["y_min"],  # height
        edgecolor=bbox_color,
        facecolor="none",
        linewidth=3,
    )
    ax.add_patch(rect)

    # Add text showing the error value
    status_text = (
        f"ANOMALY (Score: {error_value:.4f})"
        if is_anomaly
        else f"Normal (Score: {error_value:.4f})"
    )
    ax.text(
        10,
        30,
        status_text,
        fontsize=12,
        color="white",
        bbox=dict(facecolor="black", alpha=0.7),
    )

    # Remove axes
    ax.axis("off")
    plt.tight_layout()

    # Convert matplotlib figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)


def generate_pca_projection(
    embeddings_df: pd.DataFrame,
    highlight_anomalies: bool = True,
    use_interactive: bool = False,
    include_paths: bool = False,
) -> Tuple[Union[str, bytes], Dict[str, Any], float]:
    """
    Generate a PCA projection visualization of image embeddings.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        DataFrame containing embeddings data
    highlight_anomalies : bool
        Whether to highlight anomalies in the visualization
    use_interactive : bool
        Whether to create an interactive Plotly visualization
    include_paths : bool
        Whether to include image paths in the result

    Returns
    -------
    Tuple[Union[str, bytes], Dict[str, Any], float]
        Tuple containing:
        - The visualization (base64 encoded image or HTML)
        - Raw projection data
        - The anomaly threshold value
    """
    # Calculate the anomaly threshold
    threshold = (
        embeddings_df["reconstruction_error"].mean()
        + 2 * embeddings_df["reconstruction_error"].std()
    )

    # Create the projection data dictionary
    projection_data = {"points": [], "anomalies": []}

    # Populate projection data
    for _, row in embeddings_df.iterrows():
        point_data = {
            "x": float(row["PC1"]),
            "y": float(row["PC2"]),
            "error": float(row["reconstruction_error"]),
            "is_anomaly": bool(row["is_anomaly"]),
        }

        if include_paths and "image_path" in row:
            point_data["image_path"] = row["image_path"]

        if row["is_anomaly"]:
            projection_data["anomalies"].append(point_data)
        else:
            projection_data["points"].append(point_data)

    if use_interactive:
        # Create interactive Plotly visualization
        fig = px.scatter(
            embeddings_df,
            x="PC1",
            y="PC2",
            color="reconstruction_error",
            hover_data=["reconstruction_error"]
            + (["image_path"] if include_paths else []),
            title="PCA Projection of Image Embeddings",
            labels={
                "PC1": "Principal Component 1",
                "PC2": "Principal Component 2",
                "reconstruction_error": "Reconstruction Error",
            },
            color_continuous_scale="viridis",
        )

        if highlight_anomalies:
            # Add markers for anomalies
            anomalies = embeddings_df[embeddings_df["is_anomaly"]]
            fig.add_scatter(
                x=anomalies["PC1"],
                y=anomalies["PC2"],
                mode="markers",
                marker=dict(
                    size=15, color="rgba(255,0,0,0)", line=dict(color="red", width=2)
                ),
                name="Anomalies",
                hoverinfo="skip",
            )

        fig.update_layout(height=600, width=900)

        # Convert to HTML
        visualization = fig.to_html(include_plotlyjs="cdn", full_html=False)
        return visualization, projection_data, threshold
    else:
        # Create static matplotlib visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_df["PC1"],
            embeddings_df["PC2"],
            c=embeddings_df["reconstruction_error"],
            cmap="viridis",
            s=50,
            alpha=0.8,
        )

        if highlight_anomalies:
            # Highlight anomalies with a red circle
            anomalies = embeddings_df[embeddings_df["is_anomaly"]]
            plt.scatter(
                anomalies["PC1"],
                anomalies["PC2"],
                color="none",
                edgecolor="red",
                s=80,
                linewidth=2,
                label="Anomalies",
            )

        plt.colorbar(scatter, label="Reconstruction Error")
        plt.title("PCA Projection of Image Embeddings with Anomalies")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)

        visualization = base64.b64encode(buf.getvalue()).decode("utf-8")
        return visualization, projection_data, threshold


def calculate_anomaly_bbox(
    image_shape: Tuple[int, int], error_value: float, threshold: float
) -> Dict[str, int]:
    """
    Calculate a simulated bounding box for an anomaly.

    For demo purposes, this creates a bounding box that grows with the error value.
    In a real implementation, this would use more sophisticated techniques to identify
    the actual anomalous region.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        The height and width of the image
    error_value : float
        Reconstruction error or anomaly score
    threshold : float
        Anomaly threshold

    Returns
    -------
    Dict[str, int]
        Dictionary with bounding box coordinates
    """
    height, width = image_shape[:2]
    center_x, center_y = width // 2, height // 2

    # Box size depends on error value relative to threshold
    box_size_factor = min(0.8, max(0.2, error_value / (threshold * 3)))
    box_size = int(min(width, height) * box_size_factor)

    # Generate box coordinates
    x_min = max(0, center_x - box_size // 2)
    y_min = max(0, center_y - box_size // 2)
    x_max = min(width, center_x + box_size // 2)
    y_max = min(height, center_y + box_size // 2)

    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
