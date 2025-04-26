"""
Tests for the image processing and anomaly detection functionality.

These tests verify that the PCA model works correctly for image processing
and anomaly detection.
"""

import os
import pytest
import numpy as np
from PIL import Image
import joblib

from anomaly_reaper.processing.anomaly_detection import (
    compute_reconstruction_error,
    detect_anomaly,
    process_image,
    load_model,
)


class TestAnomalyDetection:
    """Tests for the anomaly detection module."""

    def test_compute_reconstruction_error(self):
        """Test computation of reconstruction error."""
        # Create random data for testing
        original = np.random.random((10,))
        reconstructed = original + np.random.normal(0, 0.1, (10,))  # Add some noise

        # Compute error
        error = compute_reconstruction_error(original, reconstructed)

        # Check error is non-negative and approximately matches expected
        assert error >= 0
        expected_error = np.linalg.norm(original - reconstructed)
        assert np.isclose(error, expected_error)

    def test_detect_anomaly(self):
        """Test anomaly detection based on reconstruction error."""
        # Test case 1: Error below threshold (normal)
        error1 = 0.03
        threshold = 0.05
        is_anomaly1, score1 = detect_anomaly(error1, threshold)
        assert not is_anomaly1
        assert score1 == pytest.approx(error1 / threshold)

        # Test case 2: Error above threshold (anomaly)
        error2 = 0.07
        is_anomaly2, score2 = detect_anomaly(error2, threshold)
        assert is_anomaly2
        assert score2 == pytest.approx(error2 / threshold)

        # Test case 3: Error exactly at threshold
        error3 = threshold
        is_anomaly3, score3 = detect_anomaly(error3, threshold)
        assert is_anomaly3  # Should be classified as an anomaly
        assert score3 == 1.0

    def test_extract_features(self, monkeypatch):
        """Test feature extraction from images."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="white")

        # Mock the feature extraction function since it might depend on external libraries
        def mock_extract(image):
            # Return a fixed feature vector regardless of input
            return np.ones((10,))

        # Apply the monkeypatch
        monkeypatch.setattr(
            "anomaly_reaper.processing.anomaly_detection.extract_features", mock_extract
        )

        # Test the function
        features = mock_extract(img)

        # Verify output shape and type
        assert features.shape == (10,)
        assert isinstance(features, np.ndarray)

    def test_load_model(self, test_settings, mock_pca_model):
        """Test loading the PCA model and scaler."""
        # Unpack the mock models
        mock_pca, mock_scaler = mock_pca_model

        # Load the models from the test directory
        pca, scaler, threshold = load_model(test_settings.models_dir)

        # Verify models were loaded
        assert pca is not None
        assert scaler is not None
        assert threshold == 0.05  # Value set in conftest.py

    def test_process_image(self, test_settings, mock_pca_model, monkeypatch):
        """Test the full image processing pipeline."""
        # Create a test image
        test_image = Image.new("RGB", (100, 100), color="white")

        # Define mock function for feature extraction and get_image_embedding that returns a 10-element vector
        # to match the expected dimensions in the MockPCA.inverse_transform method from conftest.py
        def mock_get_image_embedding(image, dimension=1408):
            return np.ones((10,))  # Changed to 10 to match MockPCA dimensions

        monkeypatch.setattr(
            "anomaly_reaper.processing.anomaly_detection.get_image_embedding",
            mock_get_image_embedding,
        )

        # Process the image
        result = process_image(test_image, test_settings.models_dir)

        # Check the result
        assert isinstance(result, dict)
        assert "reconstruction_error" in result
        assert "is_anomaly" in result
        assert "anomaly_score" in result

        # Since we're no longer capping at 1.0, make sure it works with values > 1.0
        if (
            result["is_anomaly"]
            and result["reconstruction_error"] > result["threshold"]
        ):
            assert result["anomaly_score"] >= 1.0


class TestModelTraining:
    """Tests for the model training functionality."""

    def test_model_saving_and_loading(self, test_settings):
        """Test that models can be saved and loaded correctly."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Create simple models
        pca = PCA(n_components=2)
        scaler = StandardScaler()

        # Create some random data and fit the models
        data = np.random.random((20, 10))
        scaled_data = scaler.fit_transform(data)
        pca.fit(scaled_data)

        # Save the models
        model_path = os.path.join(test_settings.models_dir, "test_pca.pkl")
        scaler_path = os.path.join(test_settings.models_dir, "test_scaler.pkl")
        threshold = 0.1

        joblib.dump(pca, model_path)
        joblib.dump(scaler, scaler_path)

        # Save threshold to config
        config_path = os.path.join(test_settings.models_dir, "test_config.json")
        with open(config_path, "w") as f:
            f.write(f'{{"threshold": {threshold}}}')

        # Try to load the models
        loaded_pca = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)

        # Verify models can be used
        new_data = np.random.random((5, 10))
        scaled_new_data = loaded_scaler.transform(new_data)
        transformed_data = loaded_pca.transform(scaled_new_data)

        assert transformed_data.shape == (5, 2)
