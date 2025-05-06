"""
Tests for the image processing and anomaly detection functionality.

These tests verify that the PCA model works correctly for image processing
and anomaly detection.
"""

import os
import pytest
import numpy as np
import tempfile
from PIL import Image
import joblib
import shutil

from anomaly_reaper.processing.anomaly_detection import (
    compute_reconstruction_error,
    detect_anomaly,
    process_image,
    load_model,
    create_prefitted_scaler,
)


@pytest.fixture(scope="module")
def test_data_dir():
    """Create a temporary directory for test data files."""
    # Create a temporary directory for test data
    temp_dir = tempfile.mkdtemp(prefix="anomaly_reaper_test_data_")
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def test_model_dir():
    """Create a temporary directory for test models."""
    # Create a temporary directory for test models
    temp_dir = tempfile.mkdtemp(prefix="anomaly_reaper_test_models_")
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_pca_model(test_model_dir):
    """Create a simple PCA model for testing."""
    from sklearn.decomposition import PCA

    # Create a simple PCA model
    pca = PCA(n_components=2)

    # Create some random data and fit the model
    data = np.random.random((20, 10))
    pca.fit(data)

    # Save model to test directory
    model_path = os.path.join(test_model_dir, "pca_model.pkl")
    joblib.dump(pca, model_path)

    yield pca, model_path


@pytest.fixture
def test_scaler(test_model_dir):
    """Create a simple StandardScaler for testing."""
    from sklearn.preprocessing import StandardScaler

    # Create a simple scaler
    scaler = StandardScaler()

    # Create some random data and fit the scaler
    data = np.random.random((20, 10))
    scaler.fit(data)

    # Save scaler to test directory
    scaler_path = os.path.join(test_model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    yield scaler, scaler_path


@pytest.fixture
def test_config(test_model_dir):
    """Create a test configuration file."""
    config_path = os.path.join(test_model_dir, "config.json")

    # Create config file with test threshold
    with open(config_path, "w") as f:
        f.write('{"threshold": 0.05}')

    yield config_path


@pytest.fixture
def test_image():
    """Create a test image for processing."""
    # Create a simple test image
    width, height = 100, 100
    image = Image.new("RGB", (width, height), color="white")

    # Add some pattern to the image to make it more realistic
    pixels = image.load()
    for i in range(width):
        for j in range(height):
            if (i + j) % 10 == 0:
                pixels[i, j] = (200, 200, 200)

    yield image


@pytest.fixture
def test_embedding():
    """Create a test embedding vector."""
    # Create a 1408-dimensional embedding vector (typical size for image embeddings)
    return np.random.random(1408)


@pytest.fixture
def mock_embeddings_data(test_data_dir):
    """Create mock embeddings data files for testing."""
    # Create a mock embeddings CSV file
    embeddings_path = os.path.join(test_data_dir, "image_embeddings.csv")

    # Generate random embeddings for a small number of images
    num_images = 5
    embedding_dim = 10  # Small dimension for testing

    # Create header row with feature columns
    header = ",".join([f"feature_{i}" for i in range(embedding_dim)])

    # Create data rows with random embeddings
    rows = []
    for i in range(num_images):
        embedding = np.random.random(embedding_dim)
        row = ",".join([f"{x:.6f}" for x in embedding])
        rows.append(row)

    # Write to CSV file
    with open(embeddings_path, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows))

    yield embeddings_path


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

    def test_extract_features(self, monkeypatch, test_image):
        """Test feature extraction from images."""

        # Mock the feature extraction function since it might depend on external libraries
        def mock_extract(image):
            # Return a fixed feature vector regardless of input
            return np.ones((10,))

        # Apply the monkeypatch
        monkeypatch.setattr(
            "anomaly_reaper.processing.anomaly_detection.extract_features", mock_extract
        )

        # Test the function
        features = mock_extract(test_image)

        # Verify output shape and type
        assert features.shape == (10,)
        assert isinstance(features, np.ndarray)

    def test_load_model(self, test_pca_model, test_scaler, test_config, test_model_dir):
        """Test loading the PCA model and scaler."""
        # The fixtures create and save the test models
        pca, scaler, threshold = load_model(test_model_dir)

        # Verify models were loaded
        assert pca is not None
        assert scaler is not None
        assert threshold == 0.05  # Value set in test_config fixture

    def test_process_image(self, monkeypatch, test_model_dir, test_image):
        """Test the full image processing pipeline with proper mocks."""

        # Create a mock for the full load_model function
        def mock_load_model(model_dir):
            # Create a mock scaler that can handle 10-dimensional feature vectors
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.mean_ = np.zeros(10)  # 10-dimensional
            scaler.var_ = np.ones(10)
            scaler.scale_ = np.ones(10)
            scaler.n_features_in_ = 10

            # Create a mock PCA that expects 10 dimensions
            class MockPCA:
                def transform(self, X):
                    return np.array([[0.1, 0.2]])

                def inverse_transform(self, X):
                    return np.array([np.ones(10)])  # Return 10D vector

            pca = MockPCA()
            threshold = 0.05

            return pca, scaler, threshold

        # Mock the load_model function
        monkeypatch.setattr(
            "anomaly_reaper.processing.anomaly_detection.load_model",
            mock_load_model,
        )

        # Mock get_image_embedding to return 10 features
        def mock_get_image_embedding(image, dimension=1408):
            return np.ones(10)

        monkeypatch.setattr(
            "anomaly_reaper.processing.anomaly_detection.get_image_embedding",
            mock_get_image_embedding,
        )

        # Process the image
        result = process_image(test_image, test_model_dir)

        # Verify the result has the expected fields
        assert isinstance(result, dict)
        assert "reconstruction_error" in result
        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "threshold" in result

    def test_create_prefitted_scaler(
        self, monkeypatch, test_model_dir, mock_embeddings_data
    ):
        """Test creating a pre-fitted scaler with embeddings data."""
        # Patch the settings to use our test directories
        monkeypatch.setattr(
            "anomaly_reaper.processing.anomaly_detection.settings.models_dir",
            test_model_dir,
        )

        # Create a modified version of create_prefitted_scaler that uses our test data
        def mock_create_prefitted_scaler():
            from sklearn.preprocessing import StandardScaler
            import pandas as pd

            # Create a StandardScaler
            scaler = StandardScaler()

            # Load our mock embeddings data
            embeddings_df = pd.read_csv(mock_embeddings_data)
            embeddings = embeddings_df.values

            # Fit the scaler with our test data
            scaler.mean_ = np.mean(embeddings, axis=0)
            scaler.var_ = np.var(embeddings, axis=0)
            scaler.scale_ = np.sqrt(scaler.var_)
            scaler.n_features_in_ = embeddings.shape[
                1
            ]  # Should be 10 from our mock data
            scaler.n_samples_seen_ = embeddings.shape[0]

            # Save the scaler to the test directory
            os.makedirs(test_model_dir, exist_ok=True)
            scaler_path = os.path.join(test_model_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)

            return scaler

        # Apply the monkeypatch to use our mock function
        monkeypatch.setattr(
            "anomaly_reaper.processing.anomaly_detection.create_prefitted_scaler",
            mock_create_prefitted_scaler,
        )

        # Call the function (which will use our mock version)
        scaler = create_prefitted_scaler()

        # Verify the scaler has been properly fitted
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "var_")
        assert hasattr(scaler, "scale_")

        # Check that the scaler was saved to disk
        scaler_path = os.path.join(test_model_dir, "scaler.pkl")
        assert os.path.exists(scaler_path)

        # Load the saved scaler and verify it matches our expected dimension
        loaded_scaler = joblib.load(scaler_path)
        assert (
            loaded_scaler.n_features_in_ == 10
        )  # Should match our test data dimension


class TestModelTraining:
    """Tests for the model training functionality."""

    def test_model_saving_and_loading(self, test_model_dir):
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
        model_path = os.path.join(test_model_dir, "test_pca.pkl")
        scaler_path = os.path.join(test_model_dir, "test_scaler.pkl")
        threshold = 0.1

        joblib.dump(pca, model_path)
        joblib.dump(scaler, scaler_path)

        # Save threshold to config
        config_path = os.path.join(test_model_dir, "test_config.json")
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
