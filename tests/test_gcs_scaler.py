"""
Integration test for verifying that a scaler is correctly uploaded to GCS
when cloud storage is enabled.
"""

import os
import pytest
import tempfile
import joblib
import shutil
from unittest.mock import patch, MagicMock
import numpy as np

from anomaly_reaper.config import Settings
from anomaly_reaper.processing.anomaly_detection import (
    load_model,
    create_prefitted_scaler,
)


# Define mockable classes at module level for proper pickling
class MockPCA:
    """Serializable mock PCA class for testing."""

    def __init__(self):
        self.components_ = np.array([[1, 0, 0], [0, 1, 0]])
        self.mean_ = np.array([0, 0, 0])

    def transform(self, X):
        return np.array([[0.1, 0.2]])

    def inverse_transform(self, X):
        return np.array([[0.1, 0.2, 0.3]])


class MockStorage:
    """Mock Google Cloud Storage for testing."""

    def __init__(self):
        self.blobs = {}
        self.uploaded_files = {}

    def client(self, **kwargs):
        """Create a mock client."""
        mock_client = MagicMock()
        mock_client.bucket = self.bucket
        return mock_client

    def bucket(self, bucket_name):
        """Create a mock bucket."""
        mock_bucket = MagicMock()
        mock_bucket.blob = self.blob
        mock_bucket.name = bucket_name
        return mock_bucket

    def blob(self, blob_name):
        """Create a mock blob."""
        if blob_name not in self.blobs:
            mock_blob = MagicMock()
            mock_blob.name = blob_name
            mock_blob.upload_from_filename = lambda filename: self.store_upload(
                blob_name, filename
            )
            mock_blob.download_to_filename = lambda filename: self.download_to_file(
                blob_name, filename
            )
            mock_blob.exists = lambda: blob_name in self.uploaded_files
            self.blobs[blob_name] = mock_blob
        return self.blobs[blob_name]

    def store_upload(self, blob_name, filename):
        """Store an uploaded file."""
        with open(filename, "rb") as f:
            self.uploaded_files[blob_name] = f.read()

    def download_to_file(self, blob_name, filename):
        """Download a file to a local path."""
        if blob_name not in self.uploaded_files:
            raise Exception(f"Blob {blob_name} not found")

        with open(filename, "wb") as f:
            f.write(self.uploaded_files[blob_name])

    def has_blob(self, blob_name):
        """Check if a blob exists."""
        return blob_name in self.uploaded_files

    def reset(self):
        """Reset the mock storage state."""
        self.blobs = {}
        self.uploaded_files = {}


@pytest.fixture(scope="module")
def test_base_dirs():
    """Create temporary directories for models, uploads, and data."""
    # Create temporary directories
    models_dir = tempfile.mkdtemp(prefix="test_gcs_models_")
    uploads_dir = tempfile.mkdtemp(prefix="test_gcs_uploads_")
    data_dir = tempfile.mkdtemp(prefix="test_gcs_data_")

    # Create fixtures directory if it doesn't exist yet
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    yield models_dir, uploads_dir, data_dir, fixtures_dir

    # Clean up after tests
    shutil.rmtree(models_dir, ignore_errors=True)
    shutil.rmtree(uploads_dir, ignore_errors=True)
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def gcs_settings(test_base_dirs):
    """Create test settings with GCS enabled."""
    models_dir, uploads_dir, data_dir, _ = test_base_dirs

    # Create test settings with GCS enabled
    test_settings = Settings(
        models_dir=models_dir,
        uploads_dir=uploads_dir,
        data_dir=data_dir,
        use_cloud_storage=True,
        project_id="test-project",
        gcs_bucket_name="test-bucket",
        gcs_images_prefix="anomaly_reaper/images",
    )

    yield test_settings


@pytest.fixture
def mock_storage():
    """Create a mock storage object that resets between tests."""
    storage = MockStorage()
    yield storage
    # Reset the mock storage after each test
    storage.reset()


@pytest.fixture
def mock_scaler_file(test_base_dirs):
    """Create a mock scaler file for testing."""
    _, _, _, fixtures_dir = test_base_dirs

    scaler_path = os.path.join(fixtures_dir, "mock_scaler.pkl")

    # Create the scaler if it doesn't exist
    if not os.path.exists(scaler_path):
        from sklearn.preprocessing import StandardScaler

        # Create a simple scaler for testing
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(3)
        scaler.var_ = np.ones(3)
        scaler.scale_ = np.ones(3)
        scaler.n_features_in_ = 3

        # Save the scaler for future use
        joblib.dump(scaler, scaler_path)

    yield scaler_path


@pytest.fixture
def setup_test_environment(gcs_settings, mock_scaler_file):
    """Set up the test environment with necessary files."""
    # Copy the mock scaler to the test models directory
    scaler_path = os.path.join(gcs_settings.models_dir, "scaler.pkl")
    shutil.copy(mock_scaler_file, scaler_path)

    # Create a mock PCA model file
    pca_path = os.path.join(gcs_settings.models_dir, "pca_model.pkl")
    mock_pca = MockPCA()
    joblib.dump(mock_pca, pca_path)

    yield gcs_settings.models_dir

    # Cleanup is handled by test_base_dirs fixture


def test_create_and_upload_scaler_to_gcs(
    gcs_settings, mock_storage, setup_test_environment
):
    """
    Test that a scaler is created and uploaded to GCS when cloud storage is enabled.
    """
    # Patch the Google Cloud Storage client
    with patch("google.cloud.storage.Client", mock_storage.client):
        # Call the create_prefitted_scaler function which should upload to GCS
        with patch(
            "anomaly_reaper.processing.anomaly_detection.settings", gcs_settings
        ):
            scaler = create_prefitted_scaler()

            # Check that the scaler was created and returned
            assert scaler is not None

            # Check that the scaler was uploaded to GCS
            expected_blob_name = (
                "anomaly_reaper/scaler.pkl"  # This should be the fixed path we use
            )
            assert mock_storage.has_blob(expected_blob_name), (
                f"Scaler was not uploaded to {expected_blob_name}"
            )

        # Now test that load_model can find the scaler in GCS
        with patch(
            "anomaly_reaper.processing.anomaly_detection.settings", gcs_settings
        ):
            # First, delete the local scaler to force loading from GCS
            scaler_path = os.path.join(gcs_settings.models_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                os.remove(scaler_path)

            # Try loading the model, which should fetch the scaler from GCS
            pca_model, loaded_scaler, threshold = load_model(gcs_settings.models_dir)

            # Check that the scaler was loaded from GCS
            assert loaded_scaler is not None, "Failed to load scaler from GCS"


def test_scaler_upload_with_existing_gcs_scaler(
    gcs_settings, mock_storage, setup_test_environment
):
    """
    Test that the scaler is not re-uploaded if it already exists in GCS.
    """
    # Pre-populate GCS with a scaler
    gcs_scaler_path = "anomaly_reaper/scaler.pkl"
    mock_storage.store_upload(
        gcs_scaler_path, os.path.join(gcs_settings.models_dir, "scaler.pkl")
    )

    # Patch the Google Cloud Storage client
    with patch("google.cloud.storage.Client", mock_storage.client):
        # Store the current upload count
        initial_upload_count = len(mock_storage.uploaded_files)

        # Call the create_prefitted_scaler function with GCS settings
        with patch(
            "anomaly_reaper.processing.anomaly_detection.settings", gcs_settings
        ):
            create_prefitted_scaler()

            # Check that the scaler was not re-uploaded to GCS
            assert len(mock_storage.uploaded_files) == initial_upload_count, (
                "Scaler was re-uploaded when it shouldn't have been"
            )


def test_load_model_gcs_fallback(gcs_settings, mock_storage, setup_test_environment):
    """
    Test that load_model falls back to local files if GCS is unavailable.
    """

    # Patch the Google Cloud Storage client to raise an exception
    def mock_client_with_exception(**kwargs):
        raise Exception("GCS unavailable")

    with patch("google.cloud.storage.Client", side_effect=mock_client_with_exception):
        with patch(
            "anomaly_reaper.processing.anomaly_detection.settings", gcs_settings
        ):
            # Should fall back to local files
            pca_model, scaler, threshold = load_model(gcs_settings.models_dir)

            # Check that we still got valid models
            assert pca_model is not None, "Failed to fall back to local PCA model"
            assert scaler is not None, "Failed to fall back to local scaler"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
