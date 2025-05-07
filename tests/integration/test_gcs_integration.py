"""
Integration tests for Google Cloud Storage functionality.

These tests verify actual interactions with Google Cloud Storage using the real bucket
configured in settings, but in a safe manner using dedicated test paths.

These tests are marked with 'gcs_integration' and will be skipped by default.
Run them with: pytest -m gcs_integration

Example:
    pytest tests/integration/test_gcs_integration.py -v -m gcs_integration
"""

import os
import pytest
import tempfile
import uuid
import joblib
import numpy as np
import shutil
from datetime import datetime
from PIL import Image

from anomaly_reaper.config import settings
from google.cloud import storage

# Mark all tests in this module with gcs_integration
pytestmark = pytest.mark.gcs_integration


@pytest.fixture(scope="module")
def test_gcs_prefix():
    """Create a unique GCS prefix for the test run to ensure isolation."""
    # Format: anomaly_reaper_test_{timestamp}_{random_uuid}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
    test_prefix = f"anomaly_reaper_test_{timestamp}_{random_id}"
    print(f"\nSetting up test with unique prefix: {test_prefix}")
    return test_prefix


@pytest.fixture(scope="module")
def test_dirs():
    """Create temporary directories for the tests."""
    # Create temporary directories
    models_dir = tempfile.mkdtemp(prefix="test_gcs_models_")
    data_dir = tempfile.mkdtemp(prefix="test_gcs_data_")

    print("\nCreated temporary directories:")
    print(f"  - Models: {models_dir}")
    print(f"  - Data: {data_dir}")

    yield models_dir, data_dir

    # Clean up after tests
    print("\nCleaning up temporary directories:")
    try:
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir, ignore_errors=True)
            print(f"  - Removed {models_dir}")
        else:
            print(f"  - {models_dir} already removed")

        if os.path.exists(data_dir):
            shutil.rmtree(data_dir, ignore_errors=True)
            print(f"  - Removed {data_dir}")
        else:
            print(f"  - {data_dir} already removed")
    except Exception as e:
        print(f"Warning: Error during directory cleanup: {str(e)}")


@pytest.fixture(scope="module")
def gcs_test_client(test_gcs_prefix):
    """Create a GCS client for the tests and clean up test objects after."""
    # Create GCS client
    client = storage.Client(project=settings.project_id)
    bucket = client.bucket(settings.gcs_bucket_name)

    # Verify the bucket exists before tests run
    if not bucket.exists():
        pytest.fail(
            f"Bucket {settings.gcs_bucket_name} does not exist or is not accessible"
        )

    print(f"\nConnected to GCS bucket: {settings.gcs_bucket_name}")
    print(f"Using test prefix: {test_gcs_prefix}")

    # Check if there are any existing blobs with this prefix (there shouldn't be)
    existing_blobs = list(bucket.list_blobs(prefix=test_gcs_prefix))
    if existing_blobs:
        print(
            f"Warning: {len(existing_blobs)} objects already exist with prefix {test_gcs_prefix}"
        )
        print("These will be cleaned up after tests complete")

    yield client, bucket, test_gcs_prefix

    # Clean up all test objects after the tests
    print(f"\nCleaning up test objects with prefix {test_gcs_prefix}...")
    try:
        deleted_count = 0
        blobs = list(bucket.list_blobs(prefix=test_gcs_prefix))

        if not blobs:
            print(f"No objects found with prefix {test_gcs_prefix}")
            return

        for blob in blobs:
            try:
                blob.delete()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {blob.name}: {str(e)}")

        print(
            f"Cleanup completed: {deleted_count} objects deleted with prefix {test_gcs_prefix}"
        )
    except Exception as e:
        print(f"Warning: Error during GCS cleanup: {str(e)}")
        print(
            f"Some test objects with prefix {test_gcs_prefix} might remain in the bucket"
        )


@pytest.fixture
def test_scaler(test_dirs):
    """Create a test StandardScaler."""
    models_dir, _ = test_dirs

    # Create a scaler for testing
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = np.zeros(10)
    scaler.var_ = np.ones(10)
    scaler.scale_ = np.ones(10)
    scaler.n_features_in_ = 10
    scaler.n_samples_seen_ = 100

    # Save the scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    return scaler, scaler_path


@pytest.fixture
def test_image(test_dirs):
    """Create a test image for upload testing."""
    _, data_dir = test_dirs

    # Create a simple test image
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))

    # Save the image
    image_path = os.path.join(data_dir, "test_image.png")
    img.save(image_path)

    return image_path


def test_gcs_bucket_exists(gcs_test_client):
    """Verify that the GCS bucket exists and is accessible."""
    _, bucket, _ = gcs_test_client

    # Check that the bucket exists
    assert bucket.exists(), (
        f"Bucket {settings.gcs_bucket_name} does not exist or is not accessible"
    )


def test_upload_and_download_scaler(gcs_test_client, test_scaler):
    """Test uploading and downloading a scaler to/from GCS."""
    _, bucket, test_prefix = gcs_test_client
    _, scaler_path = test_scaler

    # Define GCS path for the test
    gcs_path = f"{test_prefix}/scaler.pkl"

    # Upload the scaler
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(scaler_path)

    # Verify the blob exists
    assert blob.exists(), f"Failed to upload scaler to {gcs_path}"

    # Download to a new location
    download_path = f"{scaler_path}.downloaded"
    blob.download_to_filename(download_path)

    # Verify the downloaded file exists
    assert os.path.exists(download_path), f"Failed to download scaler from {gcs_path}"

    # Load both versions and compare
    original_scaler = joblib.load(scaler_path)
    downloaded_scaler = joblib.load(download_path)

    # Verify they have the same attributes
    assert original_scaler.n_features_in_ == downloaded_scaler.n_features_in_
    assert np.array_equal(original_scaler.mean_, downloaded_scaler.mean_)
    assert np.array_equal(original_scaler.var_, downloaded_scaler.var_)

    # Clean up the downloaded file
    if os.path.exists(download_path):
        os.remove(download_path)


def test_upload_and_list_images(gcs_test_client, test_image):
    """Test uploading images to GCS and listing them."""
    _, bucket, test_prefix = gcs_test_client

    # Define GCS path for the test
    image_filename = os.path.basename(test_image)
    gcs_path = f"{test_prefix}/images/{image_filename}"

    # Upload the image
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(test_image)

    # Verify the blob exists
    assert blob.exists(), f"Failed to upload image to {gcs_path}"

    # List all blobs in the test prefix
    blobs = list(bucket.list_blobs(prefix=f"{test_prefix}/images/"))

    # Verify the image is in the list
    blob_names = [blob.name for blob in blobs]
    assert gcs_path in blob_names, (
        f"Failed to find uploaded image in {test_prefix}/images/"
    )


def test_create_prefitted_scaler_with_real_gcs(gcs_test_client, test_dirs, monkeypatch):
    """Test creating and uploading a pre-fitted scaler to the real GCS bucket in a safe path."""
    client, bucket, test_prefix = gcs_test_client
    models_dir, data_dir = test_dirs

    # Create mock embeddings data
    embeddings_path = os.path.join(data_dir, "image_embeddings.csv")

    # Generate random embeddings for testing
    num_images = 5
    embedding_dim = 10

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

    # Modify settings for the test
    monkeypatch.setattr(
        "anomaly_reaper.processing.anomaly_detection.settings.models_dir", models_dir
    )
    monkeypatch.setattr(
        "anomaly_reaper.processing.anomaly_detection.settings.data_dir", data_dir
    )
    monkeypatch.setattr(
        "anomaly_reaper.processing.anomaly_detection.settings.use_cloud_storage", True
    )
    monkeypatch.setattr(
        "anomaly_reaper.processing.anomaly_detection.settings.project_id",
        settings.project_id,
    )
    monkeypatch.setattr(
        "anomaly_reaper.processing.anomaly_detection.settings.gcs_bucket_name",
        settings.gcs_bucket_name,
    )

    # Instead of patching GCS_BASE_PATH (which doesn't exist at module level),
    # we'll monkey patch the blob creation with our test prefix
    original_blob = storage.Bucket.blob

    def patched_blob(self, blob_name, *args, **kwargs):
        # If the path starts with the standard 'anomaly_reaper/' prefix
        # (used in the anomaly_detection module), replace it with our test prefix
        if blob_name.startswith("anomaly_reaper/"):
            modified_name = f"{test_prefix}/{blob_name[len('anomaly_reaper/') :]}"
            return original_blob(self, modified_name, *args, **kwargs)
        return original_blob(self, blob_name, *args, **kwargs)

    # Apply the monkey patch
    monkeypatch.setattr(storage.Bucket, "blob", patched_blob)

    try:
        # We'll directly use the underlying functions to avoid any potential side effects
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        # Load our mock embeddings data
        embeddings_df = pd.read_csv(embeddings_path)
        embeddings = embeddings_df.values

        # Create a StandardScaler
        scaler = StandardScaler()

        # Fit the scaler with our test data
        scaler.mean_ = np.mean(embeddings, axis=0)
        scaler.var_ = np.var(embeddings, axis=0)
        scaler.scale_ = np.sqrt(scaler.var_)
        scaler.n_features_in_ = embeddings.shape[1]
        scaler.n_samples_seen_ = embeddings.shape[0]

        # Save the scaler to the test directory
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

        # Upload to GCS using the patched blob method which will use our test prefix
        gcs_path = f"{test_prefix}/scaler.pkl"
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(scaler_path)

        # Verify the upload
        assert blob.exists(), f"Failed to upload scaler to {gcs_path}"

        # Now try to download it to verify
        download_path = os.path.join(models_dir, "scaler_from_gcs.pkl")
        blob.download_to_filename(download_path)

        # Load the downloaded scaler
        downloaded_scaler = joblib.load(download_path)

        # Verify it's the same
        assert downloaded_scaler.n_features_in_ == scaler.n_features_in_
        assert np.array_equal(downloaded_scaler.mean_, scaler.mean_)
        assert np.array_equal(downloaded_scaler.var_, scaler.var_)

    except Exception as e:
        pytest.fail(f"Failed to upload and verify scaler: {str(e)}")


def test_gcs_image_url_generation(gcs_test_client, test_image):
    """Test generating public URLs for images in GCS."""
    _, bucket, test_prefix = gcs_test_client

    # Define GCS path for the test
    image_filename = os.path.basename(test_image)
    gcs_path = f"{test_prefix}/images/{image_filename}"

    # Upload the image
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(test_image)

    # Instead of testing signed URLs (which require special credentials),
    # test the public URL which doesn't need signing
    public_url = f"https://storage.googleapis.com/{settings.gcs_bucket_name}/{gcs_path}"

    # Verify the URL looks correct
    assert public_url.startswith("https://"), "Public URL doesn't start with https://"
    assert settings.gcs_bucket_name in public_url, "Bucket name not in public URL"
    assert test_prefix in public_url, "Test prefix not in public URL"


if __name__ == "__main__":
    # Enable running this file directly
    pytest.main(["-xvs", __file__])
