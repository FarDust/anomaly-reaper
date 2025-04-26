"""
Shared fixtures and configuration for the test suite.
"""

import os
import pytest
import tempfile
import numpy as np
import joblib
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from anomaly_reaper.models import Base
from anomaly_reaper.config import Settings
from anomaly_reaper.interfaces.server import app


# Define mock classes at module level for proper pickling
class MockPCA:
    """Mock PCA model for testing."""

    def transform(self, X):
        # Return a dummy transformed matrix with reduced dimensions
        return np.random.random((X.shape[0], 2))

    def inverse_transform(self, X_transformed):
        # Return a dummy reconstructed matrix
        return np.random.random((X_transformed.shape[0], 10))


class MockScaler:
    """Mock StandardScaler for testing."""

    def transform(self, X):
        # Return a standardized matrix of the same shape
        return X  # Simply return the input for testing


@pytest.fixture
def test_settings():
    """Create test settings with temporary directories and test database."""
    # Create temporary directories for models and uploads
    with (
        tempfile.TemporaryDirectory() as models_dir,
        tempfile.TemporaryDirectory() as uploads_dir,
    ):
        # Create test settings with temporary directories
        test_settings = Settings(
            models_dir=models_dir,
            uploads_dir=uploads_dir,
            db_url="sqlite:///./test_anomaly_reaper.db",
            anomaly_threshold=0.05,
        )

        # Create the directories
        test_settings.create_directories()

        yield test_settings


@pytest.fixture
def test_db(test_settings):
    """Create a test database with tables."""
    # Create database engine and tables
    engine = create_engine(test_settings.db_url)
    Base.metadata.create_all(engine)

    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Return engine and session
    yield engine, TestingSessionLocal

    # Clean up: drop tables after tests
    Base.metadata.drop_all(engine)

    # Remove the database file
    if test_settings.db_url.startswith("sqlite:///"):
        db_path = test_settings.db_url.replace("sqlite:///", "")
        if os.path.exists(db_path) and db_path != ":memory:":
            os.unlink(db_path)


@pytest.fixture
def db_session(test_db):
    """Provide a test database session."""
    engine, TestingSessionLocal = test_db
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def mock_pca_model(test_settings):
    """Create a mock PCA model and save it to the test models directory."""
    # Create instances of the mock classes
    mock_pca = MockPCA()
    mock_scaler = MockScaler()

    # Save the mock models to the test models directory
    model_path = os.path.join(test_settings.models_dir, "pca_model.pkl")
    scaler_path = os.path.join(test_settings.models_dir, "scaler.pkl")

    joblib.dump(mock_pca, model_path)
    joblib.dump(mock_scaler, scaler_path)

    # Create a config file with threshold
    config_path = os.path.join(test_settings.models_dir, "config.json")
    with open(config_path, "w") as f:
        f.write('{"threshold": 0.05}')

    yield mock_pca, mock_scaler


@pytest.fixture
def test_image(test_settings):
    """Create a test image and save it to a temporary file."""
    # Create a simple test image
    img = Image.new("RGB", (100, 100), color="white")

    # Save the image to the uploads directory
    img_path = os.path.join(test_settings.uploads_dir, "test_image.jpg")
    img.save(img_path)

    yield img_path

    # Clean up
    if os.path.exists(img_path):
        os.remove(img_path)


@pytest.fixture
def client(test_settings, mock_pca_model, monkeypatch):
    """Create a FastAPI TestClient with mocked settings."""
    # Monkeypatch the app to use test settings
    from anomaly_reaper.interfaces.server import settings as app_settings

    # Override the app settings with test settings
    for key, value in test_settings.model_dump().items():
        monkeypatch.setattr(app_settings, key, value)

    # Override the get_db dependency
    def override_get_db():
        engine = create_engine(test_settings.db_url)
        TestingSessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    from anomaly_reaper.interfaces.server import get_db

    app.dependency_overrides[get_db] = override_get_db

    # Create test client
    client = TestClient(app)

    # Reset dependency overrides after tests
    yield client

    app.dependency_overrides = {}
