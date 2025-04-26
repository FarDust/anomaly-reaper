"""
Integration tests for the API server interface.

These tests verify that the API endpoints correctly handle requests,
process data, and return appropriate responses.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from anomaly_reaper.interfaces.server import app
from anomaly_reaper.models import ImageRecord


@pytest.fixture
def client():
    """Get test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_db_query():
    """Mock database query operations."""
    with patch("anomaly_reaper.interfaces.server.query_images") as mock:
        mock.return_value = [
            ImageRecord(
                id="test-id-1",
                filename="test1.jpg",
                path="/path/to/test1.jpg",
                reconstruction_error=0.05,
                is_anomaly=False,
                anomaly_score=0.5,
                timestamp="2025-04-25T10:00:00Z",
            ),
            ImageRecord(
                id="test-id-2",
                filename="test2.jpg",
                path="/path/to/test2.jpg",
                reconstruction_error=0.15,
                is_anomaly=True,
                anomaly_score=1.5,
                timestamp="2025-04-25T10:30:00Z",
            ),
        ]
        yield mock


@pytest.fixture
def mock_process_directory(monkeypatch, test_settings):
    """Mock the process_directory function for API."""
    with patch("anomaly_reaper.interfaces.server.process_directory") as mock:
        # Patch the models_dir setting to use test_settings.models_dir
        monkeypatch.setattr(
            "anomaly_reaper.interfaces.server.settings.models_dir",
            test_settings.models_dir,
        )
        mock.return_value = ["image1.jpg", "image2.jpg"]
        yield mock


@pytest.fixture
def mock_db_store():
    """Mock database store operations."""
    with patch("anomaly_reaper.interfaces.server.store_results_in_db") as mock:
        yield mock


@pytest.fixture
def mock_classify_image():
    """Mock the classify_image function."""
    with patch("anomaly_reaper.interfaces.server.classify_image") as mock:
        mock.return_value = {"id": "test-id-1", "user_classification": True}
        yield mock


@pytest.fixture
def mock_get_image():
    """Mock the get_image function."""
    with patch("anomaly_reaper.interfaces.server.get_image_by_id") as mock:
        # Create a mock image byte stream
        mock.return_value = {
            "image_data": b"mock_image_bytes",
            "content_type": "image/jpeg",
        }
        yield mock


class TestAPIServer:
    """Tests for the API server."""

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_list_images(self, client, mock_db_query):
        """Test listing all images."""
        response = client.get("/images/")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "test-id-1"
        assert data[1]["id"] == "test-id-2"

        # Verify mock was called
        mock_db_query.assert_called_once_with(anomalies_only=False)

    def test_list_anomalies(self, client, mock_db_query):
        """Test listing only anomalous images."""
        response = client.get("/images/?anomalies_only=true")
        assert response.status_code == 200

        # Verify mock was called with anomalies_only=True
        mock_db_query.assert_called_once_with(anomalies_only=True)

    def test_get_image_details(self, client, mock_db_query):
        """Test getting details for a specific image."""
        # Set up the mock to return a specific image
        mock_db_query.return_value = [
            ImageRecord(
                id="test-id-1",
                filename="test1.jpg",
                path="/path/to/test1.jpg",
                reconstruction_error=0.05,
                is_anomaly=False,
                anomaly_score=0.5,
            )
        ]

        response = client.get("/images/test-id-1")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "test-id-1"
        assert data["filename"] == "test1.jpg"

        # Verify mock was called correctly
        mock_db_query.assert_called_once()

    def test_get_image_not_found(self, client, mock_db_query):
        """Test getting a non-existent image."""
        # Set up the mock to return an empty list (no image found)
        mock_db_query.return_value = []

        response = client.get("/images/non-existent-id")
        assert response.status_code == 404

    def test_get_image_file(self, client, mock_get_image):
        """Test retrieving the actual image file."""
        response = client.get("/images/test-id-1/file")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

        # Verify mock was called correctly
        mock_get_image.assert_called_once_with("test-id-1")

    def test_get_image_file_not_found(self, client, mock_get_image):
        """Test retrieving a non-existent image file."""
        # Set up the mock to return None (image not found)
        mock_get_image.return_value = None

        response = client.get("/images/non-existent-id/file")
        assert response.status_code == 404

    def test_process_directory(
        self, client, mock_process_directory, mock_db_store, test_settings
    ):
        """Test processing a directory through the API."""
        # Create a test directory path
        test_dir = os.path.join(test_settings.data_dir, "test_images")

        # Make the API call
        response = client.post(
            "/process/", json={"directory_path": test_dir, "anomaly_threshold": 0.7}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["processed_count"] == 2
        assert len(data["processed_files"]) == 2

        # Verify mocks were called correctly
        mock_process_directory.assert_called_once_with(
            test_dir, test_settings.models_dir, 0.7
        )
        mock_db_store.assert_called_once()

    def test_classify_image(self, client, mock_classify_image):
        """Test classifying an image."""
        response = client.post(
            "/images/test-id-1/classify",
            json={"is_anomaly": True, "comment": "This is definitely an anomaly"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-id-1"

        # Verify mock was called correctly
        mock_classify_image.assert_called_once_with(
            "test-id-1", True, "This is definitely an anomaly"
        )

    def test_export_csv(self, client, mock_db_query):
        """Test exporting data as CSV."""
        response = client.get("/export/csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "test-id-1" in response.text
        assert "test-id-2" in response.text

        # Verify mock was called correctly
        mock_db_query.assert_called_once_with(anomalies_only=False)

    def test_export_anomalies_csv(self, client, mock_db_query):
        """Test exporting only anomalies as CSV."""
        response = client.get("/export/csv?anomalies_only=true")
        assert response.status_code == 200

        # Verify mock was called with anomalies_only=True
        mock_db_query.assert_called_once_with(anomalies_only=True)

    def test_statistics(self, client, mock_db_query):
        """Test getting statistics about processed images."""
        # Mock additional function needed for statistics
        with patch("anomaly_reaper.interfaces.server.get_statistics") as mock_stats:
            mock_stats.return_value = {
                "total_images": 100,
                "anomaly_count": 15,
                "anomaly_percentage": 15.0,
                "average_score": 0.65,
            }

            response = client.get("/statistics")
            assert response.status_code == 200

            data = response.json()
            assert data["total_images"] == 100
            assert data["anomaly_count"] == 15
            assert data["anomaly_percentage"] == 15.0
