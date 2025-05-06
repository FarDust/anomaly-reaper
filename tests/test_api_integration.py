"""
Integration tests for the API server interface.

These tests verify that the API endpoints correctly handle requests,
process data, and return appropriate responses.
"""

import pytest
import uuid
import os
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import patch, MagicMock, mock_open

from anomaly_reaper.interfaces.api import app
from anomaly_reaper.infrastructure.database.models import (
    ImageRecord,
    Classification,
    engine,
    create_tables,
)


@pytest.fixture(scope="module")
def setup_test_db():
    """Set up test database tables and clean up afterward.

    This fixture ensures tables exist before tests and cleans
    up test data after all tests in the module have run.
    """
    # Create tables before tests
    create_tables()

    # Provide test session
    yield

    # Clean up any test data after all tests
    with Session(engine) as cleanup_session:
        # Delete any test records (those with id containing 'test-')
        cleanup_session.query(Classification).filter(
            Classification.id.like("%test-%")
        ).delete()
        cleanup_session.query(ImageRecord).filter(
            ImageRecord.id.like("%test-%")
        ).delete()
        # Commit the cleanup
        cleanup_session.commit()


@pytest.fixture
def test_db_session():
    """Provide a test database session with automatic rollback."""
    with Session(engine) as session:
        yield session
        # Rollback changes if the test didn't commit them
        session.rollback()


@pytest.fixture
def test_image_record(test_db_session, setup_test_db):
    """Create a test image record for testing.

    Returns the record's ID for tests to use, and automatically
    cleans up afterward.
    """
    # Create a unique test ID prefixed with test- for easy identification
    test_id = f"test-{uuid.uuid4()}"
    test_datetime = datetime.now(timezone.utc)

    # Create a test record in the database
    test_image = ImageRecord(
        id=test_id,
        filename="test_image.jpg",
        path="/path/to/test_image.jpg",
        reconstruction_error=0.05,
        is_anomaly=False,
        anomaly_score=0.5,
        processed_at=test_datetime,
    )
    test_db_session.add(test_image)
    test_db_session.commit()

    # Yield the ID to the test
    yield test_id


@pytest.fixture
def client():
    """Get test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_db_query():
    """Mock database query operations."""
    with patch("anomaly_reaper.interfaces.api.query_images") as mock:
        mock.return_value = (
            [
                ImageRecord(
                    id="test-id-1",
                    filename="test1.jpg",
                    path="/path/to/test1.jpg",
                    reconstruction_error=0.05,
                    is_anomaly=False,
                    anomaly_score=0.5,
                    processed_at="2025-04-25T10:00:00Z",
                ),
                ImageRecord(
                    id="test-id-2",
                    filename="test2.jpg",
                    path="/path/to/test2.jpg",
                    reconstruction_error=0.15,
                    is_anomaly=True,
                    anomaly_score=1.5,
                    processed_at="2025-04-25T10:30:00Z",
                ),
            ],
            2,  # total count
            1,  # total pages
        )
        yield mock


@pytest.fixture
def mock_process_directory(monkeypatch):
    """Mock the process_directory function for API."""
    with patch("anomaly_reaper.interfaces.api.process_directory") as mock:
        # Patch the models_dir setting directly
        monkeypatch.setattr(
            "anomaly_reaper.interfaces.api.settings.models_dir",
            "/mock/models/dir",
        )
        mock.return_value = ["image1.jpg", "image2.jpg"]
        yield mock


@pytest.fixture
def mock_db_store():
    """Mock database store operations."""
    with patch("anomaly_reaper.interfaces.api.store_results_in_db") as mock:
        yield mock


@pytest.fixture
def mock_classify_image():
    """Mock the classify_image function."""
    with patch("anomaly_reaper.interfaces.api.classify_image") as mock:
        mock.return_value = {"id": "test-id-1", "user_classification": True}
        yield mock


@pytest.fixture
def mock_get_image():
    """Mock the get_image function."""
    with patch("anomaly_reaper.interfaces.api.get_image_by_id") as mock:
        # Create a mock image byte stream
        mock.return_value = {
            "image_data": b"mock_image_bytes",
            "content_type": "image/jpeg",
        }
        yield mock


@pytest.fixture
def mock_get_visualization():
    """Mock the visualization generation function."""
    # First, mock the get_image_by_id function
    with patch("anomaly_reaper.interfaces.api.get_image_by_id") as mock_img:
        # Mock image data
        mock_img.return_value = {
            "image_data": b"mock_image_bytes",
            "content_type": "image/jpeg",
        }

        # Mock PIL.Image.open
        with patch("PIL.Image.open") as mock_pil:
            mock_image = MagicMock()
            mock_image.size = (100, 100)  # width, height
            mock_pil.return_value = mock_image

            # Mock encode_image_to_base64
            with patch(
                "anomaly_reaper.utils.visualization.encode_image_to_base64"
            ) as mock_encode:
                mock_encode.return_value = "base64encodedstring"

                # Mock create_anomaly_heatmap
                with patch(
                    "anomaly_reaper.utils.visualization.create_anomaly_heatmap"
                ) as mock_heatmap:
                    mock_heatmap.return_value = mock_image

                    # Mock calculate_anomaly_bbox
                    with patch(
                        "anomaly_reaper.utils.visualization.calculate_anomaly_bbox"
                    ) as mock_bbox:
                        mock_bbox.return_value = [10, 10, 90, 90]  # x, y, width, height

                        # Mock create_bounding_box_visualization
                        with patch(
                            "anomaly_reaper.utils.visualization.create_bounding_box_visualization"
                        ) as mock_bb:
                            mock_bb.return_value = mock_image
                            yield mock_img


@pytest.fixture
def mock_dashboard_stats():
    """Mock dashboard statistics."""
    with patch("anomaly_reaper.interfaces.api.get_dashboard_statistics") as mock:
        mock.return_value = {
            "total_images": 100,
            "total_anomalies": 15,
            "user_confirmed_anomalies": 10,
            "unclassified_anomalies": 5,
            "false_positives": 3,
            "false_negatives": 2,
            "recent_activity": [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "image_id": "test-id-1",
                    "action": "classification",
                    "is_anomaly": True,
                }
            ],
        }
        yield mock


@pytest.fixture
def mock_classification_history():
    """Mock classification history."""
    with patch("anomaly_reaper.interfaces.api.Classification") as mock:
        mock.return_value = [
            {
                "id": "class-1",
                "image_id": "test-id-1",
                "user_classification": True,
                "comment": "Test comment",
                "timestamp": datetime.now(timezone.utc),
            }
        ]
        yield mock


@pytest.fixture
def mock_similar_images():
    """Mock similar images search."""
    with patch("anomaly_reaper.interfaces.api.ImageRecord") as mock:
        yield mock


@pytest.fixture
def mock_sync_data():
    """Mock sync anomaly data function."""
    with patch("anomaly_reaper.interfaces.api.pd.read_csv") as mock_pd:
        # Mock the pandas dataframe
        import pandas as pd

        # Create a more complete mock dataframe that matches what the API expects
        mock_df = pd.DataFrame(
            {
                "image_path": ["path/to/image1.jpg", "path/to/image2.jpg"],
                "reconstruction_error": [0.05, 0.15],
                "is_anomaly": [False, True],
                "anomaly_score": [0.3, 0.8],
            }
        )
        mock_pd.return_value = mock_df

        # Mock file operations and path handling
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            # Mock file copying
            with patch("shutil.copy") as mock_copy:
                with patch(
                    "anomaly_reaper.interfaces.api.store_results_in_db"
                ) as mock_store:
                    mock_store.return_value = 2  # 2 records processed
                    yield mock_pd


@pytest.fixture
def mock_pca_visualize():
    """Mock PCA visualization."""
    with patch("anomaly_reaper.interfaces.api.generate_pca_projection") as mock:
        mock.return_value = (
            "base64encodedstring",
            [{"x": 1, "y": 2, "is_anomaly": True}],
            0.1,
        )
        yield mock


@pytest.fixture
def test_image_path():
    """Create a test image fixture with a known path for testing."""
    # Use the test image we created earlier
    test_img_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests",
        "fixtures",
        "images",
        "test_image.png",
    )
    # Verify the image exists
    assert os.path.exists(test_img_path), f"Test image not found at {test_img_path}"
    return test_img_path


class TestAPIServer:
    """Tests for the API server."""

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        response_data = response.json()
        # Check only that the required fields are present with correct types
        assert response_data["status"] == "ok"
        assert "version" in response_data
        assert "timestamp" in response_data

    def test_list_images(self, client, mock_db_query):
        """Test listing all images."""
        response = client.get("/images/")
        assert response.status_code == 200

        data = response.json()
        # Check paginated response structure
        assert "results" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data

        # Check the image results
        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == "test-id-1"
        assert data["results"][1]["id"] == "test-id-2"
        assert data["total_count"] == 2
        assert data["total_pages"] == 1

        # Verify mock was called with correct parameters
        mock_db_query.assert_called_once_with(
            anomalies_only=False,
            page=1,
            page_size=9,
            sort_by="processed_at",
            sort_order="desc",
        )

    def test_list_anomalies(self, client, mock_db_query):
        """Test listing only anomalous images."""
        response = client.get("/images/?anomalies_only=true")
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "total_count" in data

        # Verify mock was called with correct parameters
        mock_db_query.assert_called_once_with(
            anomalies_only=True,
            page=1,
            page_size=9,
            sort_by="processed_at",
            sort_order="desc",
        )

    def test_get_image_details(self, client, test_image_record):
        """Test getting details for a specific image."""
        # test_image_record fixture creates a record and returns its ID

        # Mock the query_images function for this particular test
        with patch("anomaly_reaper.interfaces.api.query_images") as mock_query:
            # Create a mock image record to return
            mock_image = ImageRecord(
                id=test_image_record,
                filename="test_image.jpg",
                path="/path/to/test_image.jpg",
                reconstruction_error=0.05,
                is_anomaly=False,
                anomaly_score=0.5,
                processed_at=datetime.now(timezone.utc),
            )
            mock_query.return_value = [mock_image]

            # Test retrieving the image
            response = client.get(f"/images/{test_image_record}")
            assert response.status_code == 200

            data = response.json()
            assert data["id"] == test_image_record
            assert data["filename"] == "test_image.jpg"
            assert (
                "timestamp" in data
            )  # Verify the API converts processed_at to timestamp

    def test_get_image_not_found(self, client, mock_db_query):
        """Test getting a non-existent image."""
        # Set up the mock to return an empty list (no image found)
        mock_db_query.return_value = []

        response = client.get("/images/non-existent-id")
        assert response.status_code == 404

    def test_get_image_file(self, client, mock_get_image):
        """Test retrieving the actual image file."""
        # Use patch.object instead of mocking the entire ImageRecord class
        with patch("sqlalchemy.orm.Session") as mock_session:
            # Create a mock image record instance
            mock_record = MagicMock()
            mock_record.id = "test-id-1"
            mock_record.path = "/path/to/image.jpg"

            # Set up session mock to return our mock record
            mock_session_instance = MagicMock()
            # Instead of mocking ImageRecord, mock the query chain directly
            mock_query = MagicMock()
            mock_query.filter.return_value.first.return_value = mock_record
            mock_session_instance.query.return_value = mock_query
            mock_session.return_value.__enter__.return_value = mock_session_instance

            # Mock file operations
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = True

                with patch(
                    "builtins.open", mock_open(read_data=b"mock_binary_data")
                ) as mock_open_file:
                    # Execute test
                    response = client.get("/images/test-id-1/file")
                    assert response.status_code == 200
                    assert response.content == b"mock_binary_data"

    def test_get_image_file_not_found(self, client, mock_get_image):
        """Test retrieving a non-existent image file."""
        # Set up the mock to return None (image not found)
        mock_get_image.return_value = None

        response = client.get("/images/non-existent-id/file")
        assert response.status_code == 404

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

    def test_statistics(self, client, mock_db_query):
        """Test getting statistics about processed images."""
        # Mock additional function needed for statistics
        with patch("anomaly_reaper.interfaces.api.get_statistics") as mock_stats:
            # Update the mock to return data matching the StatisticsResponse schema
            mock_stats.return_value = {
                "total_images": 100,
                "anomalies_detected": 15,
                "classified_images": 20,
                "average_anomaly_score": 0.65,
                "storage_type": "Google Cloud Storage",
                "storage_location": "test-bucket-name",  # Changed from real bucket name to mock value
            }

            response = client.get("/statistics")
            assert response.status_code == 200

            data = response.json()
            assert data["total_images"] == 100
            assert data["anomalies_detected"] == 15
            assert data["average_anomaly_score"] == 0.65

    def test_get_image_visualization(self, client, mock_get_visualization):
        """Test getting image visualizations."""
        # Mock database query to find the image record
        with patch("sqlalchemy.orm.Session") as mock_session:
            # Create a mock image record instance
            mock_record = MagicMock()
            mock_record.id = "test-id-1"
            mock_record.path = "/path/to/image.jpg"
            mock_record.reconstruction_error = 0.05
            mock_record.is_anomaly = True
            mock_record.anomaly_score = 0.8

            # Set up session mock to return our mock record
            mock_session_instance = MagicMock()
            # Mock the query chain directly instead of patching ImageRecord
            mock_query = MagicMock()
            mock_query.filter.return_value.first.return_value = mock_record
            mock_session_instance.query.return_value = mock_query
            mock_session.return_value.__enter__.return_value = mock_session_instance

            # Execute test
            response = client.get("/images/test-id-1/visualization")
            assert response.status_code == 200

            # Check response structure
            data = response.json()
            assert data["image_id"] == "test-id-1"
            assert "visualizations" in data
            assert len(data["visualizations"]) == 3  # Original, heatmap, bbox
            assert "reconstruction_error" in data
            assert "is_anomaly" in data
            assert "anomaly_score" in data

            # Check visualization types
            viz_types = [v["visualization_type"] for v in data["visualizations"]]
            assert "ORIGINAL" in viz_types
            assert "ANOMALY_HEATMAP" in viz_types
            assert "BOUNDING_BOX" in viz_types

    def test_get_dashboard_statistics(self, client):
        """Test getting dashboard statistics."""
        with patch("anomaly_reaper.interfaces.api.get_db") as mock_get_db:
            # Setup more explicit mock session behavior
            mock_session = MagicMock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Mock query results for total images
            total_images_query = MagicMock()
            total_images_query.scalar.return_value = 100

            # Mock query results for anomalies
            anomalies_query = MagicMock()
            anomalies_query.scalar.return_value = 15

            # Mock query results for classifications
            classifications_query = MagicMock()
            classifications_result = [
                # Format: (Classification, ImageRecord.is_anomaly)
                (MagicMock(user_classification=True), True),  # True positive
                (MagicMock(user_classification=False), True),  # False negative
                (MagicMock(user_classification=True), False),  # False positive
            ]
            classifications_query.all.return_value = classifications_result

            # Mock query results for unclassified anomalies
            unclassified_query = MagicMock()
            unclassified_query.count.return_value = 5

            # Mock query results for recent activity
            recent_query = MagicMock()
            recent_query.all.return_value = [
                MagicMock(
                    timestamp=datetime.now(timezone.utc),
                    image_id="test-id-1",
                    user_classification=True,
                    comment="Test comment",
                )
            ]

            # Set up query chain side effects
            mock_session.query.side_effect = [
                total_images_query,  # First query: total images
                anomalies_query,  # Second query: anomalies count
                classifications_query,  # Third query: classifications
                unclassified_query,  # Fourth query: unclassified anomalies
                recent_query,  # Fifth query: recent activity
            ]

            # Test the endpoint
            response = client.get("/statistics/dashboard")
            assert response.status_code == 200

            # Check response structure
            data = response.json()
            assert "total_images" in data
            assert "total_anomalies" in data
            assert "user_confirmed_anomalies" in data
            assert "unclassified_anomalies" in data
            assert "false_positives" in data
            assert "false_negatives" in data
            assert "recent_activity" in data

    def test_search_images(self, client):
        """Test advanced search for images with filters."""
        with patch("anomaly_reaper.interfaces.api.get_db") as mock_get_db:
            # Mock session behavior
            mock_session = MagicMock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Mock query building and result
            mock_query = MagicMock()
            mock_session.query.return_value = mock_query
            mock_query.filter.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.count.return_value = 2
            mock_query.order_by.return_value = mock_query

            # Mock the query results - use a simpler approach that's more predictable
            mock_results = [
                MagicMock(
                    id="test-id-1",
                    filename="test1.jpg",
                    path="/path/to/test1.jpg",
                    reconstruction_error=0.05,
                    is_anomaly=False,
                    anomaly_score=0.5,
                    processed_at=datetime.now(timezone.utc),
                ),
                MagicMock(
                    id="test-id-2",
                    filename="test2.jpg",
                    path="/path/to/test2.jpg",
                    reconstruction_error=0.15,
                    is_anomaly=True,
                    anomaly_score=0.8,
                    processed_at=datetime.now(timezone.utc),
                ),
            ]
            mock_query.offset.return_value.limit.return_value.all.return_value = (
                mock_results
            )

            # Patch the image_to_dict function to convert our mocks to dicts
            with patch("anomaly_reaper.interfaces.api.image_to_dict") as mock_to_dict:
                # Make the mock return predictable dict results that match our test expectations
                mock_to_dict.side_effect = [
                    {
                        "id": "test-id-1",
                        "filename": "test1.jpg",
                        "anomaly_score": 0.5,
                        "is_anomaly": False,
                        "reconstruction_error": 0.05,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    {
                        "id": "test-id-2",
                        "filename": "test2.jpg",
                        "anomaly_score": 0.8,
                        "is_anomaly": True,
                        "reconstruction_error": 0.15,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                ]

                # Test with various filter parameters
                response = client.get(
                    "/images/search?is_anomaly=true&min_score=0.5&page=1&page_size=10&sort_by=anomaly_score&sort_order=desc"
                )
                assert response.status_code == 200

                # Check response structure
                data = response.json()
                assert "results" in data
                assert "total_count" in data
                assert "page" in data
                assert "page_size" in data
                assert "total_pages" in data
                assert (
                    len(data["results"]) == 2
                )  # We expect exactly 2 results based on our mock

    def test_classification_history(self, client, test_image_record):
        """Test getting classification history for an image."""
        with patch("anomaly_reaper.interfaces.api.get_db") as mock_get_db:
            # Mock session behavior
            mock_session = MagicMock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Mock image record
            mock_image = MagicMock()
            mock_image.id = test_image_record

            # Mock query to find image using session.query directly
            mock_image_query = MagicMock()
            mock_image_query.filter.return_value.first.return_value = mock_image

            # Mock classification records
            mock_classifications = [
                MagicMock(
                    id=f"class-{i}",
                    image_id=test_image_record,
                    user_classification=(i % 2 == 0),  # Alternate True/False
                    comment=f"Test comment {i}",
                    timestamp=datetime.now(timezone.utc),
                )
                for i in range(1, 4)  # 3 classifications
            ]

            # Mock query for classifications
            mock_class_query = MagicMock()
            mock_class_query.filter.return_value.order_by.return_value.all.return_value = mock_classifications

            # Set up query side effects for the session
            mock_session.query.side_effect = [mock_image_query, mock_class_query]

            # Test the endpoint
            response = client.get(f"/images/{test_image_record}/classifications")
            assert response.status_code == 200

            # Check response structure
            data = response.json()
            assert data["image_id"] == test_image_record
            assert "classifications" in data
            assert len(data["classifications"]) == 3

    def test_batch_update_classifications(self, client):
        """Test batch updating classifications for multiple images."""
        with patch("anomaly_reaper.interfaces.api.classify_image") as mock_classify:
            # Mock successful classification for first image
            # Mock failed classification for second image
            mock_classify.side_effect = [
                {"id": "test-id-1", "user_classification": True},
                Exception(
                    "Failed to classify image"
                ),  # This will make second image fail
            ]

            # Test endpoint with batch request
            request_data = {
                "image_ids": ["test-id-1", "test-id-2"],
                "is_anomaly": True,
                "comment": "Batch classification test",
            }

            response = client.patch("/images/classifications", json=request_data)
            assert response.status_code == 200

            # Check response structure
            data = response.json()
            assert data["total"] == 2
            assert data["successful"] == 1
            assert data["failed"] == 1
            assert "test-id-2" in data["failed_ids"]

            # Verify mock was called correctly
            assert mock_classify.call_count == 2
            mock_classify.assert_any_call(
                "test-id-1", True, "Batch classification test"
            )
            mock_classify.assert_any_call(
                "test-id-2", True, "Batch classification test"
            )

    def test_similar_images(self, client, test_image_record):
        """Test finding similar anomalous images."""
        with patch("anomaly_reaper.interfaces.api.get_db") as mock_get_db:
            # Mock database session
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session

            # Mock reference image
            mock_ref_img = MagicMock()
            mock_ref_img.id = test_image_record
            mock_ref_img.reconstruction_error = 0.1
            # Make sure anomaly_score is within expected range (0-1)
            mock_ref_img.anomaly_score = 0.75
            mock_ref_img.is_anomaly = True

            # Mock query to find reference image
            mock_session.query.return_value.filter.return_value.first.return_value = (
                mock_ref_img
            )

            # Mock potential matching images
            mock_matches = [
                MagicMock(
                    id=f"test-similar-{i}",
                    filename=f"similar{i}.jpg",
                    path="/path/to/similar{i}.jpg",
                    reconstruction_error=0.1 + (i * 0.01),  # Slightly different errors
                    is_anomaly=True,
                    # Make sure scores are within expected range (0-1)
                    anomaly_score=0.75 - (i * 0.1),  # Slightly different scores
                    processed_at=datetime.now(timezone.utc),
                )
                for i in range(1, 4)  # 3 similar images
            ]

            # Setup another query mock for the second query (finding similar images)
            mock_session.query.return_value.filter.return_value.all.return_value = (
                mock_matches
            )

            # Test the endpoint
            response = client.get(
                f"/images/{test_image_record}/similarities?limit=5&min_score=0.5"
            )
            assert response.status_code == 200

            # Check response structure
            data = response.json()
            assert data["reference_image_id"] == test_image_record
            assert "similar_images" in data
            assert (
                len(data["similar_images"]) > 0
            )  # Should have at least one similar image

            # Verify each similar result has image details and a similarity score
            for similar in data["similar_images"]:
                assert "image" in similar
                assert "similarity_score" in similar
                assert (
                    0 <= similar["similarity_score"] <= 1
                )  # Score should be normalized to 0-1

    def test_export_images(self, client):
        """Test exporting images data in different formats."""
        with patch("anomaly_reaper.interfaces.api.pd.DataFrame") as mock_df:
            # Mock file operations
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp_instance = MagicMock()
                mock_temp_instance.name = "/tmp/mock_export.csv"
                mock_temp.return_value.__enter__.return_value = mock_temp_instance

                # Mock query for images
                with patch("anomaly_reaper.interfaces.api.ImageRecord") as mock_record:
                    # Mock query building and results
                    mock_query = MagicMock()
                    mock_query.filter.return_value = mock_query
                    mock_query.order_by.return_value = mock_query

                    # Mock result records
                    mock_records = [
                        MagicMock(
                            id=f"test-id-{i}",
                            filename=f"test{i}.jpg",
                            path=f"/path/to/test{i}.jpg",
                            reconstruction_error=0.05 + (i * 0.01),
                            is_anomaly=(i % 2 == 0),
                            anomaly_score=0.5 + (i * 0.1),
                            processed_at=datetime.now(timezone.utc),
                        )
                        for i in range(1, 4)  # 3 records
                    ]
                    mock_query.all.return_value = mock_records

                    # Set up the session mock
                    with patch("sqlalchemy.orm.Session") as mock_session:
                        mock_session_instance = MagicMock()
                        mock_session_instance.query.return_value = mock_query

                        mock_session.return_value.__enter__.return_value = (
                            mock_session_instance
                        )

                        # Test CSV export
                        response = client.get("/images/export?format=CSV")
                        assert response.status_code == 200
                        assert response.headers["content-type"] == "text/csv"
                        assert (
                            "anomaly_data_export_"
                            in response.headers["content-disposition"]
                        )
                        assert ".csv" in response.headers["content-disposition"]

                        # Test JSON export
                        response = client.get("/images/export?format=JSON")
                        assert response.status_code == 200
                        assert response.headers["content-type"] == "application/json"
                        assert (
                            "anomaly_data_export_"
                            in response.headers["content-disposition"]
                        )
                        assert ".json" in response.headers["content-disposition"]

                        # Test with filters
                        response = client.get(
                            "/images/export?format=JSON&is_anomaly=true&min_score=0.8"
                        )
                        assert response.status_code == 200

    def test_sync_anomaly_data(self, client):
        """Test synchronizing anomaly data from external sources."""
        # Mock pandas read_csv
        with patch("pandas.read_csv") as mock_read_csv:
            import pandas as pd

            # Create a mock dataframe with test data and proper data types
            mock_df = pd.DataFrame(
                {
                    "image_path": ["path/to/image1.jpg", "path/to/image2.jpg"],
                    "reconstruction_error": [0.05, 0.15],
                    "is_anomaly": [False, True],
                    # Make sure anomaly_score is within expected range (0-1)
                    "anomaly_score": [0.3, 0.8],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock file operations
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = True

                with patch("shutil.copy") as mock_copy:
                    # Mock the database session
                    with patch("sqlalchemy.orm.Session") as mock_session:
                        mock_session_instance = MagicMock()

                        # Mock existing records check to prevent duplicates
                        mock_query_result = [
                            MagicMock(
                                filename="existing.jpg", path="/path/to/existing.jpg"
                            )
                        ]
                        mock_session_instance.query.return_value.filter.return_value.all.return_value = mock_query_result

                        mock_session.return_value.__enter__.return_value = (
                            mock_session_instance
                        )

                        # Test the endpoint
                        response = client.post("/sync-anomaly-data")
                        assert response.status_code == 200

                        # Check response structure
                        data = response.json()
                        assert "message" in data
                        assert "imported_count" in data
                        assert "anomaly_count" in data
                        assert "skipped_count" in data
                        assert "error_count" in data
                        assert "total_records" in data

                        # Verify database was updated
                        assert (
                            mock_session_instance.add.call_count >= 1
                        )  # At least one record added
                        assert mock_session_instance.commit.call_count >= 1

    def test_pca_visualization(self, client):
        """Test PCA projection visualization."""
        # Mock pandas read_csv
        with patch("pandas.read_csv") as mock_read_csv:
            import pandas as pd

            # Create a mock dataframe with test data with proper data types
            mock_df = pd.DataFrame(
                {
                    "image_path": ["path/to/image1.jpg", "path/to/image2.jpg"],
                    "reconstruction_error": [0.05, 0.15],
                    "is_anomaly": [False, True],
                    # Make sure anomaly_score is within expected range (0-1)
                    "anomaly_score": [0.5, 0.75],
                    "pca_1": [1.0, 2.0],
                    "pca_2": [3.0, 4.0],
                }
            )
            mock_read_csv.return_value = mock_df

            # Mock generate_pca_projection function
            with patch(
                "anomaly_reaper.interfaces.api.generate_pca_projection"
            ) as mock_pca:
                # Return values expected by the API implementation
                mock_pca.return_value = (
                    "base64encodedvisualization",  # Base64 encoded visualization
                    {
                        "points": [
                            {
                                "x": 1,
                                "y": 2,
                                "is_anomaly": True,
                                "image_path": "path/to/image1.jpg",
                            },
                            {
                                "x": 3,
                                "y": 4,
                                "is_anomaly": False,
                                "image_path": "path/to/image2.jpg",
                            },
                        ]
                    },  # Projection data as a dict with points array
                    0.1,  # Threshold
                )

                # Test with default parameters
                response = client.post("/visualizations/pca", json={})
                assert response.status_code == 200

                data = response.json()
                assert "visualization" in data
                assert data["visualization"] == "base64encodedvisualization"
                assert "anomaly_threshold" in data
                assert "total_points" in data
                assert "anomaly_count" in data

                # Test with custom parameters
                response = client.post(
                    "/visualizations/pca",
                    json={
                        "highlight_anomalies": True,
                        "use_interactive": True,
                        "include_image_paths": True,
                        "filter": {"is_anomaly": True},
                    },
                )
                assert response.status_code == 200

                # Don't directly compare DataFrames as it causes issues in assert_called_with
                # Instead check that the mock was called the correct number of times
                assert mock_pca.call_count == 2

                # Get the last call arguments
                args, kwargs = mock_pca.call_args
                # Check that the right parameters were passed
                assert kwargs["highlight_anomalies"] is True
                assert kwargs["use_interactive"] is True
                assert kwargs["include_paths"] is True

    def test_real_image_upload(self, client, test_image_path):
        """Test uploading a real image file instead of using mocks."""
        # Open the actual test image file for upload
        with open(test_image_path, "rb") as test_image:
            files = {"file": ("test_image.png", test_image, "image/png")}

            # Mock the image processing functions but use real file
            with patch("anomaly_reaper.interfaces.api.process_image") as mock_process:
                # Create a mock result that matches what the API expects
                mock_process.return_value = {
                    "id": "test-upload-123",
                    "filename": "test_image.png",
                    "path": "/path/to/saved/test_image.png",
                    "reconstruction_error": 0.05,
                    "is_anomaly": False,
                    "anomaly_score": 0.3,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }

                # Test uploading the image - using the correct endpoint path "/images" instead of "/images/upload"
                response = client.post("/images", files=files)
                assert response.status_code == 200

                # Check the response structure
                data = response.json()
                assert data["id"] == "test-upload-123"
                assert data["filename"] == "test_image.png"
                assert "timestamp" in data
                assert "anomaly_score" in data
                assert "is_anomaly" in data

                # Verify the mock was called with actual file data
                mock_process.assert_called_once()
                # The first argument should be the actual file data
                args, kwargs = mock_process.call_args
                assert isinstance(args[0], bytes)  # Verify file data was passed

    def test_real_image_processing_flow(self, client, test_image_path):
        """Test a more realistic image processing flow with a real image file."""
        # Open the actual test image file for upload
        with open(test_image_path, "rb") as test_image:
            files = {"file": ("test_image.png", test_image, "image/png")}

            # Setup session patching to track what gets stored in the database
            with patch("sqlalchemy.orm.Session") as mock_session:
                mock_session_instance = MagicMock()
                mock_session.return_value.__enter__.return_value = mock_session_instance

                # Only mock the PCA model loading function with the correct import path
                with patch(
                    "anomaly_reaper.processing.anomaly_detection.load_model"
                ) as mock_load_model:
                    import numpy as np

                    # Create a simple mock model that returns low reconstruction error (not anomalous)
                    mock_model = MagicMock()
                    mock_model.transform.return_value = np.array(
                        [[0.1, 0.2, 0.3]]
                    )  # PCA components
                    mock_model.inverse_transform.return_value = np.array(
                        [[0.1, 0.2, 0.3]]
                    )
                    mock_load_model.return_value = (
                        mock_model,
                        MagicMock(),
                    )  # (model, scaler)

                    # Mock the function that extracts features from images to avoid
                    # needing actual ML model dependencies in integration tests
                    with patch(
                        "anomaly_reaper.processing.feature_extractor.extract_features"
                    ) as mock_extract:
                        # Return simple feature vector
                        mock_extract.return_value = np.array([0.1, 0.2, 0.3])

                        # Mock image file saving to avoid affecting real filesystem
                        with patch("shutil.copy") as mock_copy:
                            with patch("uuid.uuid4") as mock_uuid:
                                mock_uuid.return_value = uuid.UUID(
                                    "12345678-1234-5678-1234-567812345678"
                                )

                                # Execute the upload request - using the correct endpoint path "/images"
                                response = client.post("/images", files=files)
                                assert response.status_code == 200

                                # Check response structure
                                data = response.json()
                                assert "id" in data
                                assert "filename" in data
                                assert "timestamp" in data
                                assert "is_anomaly" in data
                                assert "anomaly_score" in data

                                # Verify image was processed correctly
                                # Check database record was created
                                assert mock_session_instance.add.call_count >= 1
                                assert mock_session_instance.commit.call_count >= 1

                                # UUID should have been generated for the image ID
                                mock_uuid.assert_called()

                                # Feature extraction should have been called with image data
                                mock_extract.assert_called_once()

                                # Now get the image details via the API
                                image_id = data["id"]

                                # Mock out the query for retrieving image details
                                with patch(
                                    "anomaly_reaper.interfaces.api.query_images"
                                ) as mock_query:
                                    mock_image = MagicMock()
                                    mock_image.id = image_id
                                    mock_image.filename = "test_image.png"
                                    mock_image.path = "/path/to/test_image.png"
                                    mock_image.reconstruction_error = 0.05
                                    mock_image.is_anomaly = False
                                    mock_image.anomaly_score = 0.3
                                    mock_image.processed_at = datetime.now(timezone.utc)

                                    mock_query.return_value = ([mock_image], 1, 1)

                                    # Get the image details through the API
                                    details_response = client.get(f"/images/{image_id}")
                                    assert details_response.status_code == 200

                                    # Check that details match what we expect
                                    details_data = details_response.json()
                                    assert details_data["id"] == image_id
                                    assert details_data["filename"] == "test_image.png"
                                    assert not details_data["is_anomaly"]
                                    assert "anomaly_score" in details_data

    def test_reclassify_image(self, client, test_image_record):
        """Test reclassifying an image multiple times to verify history is maintained."""
        # Mock the classification function
        with patch("anomaly_reaper.interfaces.api.classify_image") as mock_classify:
            # Return success for each classification
            mock_classify.return_value = True

            # First classification - mark as anomaly
            response1 = client.post(
                f"/images/{test_image_record}/classify",
                json={"is_anomaly": True, "comment": "This looks anomalous"},
            )
            assert response1.status_code == 200
            data1 = response1.json()
            assert data1["user_classification"] is True

            # Second classification - change to normal
            response2 = client.post(
                f"/images/{test_image_record}/classify",
                json={
                    "is_anomaly": False,
                    "comment": "On second thought, this is normal",
                },
            )
            assert response2.status_code == 200
            data2 = response2.json()
            assert data2["user_classification"] is False

            # Third classification - back to anomaly
            response3 = client.post(
                f"/images/{test_image_record}/classify",
                json={"is_anomaly": True, "comment": "No, definitely anomalous"},
            )
            assert response3.status_code == 200
            data3 = response3.json()
            assert data3["user_classification"] is True

            # Verify classification history was maintained by mocking the DB lookup
            with patch("anomaly_reaper.interfaces.api.get_db") as mock_get_db:
                # Mock session behavior
                mock_session = MagicMock()
                mock_get_db.return_value.__enter__.return_value = mock_session

                # Mock query to find image
                mock_image_query = MagicMock()
                mock_image = MagicMock(id=test_image_record)
                mock_image_query.filter.return_value.first.return_value = mock_image

                # Mock classifications
                mock_classifications = [
                    MagicMock(
                        id=f"test-class-{i}",
                        image_id=test_image_record,
                        user_classification=(
                            [True, False, True][i - 1]
                        ),  # Match our classifications above
                        comment=[
                            "This looks anomalous",
                            "On second thought, this is normal",
                            "No, definitely anomalous",
                        ][i - 1],
                        timestamp=datetime.now(timezone.utc),
                    )
                    for i in range(1, 4)
                ]
                mock_class_query = MagicMock()
                mock_class_query.filter.return_value.order_by.return_value.all.return_value = mock_classifications

                # Set up query side effects
                mock_session.query.side_effect = [mock_image_query, mock_class_query]

                # Get classification history
                history_response = client.get(
                    f"/images/{test_image_record}/classifications"
                )
                assert history_response.status_code == 200
                history_data = history_response.json()

                # Verify history data structure
                assert history_data["image_id"] == test_image_record
                assert len(history_data["classifications"]) == 3

                # Check that the most recent classification is first (reverse order)
                assert history_data["classifications"][0]["user_classification"] is True
                assert (
                    "No, definitely anomalous"
                    in history_data["classifications"][0]["comment"]
                )
                assert (
                    history_data["classifications"][1]["user_classification"] is False
                )
                assert history_data["classifications"][2]["user_classification"] is True

    def test_high_reconstruction_error_handling(self, client):
        """Test handling of images with high reconstruction error."""
        # Mock file upload
        mock_file_content = b"mock_image_data"
        files = {"file": ("test_high_error.png", mock_file_content, "image/png")}

        # Mock process_image to return very high reconstruction error
        with patch("anomaly_reaper.interfaces.api.process_image") as mock_process:
            # Return higher than threshold error
            mock_process.return_value = {
                "reconstruction_error": 100.0,  # Very high error
                "is_anomaly": True,
                "anomaly_score": 10.0,
            }

            # Mock file operations
            with patch("builtins.open", mock_open()):
                # Mock UUID generation
                with patch("uuid.uuid4") as mock_uuid:
                    mock_uuid.return_value = "test-uuid"

                    # Execute request
                    response = client.post("/images", files=files)

                    # High reconstruction error should return 422 status
                    assert response.status_code == 422

                    data = response.json()
                    assert "reconstruction_error" in data
                    assert data["reconstruction_error"] >= 100.0
                    assert "is_anomaly" in data

                    # No ID should be returned as image wasn't stored
                    assert data["id"] == ""

    def test_batch_classification_request_validation(self, client):
        """Test validation of batch classification requests."""
        # Test with empty image_ids list
        empty_request = {"image_ids": [], "is_anomaly": True, "comment": "Test comment"}

        # Mock the validation behavior to return the expected error status
        with patch("anomaly_reaper.interfaces.api.get_db") as mock_get_db:
            # Mock session behavior
            mock_session = MagicMock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Mock the validation to fail for empty image_ids
            response_empty = client.patch("/images/classifications", json=empty_request)
            assert response_empty.status_code == 422  # Validation error expected

            # Test with missing is_anomaly field
            missing_field_request = {
                "image_ids": ["test-id-1", "test-id-2"],
                "comment": "Test comment",
            }
            response_missing = client.patch(
                "/images/classifications", json=missing_field_request
            )
            assert response_missing.status_code == 422  # Validation error expected

            # Test with valid request
            valid_request = {
                "image_ids": ["test-id-1", "test-id-2"],
                "is_anomaly": True,
                "comment": "Valid test comment",
            }

            # Mock behavior for valid request
            with patch("anomaly_reaper.interfaces.api.classify_image") as mock_classify:
                mock_classify.return_value = {
                    "id": "test-id",
                    "user_classification": True,
                }

                response_valid = client.patch(
                    "/images/classifications", json=valid_request
                )
                assert response_valid.status_code == 200

    def test_advanced_filtering_combinations(self, client):
        """Test complex combinations of filters in the search endpoint."""
        with patch("anomaly_reaper.interfaces.api.get_db") as mock_get_db:
            # Mock session behavior
            mock_session = MagicMock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Mock query building
            mock_query = MagicMock()
            mock_query.filter.return_value = mock_query
            mock_query.join.return_value = mock_query
            mock_query.count.return_value = 3
            mock_query.order_by.return_value = mock_query
            mock_session.query.return_value = mock_query

            # Mock query results
            mock_results = [
                MagicMock(
                    id=f"test-id-{i}",
                    filename=f"test{i}.jpg",
                    path=f"/path/to/test{i}.jpg",
                    reconstruction_error=0.05 + (i * 0.05),
                    is_anomaly=bool(i % 2),  # Alternate True/False
                    anomaly_score=0.5 + (i * 0.1),
                    processed_at=datetime.now(timezone.utc),
                )
                for i in range(1, 4)  # 3 records
            ]
            mock_query.offset.return_value.limit.return_value.all.return_value = (
                mock_results
            )

            # Test complex filter combinations

            # Test 1: Date range + anomaly status + score range
            response1 = client.get(
                "/images/search?is_anomaly=true&min_score=0.5&max_score=0.8"
                + f"&start_date={datetime.now(timezone.utc).isoformat()}"
                + "&page=1&page_size=10"
            )
            assert response1.status_code == 200
            data1 = response1.json()
            assert len(data1["results"]) == 3

            # Test 2: Classification status filter
            response2 = client.get(
                "/images/search?is_classified=true&user_classification=true"
                + "&page=1&page_size=10"
            )
            assert response2.status_code == 200
            data2 = response2.json()
            assert len(data2["results"]) == 3

            # Test 3: Combining everything
            response3 = client.get(
                "/images/search?is_anomaly=true&min_score=0.5&max_score=0.9"
                + "&is_classified=true&user_classification=false"
                + f"&start_date={datetime.now(timezone.utc).isoformat()}"
                + "&page=1&page_size=10&sort_by=anomaly_score&sort_order=desc"
            )
            assert response3.status_code == 200
            data3 = response3.json()
            assert len(data3["results"]) == 3

    def test_dashboard_statistics_types(self, client):
        """Test the dashboard statistics endpoint returns correct data types."""
        with patch("anomaly_reaper.interfaces.api.get_db") as mock_get_db:
            # Mock session behavior
            mock_session = MagicMock()
            mock_get_db.return_value.__enter__.return_value = mock_session

            # Mock count queries with scalar()
            total_images_query = MagicMock()
            total_images_query.scalar.return_value = 100

            total_anomalies_query = MagicMock()
            total_anomalies_query.scalar.return_value = 15

            user_confirmed_query = MagicMock()
            user_confirmed_query.scalar.return_value = 10

            unclassified_query = MagicMock()
            unclassified_query.scalar.return_value = 5

            false_positives_query = MagicMock()
            false_positives_query.scalar.return_value = 3

            false_negatives_query = MagicMock()
            false_negatives_query.scalar.return_value = 2

            # Mock recent activity records
            recent_records = [
                MagicMock(
                    timestamp=datetime.now(timezone.utc),
                    image_id=f"test-id-{i}",
                    user_classification=bool(i % 2),
                    comment=f"Test comment {i}",
                )
                for i in range(1, 4)  # 3 records
            ]
            recent_activity_query = MagicMock()
            recent_activity_query.all.return_value = recent_records

            # Mock image records for activity
            image_records = [
                MagicMock(
                    id=f"test-id-{i}",
                    is_anomaly=bool(
                        (i + 1) % 2
                    ),  # Opposite of classification to test false pos/neg
                    anomaly_score=0.5 + (i * 0.1),
                )
                for i in range(1, 4)
            ]
            image_query = MagicMock()

            # Set up side effects for different queries
            mock_session.query.side_effect = [
                total_images_query,
                total_anomalies_query,
                user_confirmed_query,
                unclassified_query,
                false_positives_query,
                false_negatives_query,
                recent_activity_query,
                image_query,  # For each activity record's image lookup
                image_query,
                image_query,
            ]

            # For each image lookup in activity records
            image_query.filter.return_value.first.side_effect = image_records

            # Test the endpoint
            response = client.get("/statistics/dashboard")
            assert response.status_code == 200

            # Check response data types
            data = response.json()
            assert isinstance(data["total_images"], int)
            assert isinstance(data["total_anomalies"], int)
            assert isinstance(data["user_confirmed_anomalies"], int)
            assert isinstance(data["unclassified_anomalies"], int)
            assert isinstance(data["false_positives"], int)
            assert isinstance(data["false_negatives"], int)
            assert isinstance(data["recent_activity"], list)

            # Check activity record structure
            for activity in data["recent_activity"]:
                assert "timestamp" in activity
                assert "image_id" in activity
                assert "action" in activity
                assert "is_anomaly" in activity
                # Anomaly score should be a float
                if "anomaly_score" in activity:
                    assert isinstance(activity["anomaly_score"], float)
                # Model prediction should be a boolean if present
                if "model_prediction" in activity:
                    assert isinstance(activity["model_prediction"], bool)
