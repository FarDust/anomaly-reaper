"""
Unit tests for the database models and ORM functionality.

These tests verify that the SQLAlchemy and Pydantic models work as expected.
"""

import pytest
import datetime
import uuid
from sqlalchemy.exc import IntegrityError

from anomaly_reaper.models import (
    ImageRecord,
    Classification,
    ImageResponse,
    ClassificationRequest,
    ClassificationResponse,
)


class TestSQLAlchemyModels:
    """Tests for SQLAlchemy ORM models."""

    def test_image_record_creation(self, db_session):
        """Test creating an ImageRecord in the database."""
        image_id = str(uuid.uuid4())
        image = ImageRecord(
            id=image_id,
            filename="test.jpg",
            reconstruction_error=0.1,
            is_anomaly=False,
            anomaly_score=0.8,
            path="/path/to/test.jpg",
        )

        db_session.add(image)
        db_session.commit()

        # Query the image back from the database
        saved_image = db_session.query(ImageRecord).filter_by(id=image_id).first()

        assert saved_image is not None
        assert saved_image.id == image_id
        assert saved_image.filename == "test.jpg"
        assert saved_image.reconstruction_error == 0.1
        assert saved_image.is_anomaly is False
        assert saved_image.anomaly_score == 0.8
        assert saved_image.path == "/path/to/test.jpg"
        assert isinstance(saved_image.timestamp, datetime.datetime)

    def test_classification_creation(self, db_session):
        """Test creating a Classification in the database."""
        # First, create an image record
        image_id = str(uuid.uuid4())
        image = ImageRecord(
            id=image_id,
            filename="test.jpg",
            reconstruction_error=0.1,
            is_anomaly=False,
            anomaly_score=0.8,
            path="/path/to/test.jpg",
        )
        db_session.add(image)
        db_session.commit()

        # Now create a classification for this image
        classification_id = str(uuid.uuid4())
        classification = Classification(
            id=classification_id,
            image_id=image_id,
            user_classification=True,
            comment="This looks suspicious",
        )

        db_session.add(classification)
        db_session.commit()

        # Query the classification back
        saved_classification = (
            db_session.query(Classification).filter_by(id=classification_id).first()
        )

        assert saved_classification is not None
        assert saved_classification.id == classification_id
        assert saved_classification.image_id == image_id
        assert saved_classification.user_classification is True
        assert saved_classification.comment == "This looks suspicious"
        assert isinstance(saved_classification.timestamp, datetime.datetime)

    def test_classification_relationship(self, db_session):
        """Test the relationship between ImageRecord and Classification."""
        # Create an image record
        image_id = str(uuid.uuid4())
        image = ImageRecord(
            id=image_id,
            filename="relationship_test.jpg",
            reconstruction_error=0.1,
            is_anomaly=False,
            anomaly_score=0.8,
            path="/path/to/relationship_test.jpg",
        )
        db_session.add(image)

        # Create two classifications for this image
        classifications = []
        for i in range(2):
            classification = Classification(
                id=str(uuid.uuid4()),
                image_id=image_id,
                user_classification=i % 2 == 0,
                comment=f"Test comment {i}",
            )
            classifications.append(classification)
            db_session.add(classification)

        db_session.commit()

        # Reload the image from the database
        saved_image = db_session.query(ImageRecord).filter_by(id=image_id).first()

        # Test the relationship
        assert len(saved_image.classifications) == 2
        classification_comments = [c.comment for c in saved_image.classifications]
        assert "Test comment 0" in classification_comments
        assert "Test comment 1" in classification_comments

    def test_cascade_delete(self, db_session):
        """Test that deleting an ImageRecord cascades to its Classifications."""
        # Create an image record
        image_id = str(uuid.uuid4())
        image = ImageRecord(
            id=image_id,
            filename="cascade_test.jpg",
            reconstruction_error=0.1,
            is_anomaly=False,
            anomaly_score=0.8,
            path="/path/to/cascade_test.jpg",
        )
        db_session.add(image)

        # Create a classification for this image
        classification_id = str(uuid.uuid4())
        classification = Classification(
            id=classification_id,
            image_id=image_id,
            user_classification=True,
            comment="This should be deleted when the image is deleted",
        )
        db_session.add(classification)

        db_session.commit()

        # Verify the classification exists
        assert (
            db_session.query(Classification).filter_by(id=classification_id).first()
            is not None
        )

        # Delete the image
        db_session.delete(image)
        db_session.commit()

        # Verify the classification was also deleted
        assert (
            db_session.query(Classification).filter_by(id=classification_id).first()
            is None
        )

    def test_duplicate_primary_key_fails(self, db_session):
        """Test that using the same primary key twice raises an error."""
        image_id = str(uuid.uuid4())

        # Create the first image
        image1 = ImageRecord(
            id=image_id,
            filename="original.jpg",
            reconstruction_error=0.1,
            is_anomaly=False,
            anomaly_score=0.8,
            path="/path/to/original.jpg",
        )
        db_session.add(image1)
        db_session.commit()

        # Try to create another image with the same ID
        image2 = ImageRecord(
            id=image_id,  # Same ID as image1
            filename="duplicate.jpg",
            reconstruction_error=0.2,
            is_anomaly=True,
            anomaly_score=2.0,
            path="/path/to/duplicate.jpg",
        )
        db_session.add(image2)

        # This should raise an IntegrityError
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestPydanticModels:
    """Tests for Pydantic models."""

    def test_image_response_model(self):
        """Test the ImageResponse Pydantic model."""
        timestamp = datetime.datetime.now()
        data = {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "filename": "test.jpg",
            "timestamp": timestamp,
            "reconstruction_error": 0.1,
            "is_anomaly": True,
            "anomaly_score": 0.8,
            "path": "/path/to/test.jpg",
        }

        # Create the model
        image_response = ImageResponse(**data)

        # Check the fields
        assert image_response.id == data["id"]
        assert image_response.filename == data["filename"]
        assert image_response.timestamp == data["timestamp"]
        assert image_response.reconstruction_error == data["reconstruction_error"]
        assert image_response.is_anomaly == data["is_anomaly"]
        assert image_response.anomaly_score == data["anomaly_score"]
        assert image_response.path == data["path"]

        # Test serialization
        json_data = image_response.model_dump()
        assert json_data["id"] == data["id"]
        assert json_data["is_anomaly"] == data["is_anomaly"]

    def test_classification_request_model(self):
        """Test the ClassificationRequest Pydantic model."""
        # With comment
        data1 = {"user_classification": True, "comment": "This is a test comment"}
        request1 = ClassificationRequest(**data1)
        assert request1.user_classification is True
        assert request1.comment == "This is a test comment"

        # Without comment (optional field)
        data2 = {"user_classification": False}
        request2 = ClassificationRequest(**data2)
        assert request2.user_classification is False
        assert request2.comment is None

    def test_classification_response_model(self):
        """Test the ClassificationResponse Pydantic model."""
        timestamp = datetime.datetime.now()
        data = {
            "id": "123e4567-e89b-12d3-a456-426614174001",
            "image_id": "123e4567-e89b-12d3-a456-426614174000",
            "user_classification": True,
            "comment": "Test comment",
            "timestamp": timestamp,
        }

        response = ClassificationResponse(**data)

        assert response.id == data["id"]
        assert response.image_id == data["image_id"]
        assert response.user_classification == data["user_classification"]
        assert response.comment == data["comment"]
        assert response.timestamp == data["timestamp"]

        # Test serialization
        json_data = response.model_dump()
        assert json_data["id"] == data["id"]
        assert json_data["image_id"] == data["image_id"]

        # Test without comment (optional field)
        data_no_comment = {
            "id": "123e4567-e89b-12d3-a456-426614174002",
            "image_id": "123e4567-e89b-12d3-a456-426614174000",
            "user_classification": False,
            "comment": None,
            "timestamp": timestamp,
        }

        response_no_comment = ClassificationResponse(**data_no_comment)
        assert response_no_comment.comment is None
