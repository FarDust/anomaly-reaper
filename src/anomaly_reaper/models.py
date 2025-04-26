"""
Models for the anomaly_reaper application.

This module contains both SQLAlchemy database models and Pydantic models
for the anomaly detection and classification system.
"""

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Boolean,
    Float,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Generator
import datetime

# Import settings
from anomaly_reaper.config import settings

# Database setup
engine = create_engine(settings.db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Create and yield a database session.

    This function creates a new SQLAlchemy Session that will be used in a single request,
    and then closed once the request is finished.

    Yields
    ------
    Session
        SQLAlchemy Session instance

    Examples
    --------
    >>> # In FastAPI dependency injection:
    >>> def endpoint(db: Session = Depends(get_db)):
    ...     # use db here
    ...     pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# SQLAlchemy Database models
class ImageRecord(Base):
    """
    SQLAlchemy model for storing image metadata and anomaly detection results.

    This class represents the database table that stores information about
    images that have been analyzed for anomalies.

    Attributes
    ----------
    id : str
        Unique identifier for the image record
    filename : str
        Original filename of the image
    timestamp : datetime
        When the image was processed
    reconstruction_error : float
        PCA reconstruction error for the image
    is_anomaly : bool
        Whether the image was classified as an anomaly
    anomaly_score : float
        Normalized anomaly score
    path : str
        Path where the image is stored on disk
    classifications : relationship
        Relationship to user classifications of this image
    user_classified : property
        Whether this image has been classified by a user
    """

    __tablename__ = "image_records"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    reconstruction_error = Column(Float)
    is_anomaly = Column(Boolean)
    anomaly_score = Column(Float)
    path = Column(String)
    classifications = relationship(
        "Classification", back_populates="image", cascade="all, delete-orphan"
    )

    @property
    def user_classified(self) -> bool:
        """
        Check if this image has been classified by a user.

        Returns
        -------
        bool
            True if there are any user classifications, False otherwise
        """
        return len(self.classifications) > 0


class Classification(Base):
    """
    SQLAlchemy model for storing user classifications of images.

    This class represents the database table that stores user-provided
    classifications of images, including optional comments.

    Attributes
    ----------
    id : str
        Unique identifier for the classification
    image_id : str
        Foreign key reference to the associated image
    user_classification : bool
        User's classification (True for anomaly, False for normal)
    comment : str, optional
        Optional comment about the classification
    timestamp : datetime
        When the classification was made
    image : relationship
        Relationship to the parent image record
    """

    __tablename__ = "classifications"

    id = Column(String, primary_key=True)
    image_id = Column(String, ForeignKey("image_records.id"))
    user_classification = Column(Boolean)  # True for anomaly, False for normal
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    image = relationship("ImageRecord", back_populates="classifications")


# Pydantic models for request/response
class ImageResponse(BaseModel):
    """
    Pydantic model for image response data.

    This model defines the structure of the API response when returning
    image anomaly detection results.

    Attributes
    ----------
    id : str
        Unique identifier for the image
    filename : str
        Original filename of the image
    timestamp : datetime
        When the image was processed
    reconstruction_error : float
        PCA reconstruction error for the image
    is_anomaly : bool
        Whether the image was classified as an anomaly
    anomaly_score : float
        Normalized anomaly score
    path : str
        Path where the image is stored

    Examples
    --------
    >>> image_response = ImageResponse(
    ...     id="550e8400-e29b-41d4-a716-446655440000",
    ...     filename="example.jpg",
    ...     timestamp=datetime.datetime.now(),
    ...     reconstruction_error=0.123,
    ...     is_anomaly=True,
    ...     anomaly_score=2.5,
    ...     path="/uploads/example.jpg"
    ... )
    >>> print(image_response.is_anomaly)
    True
    """

    id: str
    filename: str
    timestamp: datetime.datetime
    reconstruction_error: float
    is_anomaly: bool
    anomaly_score: float
    path: str

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "filename": "example.jpg",
                    "timestamp": "2025-04-25T12:34:56.789Z",
                    "reconstruction_error": 0.123,
                    "is_anomaly": True,
                    "anomaly_score": 2.5,
                    "path": "/uploads/550e8400-e29b-41d4-a716-446655440000.jpg",
                }
            ]
        }
    )


class ClassificationRequest(BaseModel):
    """
    Pydantic model for classification request data.

    This model defines the structure of the API request when submitting
    a classification for an image.

    Attributes
    ----------
    user_classification : bool
        User's classification (True for anomaly, False for normal)
    comment : str, optional
        Optional comment about the classification

    Examples
    --------
    >>> classification = ClassificationRequest(
    ...     user_classification=True,
    ...     comment="This image shows an unusual pattern in the top right corner."
    ... )
    >>> print(classification.model_dump())
    {'user_classification': True, 'comment': 'This image shows an unusual pattern in the top right corner.'}
    """

    user_classification: bool = Field(
        ..., description="True if image is anomalous, False if normal"
    )
    comment: Optional[str] = Field(
        None, description="Optional comment about the classification"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "user_classification": True,
                    "comment": "This image shows an unusual pattern in the top right corner.",
                }
            ]
        }
    )


class ClassificationResponse(BaseModel):
    """
    Pydantic model for classification response data.

    This model defines the structure of the API response when returning
    information about an image classification.

    Attributes
    ----------
    id : str
        Unique identifier for the classification
    image_id : str
        ID of the classified image
    user_classification : bool
        User's classification (True for anomaly, False for normal)
    comment : str, optional
        Optional comment about the classification
    timestamp : datetime
        When the classification was made

    Examples
    --------
    >>> classification_response = ClassificationResponse(
    ...     id="550e8400-e29b-41d4-a716-446655440000",
    ...     image_id="550e8400-e29b-41d4-a716-446655440000",
    ...     user_classification=True,
    ...     comment="This image shows an unusual pattern in the top right corner.",
    ...     timestamp=datetime.datetime.now()
    ... )
    >>> print(classification_response.user_classification)
    True
    """

    id: str
    image_id: str
    user_classification: bool
    comment: Optional[str]
    timestamp: datetime.datetime

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "image_id": "550e8400-e29b-41d4-a716-446655440000",
                    "user_classification": True,
                    "comment": "This image shows an unusual pattern in the top right corner.",
                    "timestamp": "2025-04-25T12:34:56.789Z",
                }
            ]
        }
    )


def create_tables() -> None:
    """
    Create all database tables.

    This function creates all tables defined in SQLAlchemy models
    if they don't already exist in the database.

    Examples
    --------
    >>> create_tables()
    # Tables will be created in the database
    """
    Base.metadata.create_all(bind=engine)
