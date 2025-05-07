"""
Models for the anomaly_reaper application.

This module contains SQLAlchemy database models for the anomaly detection
and classification system.
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
from sqlalchemy.orm import relationship, sessionmaker, Session, declarative_base
from typing import Generator
import datetime

# Import settings
from anomaly_reaper.config import settings

# Database setup
engine = create_engine(
    settings.db_url,
    pool_size=20,
    max_overflow=20,
    pool_timeout=60,
    pool_recycle=360,
    pool_pre_ping=True,
)
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
    processed_at : datetime
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
    reconstruction_error = Column(Float)
    is_anomaly = Column(Boolean)
    anomaly_score = Column(Float)
    processed_at = Column(
        DateTime, default=datetime.datetime.now(datetime.timezone.utc)
    )
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
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    image = relationship("ImageRecord", back_populates="classifications")


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
