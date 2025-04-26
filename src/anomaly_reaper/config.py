"""
Configuration settings for the anomaly_reaper application.

This module uses Pydantic Settings to manage configuration options for
the API server, database, and model settings.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
import os
import logging
from typing import Optional


def configure_logging(level: str = "INFO") -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    level : str
        Logging level (debug, info, warning, error, critical)

    Examples
    --------
    >>> configure_logging(level="DEBUG")
    # Sets up logging with DEBUG level
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a logger for our application
    logger = logging.getLogger("anomaly_reaper")
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Log startup message
    logger.info(f"Logging configured with level: {level}")

    return logger


class Settings(BaseSettings):
    """
    Settings for the anomaly_reaper application.

    This class defines all configurable settings for the application using
    Pydantic's BaseSettings. Settings can be overridden via environment
    variables with the prefix 'ANOMALY_REAPER_'.

    Parameters
    ----------
    app_name : str
        Name of the application
    host : str
        Host to run the server on
    port : int
        Port to run the server on
    log_level : str
        Logging level (debug, info, warning, error, critical)
    models_dir : str
        Directory containing PCA models
    uploads_dir : str
        Directory for uploaded images
    db_url : str
        Database connection URL
    anomaly_threshold : float
        Threshold for anomaly detection
    project_id : str, optional
        Google Cloud Project ID
    location : str
        Google Cloud region

    Examples
    --------
    >>> # Create settings with default values
    >>> app_settings = Settings()
    >>> print(app_settings.host)
    127.0.0.1

    >>> # Create settings with custom values
    >>> custom_settings = Settings(
    ...     host="0.0.0.0",
    ...     port=5000,
    ...     anomaly_threshold=0.1
    ... )
    >>> print(custom_settings.port)
    5000

    >>> # Environment variables take precedence:
    >>> # export ANOMALY_REAPER_PORT=8080
    >>> # settings = Settings()  # port would be 8080
    """

    app_name: str = Field("Anomaly Reaper", description="Name of the application")
    host: str = Field("127.0.0.1", description="Host to bind the server to")
    port: int = Field(8000, description="Port to bind the server to")
    log_level: str = Field("info", description="Logging level")

    # Directory paths
    models_dir: str = Field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
        ),
        description="Directory containing PCA models",
    )
    uploads_dir: str = Field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads"
        ),
        description="Directory for uploaded images",
    )
    data_dir: str = Field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
        ),
        description="Directory for data files",
    )

    # Database configuration
    db_url: str = Field(
        "sqlite:///./anomaly_reaper.db", description="Database connection URL"
    )

    # Model configuration
    anomaly_threshold: float = Field(
        0.05, description="Threshold for anomaly detection"
    )

    # Google Cloud / VertexAI configuration
    project_id: Optional[str] = Field(None, description="Google Cloud Project ID")
    location: str = Field("us-central1", description="Google Cloud region")

    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """
        Validate that log_level is a valid logging level.

        Parameters
        ----------
        v : str
            The log level to validate

        Returns
        -------
        str
            The validated log level

        Raises
        ------
        ValueError
            If the log level is not valid
        """
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.lower()

    @field_validator("anomaly_threshold")
    def validate_threshold(cls, v: float) -> float:
        """
        Validate that anomaly_threshold is a positive value.

        Parameters
        ----------
        v : float
            The threshold value to validate

        Returns
        -------
        float
            The validated threshold value

        Raises
        ------
        ValueError
            If the threshold is not positive
        """
        if v <= 0:
            raise ValueError("Anomaly threshold must be positive")
        return v

    def create_directories(self) -> None:
        """
        Create necessary directories for the application.

        Creates the models and uploads directories if they don't exist.

        Examples
        --------
        >>> settings = Settings()
        >>> settings.create_directories()
        # Creates the models and uploads directories
        """
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(
            f"Created directories: {self.models_dir}, {self.uploads_dir}, {self.data_dir}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "ANOMALY_REAPER_",
    }


# Create a global settings instance
settings = Settings()

# Configure logging
logger = configure_logging(settings.log_level)

# Create necessary directories on module load
settings.create_directories()
