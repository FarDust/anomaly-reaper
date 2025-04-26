"""
Anomaly Reaper: Image anomaly detection using PCA.

This package provides tools for detecting anomalies in images using PCA,
along with an API for classifying and managing detected anomalies.
"""

import typer
import uvicorn
import logging
from typing import Optional
from pathlib import Path

from anomaly_reaper.config import settings, logger

app = typer.Typer(help="Anomaly Reaper CLI")


@app.command()
def run(
    host: str = typer.Option(
        settings.host, "--host", "-h", help="Host to bind the server to"
    ),
    port: int = typer.Option(
        settings.port, "--port", "-p", help="Port to bind the server to"
    ),
    log_level: str = typer.Option(
        settings.log_level,
        "--log-level",
        "-l",
        help="Logging level (debug, info, warning, error, critical)",
    ),
    reload: bool = typer.Option(
        False, "--reload", "-r", help="Enable auto-reload for development"
    ),
):
    """Run the Anomaly Reaper API server."""
    # Set the log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start the server
    logger.info(f"Starting {settings.app_name} API server at http://{host}:{port}")
    uvicorn.run(
        "anomaly_reaper.interfaces.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )


@app.command()
def version():
    """Show the version of Anomaly Reaper."""
    from importlib.metadata import version as get_version

    try:
        version = get_version("anomaly-reaper")
    except:  # noqa: E722
        version = "0.1.0"  # Default if not installed as package

    logger.info(f"{settings.app_name} version: {version}")
    typer.echo(f"{settings.app_name} version: {version}")


@app.command()
def configure(
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-t", help="Set the anomaly detection threshold"
    ),
    models_dir: Optional[Path] = typer.Option(
        None, "--models-dir", "-m", help="Set the directory containing PCA models"
    ),
    uploads_dir: Optional[Path] = typer.Option(
        None, "--uploads-dir", "-u", help="Set the directory for uploaded images"
    ),
    db_url: Optional[str] = typer.Option(
        None, "--db-url", "-d", help="Set the database connection URL"
    ),
):
    """Configure Anomaly Reaper settings."""
    import json
    import os

    # Create a configuration file
    config = {}

    if threshold is not None:
        config["anomaly_threshold"] = threshold
        logger.info(f"Setting anomaly threshold to: {threshold}")

    if models_dir is not None:
        os.makedirs(models_dir, exist_ok=True)
        config["models_dir"] = str(models_dir)
        logger.info(f"Setting models directory to: {models_dir}")

    if uploads_dir is not None:
        os.makedirs(uploads_dir, exist_ok=True)
        config["uploads_dir"] = str(uploads_dir)
        logger.info(f"Setting uploads directory to: {uploads_dir}")

    if db_url is not None:
        config["db_url"] = db_url
        logger.info(f"Setting database URL to: {db_url}")

    # Save to a configuration file that can be loaded by settings
    if config:
        config_path = os.path.join(settings.models_dir, "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to {config_path}")
        typer.echo(f"Configuration saved to {config_path}")
        for key, value in config.items():
            typer.echo(f"  {key}: {value}")
    else:
        logger.warning("No configuration options provided. Nothing changed.")
        typer.echo("No configuration options provided. Nothing changed.")


def main():
    """Main entry point for the application."""
    app()
