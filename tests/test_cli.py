"""
Tests for the Command Line Interface functionality.

These tests verify that the CLI interface properly processes commands and interacts
with the core functionality of the application.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from anomaly_reaper.interfaces.cli import app
from anomaly_reaper.models import ImageRecord


@pytest.fixture
def runner():
    """Fixture for Typer CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_process_directory(monkeypatch, test_settings):
    """Mock the process_directory function."""
    mock_func = MagicMock()
    mock_func.return_value = ["image1.jpg", "image2.jpg"]
    # Patch the function and the settings to use test_settings.models_dir
    monkeypatch.setattr("anomaly_reaper.interfaces.cli.process_directory", mock_func)
    monkeypatch.setattr(
        "anomaly_reaper.interfaces.cli.settings.models_dir", test_settings.models_dir
    )
    return mock_func


@pytest.fixture
def mock_db_operations(monkeypatch):
    """Mock database operations."""
    mock_store = MagicMock()
    monkeypatch.setattr("anomaly_reaper.interfaces.cli.store_results_in_db", mock_store)

    # For the test_list_anomalies_only test, we need different records based on the argument
    mock_query = MagicMock()
    mock_query.side_effect = (
        lambda anomalies_only=False: [
            ImageRecord(
                id="test-id-2",
                filename="test2.jpg",
                path="/path/to/test2.jpg",
                reconstruction_error=0.15,
                is_anomaly=True,
                anomaly_score=1.5,
            )
        ]
        if anomalies_only
        else [
            ImageRecord(
                id="test-id-1",
                filename="test1.jpg",
                path="/path/to/test1.jpg",
                reconstruction_error=0.05,
                is_anomaly=False,
                anomaly_score=0.5,
            ),
            ImageRecord(
                id="test-id-2",
                filename="test2.jpg",
                path="/path/to/test2.jpg",
                reconstruction_error=0.15,
                is_anomaly=True,
                anomaly_score=1.5,
            ),
        ]
    )
    monkeypatch.setattr("anomaly_reaper.interfaces.cli.query_images", mock_query)

    return mock_store, mock_query


class TestCLI:
    """Tests for the CLI interface."""

    def test_version_command(self, runner):
        """Test the version command."""
        with patch("anomaly_reaper.interfaces.cli.__version__", "0.1.0"):
            result = runner.invoke(app, ["version"])
            assert result.exit_code == 0
            assert "0.1.0" in result.stdout

    def test_process_command(
        self, runner, mock_process_directory, mock_db_operations, test_settings
    ):
        """Test the process command for directory scanning."""
        # Unpack the mocks
        mock_store, _ = mock_db_operations

        # Create a test directory path
        test_dir = os.path.join(test_settings.data_dir, "test_images")

        # Run the command
        result = runner.invoke(app, ["process", "--directory", test_dir])

        # Check result
        assert result.exit_code == 0
        assert "Processing" in result.stdout
        assert "Processed 2 images" in result.stdout

        # Verify that the mock functions were called correctly
        mock_process_directory.assert_called_once_with(
            test_dir, test_settings.models_dir, None
        )
        mock_store.assert_called_once()

    def test_process_command_with_threshold(
        self, runner, mock_process_directory, mock_db_operations, test_settings
    ):
        """Test the process command with a custom threshold."""
        # Create a test directory path
        test_dir = os.path.join(test_settings.data_dir, "test_images")

        # Run the command with a custom threshold
        result = runner.invoke(
            app, ["process", "--directory", test_dir, "--threshold", "0.7"]
        )

        # Check result
        assert result.exit_code == 0
        assert "Custom threshold: 0.7" in result.stdout

    def test_list_command(self, runner, mock_db_operations):
        """Test the list command for displaying images."""
        # Run the command
        result = runner.invoke(app, ["list"])

        # Check result
        assert result.exit_code == 0
        assert "test1.jpg" in result.stdout
        assert "test2.jpg" in result.stdout
        assert "ANOMALY" in result.stdout  # test2.jpg is an anomaly

    def test_list_anomalies_only(self, runner, mock_db_operations):
        """Test listing only anomalous images."""
        # Run the command
        result = runner.invoke(app, ["list", "--anomalies-only"])

        # Check result
        assert result.exit_code == 0
        assert "test2.jpg" in result.stdout
        assert "test1.jpg" not in result.stdout

    def test_classify_command(self, runner, mock_db_operations, monkeypatch):
        """Test the classify command."""
        # Mock the classify_image function
        mock_classify = MagicMock()
        mock_classify.return_value = {"id": "test-id", "user_classification": True}
        monkeypatch.setattr(
            "anomaly_reaper.interfaces.cli.classify_image", mock_classify
        )

        # Run the command
        result = runner.invoke(
            app, ["classify", "test-id-1", "true", "--comment", "Test comment"]
        )

        # Check result
        assert result.exit_code == 0
        assert "Classification successful" in result.stdout

        # Verify the mock was called correctly
        mock_classify.assert_called_once_with("test-id-1", True, "Test comment")

    @patch("anomaly_reaper.interfaces.cli.start_server")
    def test_serve_command(self, mock_start_server, runner):
        """Test the serve command."""
        # Run the command
        result = runner.invoke(app, ["serve", "--host", "localhost", "--port", "8000"])

        # Check result
        assert result.exit_code == 0
        assert "Starting server" in result.stdout

        # Verify the mock was called correctly
        mock_start_server.assert_called_once_with(host="localhost", port=8000)

    def test_export_command(
        self, runner, mock_db_operations, test_settings, monkeypatch
    ):
        """Test the export command."""
        # Mock the export_to_csv function
        mock_export = MagicMock()
        monkeypatch.setattr("anomaly_reaper.interfaces.cli.export_to_csv", mock_export)

        # Create a test output file path
        test_output = os.path.join(test_settings.data_dir, "export.csv")

        # Run the command
        result = runner.invoke(app, ["export", "--output", test_output])

        # Check result
        assert result.exit_code == 0
        assert f"records to {test_output}" in result.stdout

        # Verify the mock was called correctly
        mock_export.assert_called_once_with(
            output_path=test_output, anomalies_only=False
        )

    def test_export_anomalies_only(
        self, runner, mock_db_operations, test_settings, monkeypatch
    ):
        """Test exporting only anomalous images."""
        # Mock the export_to_csv function
        mock_export = MagicMock()
        monkeypatch.setattr("anomaly_reaper.interfaces.cli.export_to_csv", mock_export)

        # Create a test output file path
        test_output = os.path.join(test_settings.data_dir, "anomalies.csv")

        # Run the command
        result = runner.invoke(
            app, ["export", "--output", test_output, "--anomalies-only"]
        )

        # Check result
        assert result.exit_code == 0

        # Verify the mock was called correctly
        mock_export.assert_called_once_with(
            output_path=test_output, anomalies_only=True
        )
