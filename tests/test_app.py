"""
Integration tests for the Shiny app.
"""

from unittest.mock import Mock, patch

import pytest

from app import app_ui, server


class TestApp:

    def test_app_ui_structure(self):
        """Test that the UI has the expected structure."""
        # This is a basic test to ensure the UI object is created
        assert app_ui is not None
        # More detailed UI testing would require Shiny's test utilities

    @patch("app.FitProcessor")
    @patch("app.DiffAnalyzer")
    def test_server_initialization(self, mock_diff_analyzer, mock_fit_processor):
        """Test that server function can be called without errors."""
        # Mock inputs, outputs, session
        mock_input = Mock()
        mock_output = Mock()
        mock_session = Mock()

        # This should not raise an exception
        server_func = server(mock_input, mock_output, mock_session)

        # The server function returns None but sets up reactive handlers
        assert server_func is None
        # The server function returns None but sets up reactive handlers
        assert server_func is None
