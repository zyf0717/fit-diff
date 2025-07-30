"""
Integration tests for the Shiny app.
"""

from unittest.mock import Mock

from src.app_server import server
from src.app_ui import app_ui


def test_app_ui_structure():
    """Test that the UI has the expected structure."""
    # This is a basic test to ensure the UI object is created
    assert app_ui is not None
    # More detailed UI testing would require Shiny's test utilities


def test_server_initialization():
    """Test that server function can be called without errors."""
    # Mock inputs, outputs, session
    mock_input = Mock()
    mock_output = Mock()
    mock_session = Mock()

    # The server function returns None but sets up reactive handlers
    assert server(mock_input, mock_output, mock_session) is None
