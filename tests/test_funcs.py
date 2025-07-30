"""
Tests for FIT file processor.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.utils import process_fit


@patch("src.funcs.Stream")
@patch("src.funcs.Decoder")
def test_process_fit_file_success(mock_decoder, mock_stream):
    """Test successful FIT file processing."""
    # Mock the Garmin SDK components
    mock_stream_instance = Mock()
    mock_stream.from_file.return_value = mock_stream_instance

    mock_decoder_instance = Mock()
    mock_decoder.return_value = mock_decoder_instance

    # Mock the messages returned by decoder.read()
    mock_messages = {
        "record_mesgs": [
            {"heart_rate": 150, "speed": 5.5, "timestamp": "2023-01-01T12:00:00Z"},
            {"heart_rate": 155, "speed": 6.0, "timestamp": "2023-01-01T12:00:01Z"},
        ],
        "session_mesgs": [{"total_distance": 1000, "avg_heart_rate": 152}],
    }
    mock_decoder_instance.read.return_value = (mock_messages, [])

    # Test processing
    result = process_fit("test.fit")

    # Verify results
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # 2 record messages + 1 session message

    # Check record messages
    record_rows = result[result["message_type"] == "record"]
    assert len(record_rows) == 2
    assert record_rows.iloc[0]["heart_rate"] == 150
    assert record_rows.iloc[0]["speed"] == 5.5
    assert record_rows.iloc[1]["heart_rate"] == 155
    assert record_rows.iloc[1]["speed"] == 6.0

    # Check session message
    session_rows = result[result["message_type"] == "session"]
    assert len(session_rows) == 1
    assert session_rows.iloc[0]["total_distance"] == 1000
    assert session_rows.iloc[0]["avg_heart_rate"] == 152


@patch("src.funcs.Stream")
@patch("src.funcs.Decoder")
def test_process_fit_file_no_data(mock_decoder, mock_stream):
    """Test FIT file with no data."""
    mock_stream_instance = Mock()
    mock_stream.from_file.return_value = mock_stream_instance

    mock_decoder_instance = Mock()
    mock_decoder.return_value = mock_decoder_instance
    mock_decoder_instance.read.return_value = ({}, [])  # Empty messages

    with pytest.raises(ValueError, match="No data found"):
        process_fit("test.fit")


@patch("src.funcs.Stream")
@patch("src.funcs.Decoder")
def test_process_fit_file_unsupported_messages(mock_decoder, mock_stream):
    """Test FIT file processing - this test is now obsolete since we process all messages."""
    # Since we now process all messages, this test should pass with any valid data
    mock_stream_instance = Mock()
    mock_stream.from_file.return_value = mock_stream_instance

    mock_decoder_instance = Mock()
    mock_decoder.return_value = mock_decoder_instance

    # Mock some messages (any messages should work now)
    mock_messages = {"custom_mesgs": [{"custom_field": "custom_value"}]}
    mock_decoder_instance.read.return_value = (mock_messages, [])

    result = process_fit("test.fit")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["message_type"] == "custom"
    assert result.iloc[0]["custom_field"] == "custom_value"
