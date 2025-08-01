import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.utils import process_fit


@patch("src.utils.Stream.from_file")
@patch("src.utils.Decoder")
def test_process_fit_success(mock_decoder, mock_stream_from_file):
    mock_stream_instance = Mock()
    mock_stream_from_file.return_value = mock_stream_instance

    mock_decoder_instance = Mock()
    mock_decoder.return_value = mock_decoder_instance

    mock_messages = {
        "record_mesgs": [{"heart_rate": 150, "timestamp": "2023-01-01T12:00:00Z"}],
        "session_mesgs": [
            {"total_distance": 1000, "start_time": "2023-01-01T11:59:00Z"}
        ],
        "file_id_mesgs": [
            {
                "manufacturer": "testco",
                "product_name": "device",
                "serial_number": 1,
                "time_created": "2023-01-01T11:58:00Z",
            }
        ],
    }
    mock_decoder_instance.read.return_value = (mock_messages, [])

    session_df, record_df, meta = process_fit("test.fit")

    assert isinstance(session_df, pd.DataFrame)
    assert isinstance(record_df, pd.DataFrame)
    assert meta["manufacturer"] == "testco"
    assert len(record_df) == 1


@patch("src.utils.Stream.from_file")
@patch("src.utils.Decoder")
def test_process_fit_no_records(mock_decoder, mock_stream_from_file):
    mock_stream_instance = Mock()
    mock_stream_from_file.return_value = mock_stream_instance

    mock_decoder_instance = Mock()
    mock_decoder.return_value = mock_decoder_instance
    mock_decoder_instance.read.return_value = (
        {"record_mesgs": [], "session_mesgs": []},
        [],
    )

    with pytest.raises(ValueError):
        process_fit("bad.fit")
