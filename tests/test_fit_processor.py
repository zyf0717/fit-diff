"""
Tests for FIT file processor.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.fit_processor import FitProcessor


class TestFitProcessor:

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FitProcessor()

    def test_init(self):
        """Test FitProcessor initialization."""
        assert self.processor.supported_messages == [
            "record",
            "session",
            "lap",
            "device_info",
            "file_id",
        ]

    @patch("src.fit_processor.FitFile")
    def test_process_fit_file_success(self, mock_fitfile):
        """Test successful FIT file processing."""
        # Mock FitFile and messages
        mock_message = Mock()
        mock_message.name = "record"

        mock_field1 = Mock()
        mock_field1.name = "heart_rate"
        mock_field1.value = 150

        mock_field2 = Mock()
        mock_field2.name = "speed"
        mock_field2.value = 5.5

        mock_message.fields = [mock_field1, mock_field2]

        mock_fitfile_instance = Mock()
        mock_fitfile_instance.get_messages.return_value = [mock_message]
        mock_fitfile.return_value = mock_fitfile_instance

        # Test processing
        result = self.processor.process_fit_file("test.fit")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["message_type"] == "record"
        assert result.iloc[0]["heart_rate"] == 150
        assert result.iloc[0]["speed"] == 5.5

    @patch("src.fit_processor.FitFile")
    def test_process_fit_file_no_data(self, mock_fitfile):
        """Test FIT file with no supported messages."""
        mock_fitfile_instance = Mock()
        mock_fitfile_instance.get_messages.return_value = []
        mock_fitfile.return_value = mock_fitfile_instance

        with pytest.raises(ValueError, match="No supported data found"):
            self.processor.process_fit_file("test.fit")

    @patch("src.fit_processor.FitFile")
    def test_process_fit_file_unsupported_messages(self, mock_fitfile):
        """Test FIT file with only unsupported message types."""
        mock_message = Mock()
        mock_message.name = "unsupported_message"
        mock_message.fields = []

        mock_fitfile_instance = Mock()
        mock_fitfile_instance.get_messages.return_value = [mock_message]
        mock_fitfile.return_value = mock_fitfile_instance

        with pytest.raises(ValueError, match="No supported data found"):
            self.processor.process_fit_file("test.fit")

    def test_extract_summary_stats_empty_df(self):
        """Test summary stats extraction with empty DataFrame."""
        df = pd.DataFrame()
        stats = self.processor.extract_summary_stats(df)
        assert isinstance(stats, dict)

    def test_extract_summary_stats_with_data(self):
        """Test summary stats extraction with data."""
        df = pd.DataFrame(
            {
                "message_type": ["record", "record", "session"],
                "heart_rate": [150, 160, None],
                "speed": [5.0, 5.5, None],
                "timestamp": pd.to_datetime(
                    [
                        "2023-01-01 10:00:00",
                        "2023-01-01 10:01:00",
                        "2023-01-01 10:02:00",
                    ]
                ),
            }
        )

        stats = self.processor.extract_summary_stats(df)

        assert "message_counts" in stats
        assert stats["message_counts"]["record"] == 2
        assert stats["message_counts"]["session"] == 1

        assert "start_time" in stats
        assert "end_time" in stats
        assert "duration" in stats

        assert "heart_rate_mean" in stats
        assert stats["heart_rate_mean"] == 155.0
        assert stats["heart_rate_max"] == 160
        assert stats["heart_rate_min"] == 150
        assert stats["heart_rate_max"] == 160
        assert stats["heart_rate_min"] == 150
