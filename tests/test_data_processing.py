"""
Tests for data processing functions.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils import prepare_data_for_analysis, process_fit, remove_outliers


class TestProcessFit:
    """Test cases for process_fit function."""

    @patch("src.utils.Stream")
    @patch("src.utils.Decoder")
    def test_process_fit_success(self, mock_decoder_class, mock_stream_class):
        """Test successful FIT file processing."""
        # Setup mocks
        mock_stream = Mock()
        mock_stream_class.from_file.return_value = mock_stream

        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Mock messages
        mock_messages = {
            "record_mesgs": [
                {"heart_rate": 150, "speed": 5.5, "timestamp": "2023-01-01T12:00:00Z"},
                {"heart_rate": 155, "speed": 6.0, "timestamp": "2023-01-01T12:00:01Z"},
            ],
            "session_mesgs": [{"total_distance": 1000, "avg_heart_rate": 152}],
        }
        mock_decoder.read.return_value = (mock_messages, [])

        # Test
        session_df, record_df = process_fit("test.fit")

        # Verify
        assert isinstance(session_df, pd.DataFrame)
        assert isinstance(record_df, pd.DataFrame)
        assert len(record_df) == 2
        assert len(session_df) == 1
        assert "filename" in record_df.columns
        assert "filename" in session_df.columns
        assert record_df["filename"].iloc[0] == "test.fit"

    @patch("src.utils.Stream")
    @patch("src.utils.Decoder")
    def test_process_fit_with_position_data(
        self, mock_decoder_class, mock_stream_class
    ):
        """Test FIT file processing with GPS position data."""
        # Setup mocks
        mock_stream = Mock()
        mock_stream_class.from_file.return_value = mock_stream

        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Mock messages with position data (using raw semicircles format)
        mock_messages = {
            "record_mesgs": [
                {
                    "heart_rate": 150,
                    "position_lat": 2**30,  # Should convert to 90 degrees
                    "position_long": 2**30,  # Should convert to 90 degrees
                },
            ],
            "session_mesgs": [{"total_distance": 1000}],
        }
        mock_decoder.read.return_value = (mock_messages, [])

        # Test
        session_df, record_df = process_fit("test.fit")

        # Verify position conversion
        assert "position_lat" in record_df.columns
        assert "position_long" in record_df.columns
        assert abs(record_df["position_lat"].iloc[0] - 90.0) < 0.1
        assert abs(record_df["position_long"].iloc[0] - 90.0) < 0.1

    @patch("src.utils.Stream")
    @patch("src.utils.Decoder")
    def test_process_fit_no_record_messages(
        self, mock_decoder_class, mock_stream_class
    ):
        """Test FIT file with no record messages raises error."""
        # Setup mocks
        mock_stream = Mock()
        mock_stream_class.from_file.return_value = mock_stream

        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Mock messages with no records
        mock_messages = {
            "session_mesgs": [{"total_distance": 1000}],
        }
        mock_decoder.read.return_value = (mock_messages, [])

        # Test
        with pytest.raises(ValueError, match="No record messages found"):
            process_fit("test.fit")

    @patch("src.utils.Stream")
    @patch("src.utils.Decoder")
    def test_process_fit_no_session_messages(
        self, mock_decoder_class, mock_stream_class
    ):
        """Test FIT file with no session messages raises error."""
        # Setup mocks
        mock_stream = Mock()
        mock_stream_class.from_file.return_value = mock_stream

        mock_decoder = Mock()
        mock_decoder_class.return_value = mock_decoder

        # Mock messages with no sessions
        mock_messages = {
            "record_mesgs": [{"heart_rate": 150}],
        }
        mock_decoder.read.return_value = (mock_messages, [])

        # Test
        with pytest.raises(ValueError, match="No session messages found"):
            process_fit("test.fit")


class TestPrepareDataForAnalysis:
    """Test cases for prepare_data_for_analysis function."""

    def test_prepare_data_success(self):
        """Test successful data preparation."""
        # Create test data
        test_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-01 10:00:01"]
                ),
                "filename": ["test.fit", "test.fit"],
                "heart_rate": [150, 155],
                "speed": [5.0, 5.5],
            }
        )

        ref_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-01 10:00:01"]
                ),
                "filename": ["ref.fit", "ref.fit"],
                "heart_rate": [148, 153],
                "speed": [4.8, 5.3],
            }
        )

        # Test
        result = prepare_data_for_analysis((test_df, ref_df), "heart_rate")

        # Verify
        assert result is not None
        test_prepared, ref_prepared = result
        assert len(test_prepared) == 2
        assert len(ref_prepared) == 2
        assert "elapsed_seconds" in test_prepared.columns
        assert "elapsed_seconds" in ref_prepared.columns
        assert test_prepared["elapsed_seconds"].iloc[0] == 0
        assert test_prepared["elapsed_seconds"].iloc[1] == 1

    def test_prepare_data_no_common_timestamps(self):
        """Test data preparation with no common timestamps."""
        test_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01 10:00:00"]),
                "filename": ["test.fit"],
                "heart_rate": [150],
            }
        )

        ref_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01 11:00:00"]),
                "filename": ["ref.fit"],
                "heart_rate": [148],
            }
        )

        # Test
        result = prepare_data_for_analysis((test_df, ref_df), "heart_rate")

        # Verify
        assert result is None

    def test_prepare_data_invalid_input(self):
        """Test data preparation with invalid input."""
        # Test with None
        assert prepare_data_for_analysis(None, "heart_rate") is None

        # Test with empty tuple
        assert prepare_data_for_analysis((), "heart_rate") is None

        # Test with single dataframe
        assert prepare_data_for_analysis((pd.DataFrame(),), "heart_rate") is None

        # Test with empty dataframes
        empty_df = pd.DataFrame()
        assert prepare_data_for_analysis((empty_df, empty_df), "heart_rate") is None


class TestRemoveOutliers:
    """Test cases for remove_outliers function."""

    def create_test_data(self):
        """Create test data with outliers."""
        return pd.DataFrame(
            {
                "heart_rate": [
                    50,
                    60,
                    65,
                    70,
                    75,
                    80,
                    85,
                    90,
                    95,
                    1000,
                ],  # 1000 is extreme outlier
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="1s"),
                "filename": ["test.fit"] * 10,
            }
        )

    def test_remove_outliers_iqr(self):
        """Test IQR outlier removal."""
        df = self.create_test_data()

        # Test
        result = remove_outliers(df, "heart_rate", ["remove_iqr"])

        # Verify (200 should be removed as outlier)
        assert len(result) < len(df)
        assert 200 not in result["heart_rate"].values

    def test_remove_outliers_zscore(self):
        """Test Z-score outlier removal."""
        df = self.create_test_data()

        # Test
        result = remove_outliers(df, "heart_rate", ["remove_zscore"])

        # Verify (1000 should be removed as outlier)
        assert len(result) < len(df)
        assert 1000 not in result["heart_rate"].values

    def test_remove_outliers_percentile(self):
        """Test percentile outlier removal."""
        df = self.create_test_data()

        # Test
        result = remove_outliers(df, "heart_rate", ["remove_percentile"])

        # Verify
        assert len(result) <= len(df)

    def test_remove_outliers_multiple_methods(self):
        """Test multiple outlier removal methods."""
        df = self.create_test_data()

        # Test
        result = remove_outliers(df, "heart_rate", ["remove_iqr", "remove_zscore"])

        # Verify
        assert len(result) <= len(df)
        assert isinstance(result, pd.DataFrame)

    def test_remove_outliers_no_methods(self):
        """Test with no removal methods."""
        df = self.create_test_data()

        # Test
        result = remove_outliers(df, "heart_rate", [])

        # Verify (should return original dataframe)
        pd.testing.assert_frame_equal(result, df)

    def test_remove_outliers_invalid_input(self):
        """Test with invalid input."""
        df = self.create_test_data()

        # Test with None
        assert remove_outliers(None, "heart_rate", ["remove_iqr"]) is None

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        result = remove_outliers(empty_df, "heart_rate", ["remove_iqr"])
        assert result.empty

        # Test with missing metric
        result = remove_outliers(df, "nonexistent_metric", ["remove_iqr"])
        pd.testing.assert_frame_equal(result, df)
