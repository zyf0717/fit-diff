"""
Tests for diff analyzer.
"""

import numpy as np
import pandas as pd
import pytest

from src.diff_analyzer import DiffAnalyzer


class TestDiffAnalyzer:

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DiffAnalyzer()

    def test_compare_files_mismatch_lengths(self):
        """Test error when dataframes and filenames lengths don't match."""
        df1 = pd.DataFrame({"test": [1, 2, 3]})
        df2 = pd.DataFrame({"test": [4, 5, 6]})

        with pytest.raises(ValueError, match="Number of dataframes must match"):
            self.analyzer.compare_files([df1, df2], ["file1.fit"])

    def test_compare_files_single_file(self):
        """Test comparison with single file."""
        df = pd.DataFrame(
            {
                "message_type": ["record", "record"],
                "heart_rate": [150, 160],
                "timestamp": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-01 10:01:00"]
                ),
            }
        )

        result = self.analyzer.compare_files([df], ["test.fit"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["filename"] == "test.fit"

    def test_compare_files_multiple_files(self):
        """Test comparison with multiple files."""
        df1 = pd.DataFrame(
            {
                "message_type": ["record", "record"],
                "heart_rate": [150, 160],
                "speed": [5.0, 5.5],
                "timestamp": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-01 10:01:00"]
                ),
            }
        )

        df2 = pd.DataFrame(
            {
                "message_type": ["record", "record", "record"],
                "heart_rate": [140, 155, 165],
                "speed": [4.8, 5.2, 5.8],
                "timestamp": pd.to_datetime(
                    [
                        "2023-01-01 11:00:00",
                        "2023-01-01 11:01:00",
                        "2023-01-01 11:02:00",
                    ]
                ),
            }
        )

        result = self.analyzer.compare_files([df1, df2], ["file1.fit", "file2.fit"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "filename" in result.columns
        assert result.iloc[0]["filename"] == "file1.fit"
        assert result.iloc[1]["filename"] == "file2.fit"

    def test_extract_basic_stats_empty_df(self):
        """Test basic stats extraction with empty DataFrame."""
        df = pd.DataFrame()
        stats = self.analyzer._extract_basic_stats(df)

        assert stats["total_records"] == 0
        assert isinstance(stats, dict)

    def test_extract_basic_stats_with_data(self):
        """Test basic stats extraction with data."""
        df = pd.DataFrame(
            {
                "message_type": ["record", "record", "session"],
                "heart_rate": [150, 160, None],
                "speed": [5.0, 5.5, None],
                "power": [200, 220, None],
                "timestamp": pd.to_datetime(
                    [
                        "2023-01-01 10:00:00",
                        "2023-01-01 10:01:00",
                        "2023-01-01 10:02:00",
                    ]
                ),
            }
        )

        stats = self.analyzer._extract_basic_stats(df)

        assert stats["total_records"] == 3
        assert stats["record_records"] == 2
        assert stats["session_records"] == 1
        assert stats["duration_seconds"] == 120.0  # 2 minutes

        # Check metric averages
        assert stats["heart_rate_avg"] == 155.0
        assert stats["heart_rate_max"] == 160
        assert stats["heart_rate_min"] == 150

        assert stats["speed_avg"] == 5.25
        assert stats["power_avg"] == 210.0

    def test_calculate_differences_single_file(self):
        """Test difference calculation with single file."""
        comparison_df = pd.DataFrame(
            {
                "filename": ["file1.fit"],
                "total_records": [100],
                "heart_rate_avg": [150.0],
                "file_index": [0],
            }
        )

        result = self.analyzer._calculate_differences(comparison_df)

        assert len(result) == 1
        assert result.iloc[0]["filename"] == "file1.fit"
        assert result.iloc[0]["total_records_diff"] == 0
        assert result.iloc[0]["heart_rate_avg_diff"] == 0

    def test_calculate_differences_multiple_files(self):
        """Test difference calculation with multiple files."""
        comparison_df = pd.DataFrame(
            {
                "filename": ["file1.fit", "file2.fit"],
                "total_records": [100, 120],
                "heart_rate_avg": [150.0, 155.0],
                "file_index": [0, 1],
            }
        )

        result = self.analyzer._calculate_differences(comparison_df)

        assert len(result) == 2

        # Base file (first) should have 0 differences
        assert result.iloc[0]["total_records_diff"] == 0
        assert result.iloc[0]["heart_rate_avg_diff"] == 0

        # Second file should show differences from base
        assert result.iloc[1]["total_records_diff"] == 20
        assert result.iloc[1]["heart_rate_avg_diff"] == 5.0

    def test_calculate_differences_with_nan_values(self):
        """Test difference calculation with NaN values."""
        comparison_df = pd.DataFrame(
            {
                "filename": ["file1.fit", "file2.fit"],
                "total_records": [100, np.nan],
                "heart_rate_avg": [150.0, 155.0],
                "file_index": [0, 1],
            }
        )

        result = self.analyzer._calculate_differences(comparison_df)

        assert len(result) == 2
        assert pd.isna(result.iloc[1]["total_records_diff"])
        assert result.iloc[1]["heart_rate_avg_diff"] == 5.0
        assert result.iloc[1]["heart_rate_avg_diff"] == 5.0
