"""
Tests for statistics calculation functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    calculate_basic_stats,
    calculate_ccc,
    get_file_information,
    get_precision_stats,
    get_raw_data_sample,
    get_reliability_stats,
    get_validity_stats,
)


class TestCalculateBasicStats:
    """Test cases for calculate_basic_stats function."""

    def create_aligned_data(self):
        """Create test aligned data."""
        return pd.DataFrame(
            {
                "heart_rate_test": [150, 155, 160, 165, 170],
                "heart_rate_ref": [148, 153, 158, 163, 168],
                "elapsed_seconds_test": [0, 1, 2, 3, 4],
                "elapsed_seconds_ref": [0, 1, 2, 3, 4],
            }
        )

    def test_calculate_basic_stats_success(self):
        """Test successful basic stats calculation."""
        aligned_df = self.create_aligned_data()

        # Test
        result = calculate_basic_stats(aligned_df, "heart_rate")

        # Verify
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "stat" in result.columns
        assert "test" in result.columns
        assert "reference" in result.columns

        # Check that expected stats are present
        stats = result["stat"].tolist()
        expected_stats = ["metric", "count", "mean", "std", "min", "max", "median"]
        for stat in expected_stats:
            assert stat in stats

    def test_calculate_basic_stats_invalid_input(self):
        """Test with invalid input."""
        # Test with None
        assert calculate_basic_stats(None, "heart_rate") is None

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        assert calculate_basic_stats(empty_df, "heart_rate") is None


class TestCalculateCCC:
    """Test cases for calculate_ccc function."""

    def test_calculate_ccc_perfect_agreement(self):
        """Test CCC with perfect agreement."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 2, 3, 4, 5])

        result = calculate_ccc(x, y)

        assert result == 1.0

    def test_calculate_ccc_high_agreement(self):
        """Test CCC with high agreement."""
        x = pd.Series([1.0, 2.1, 2.9, 4.1, 4.9])
        y = pd.Series([1, 2, 3, 4, 5])

        result = calculate_ccc(x, y)

        # Should be close to 1 but not perfect
        assert 0.99 <= result <= 1.0

    def test_calculate_ccc_negative_agreement(self):
        """Test CCC with negative correlation."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([5, 4, 3, 2, 1])

        result = calculate_ccc(x, y)

        assert result == -1.0

    def test_calculate_ccc_no_agreement(self):
        """Test CCC with uncorrelated data."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([3, 1, 4, 2, 5])

        result = calculate_ccc(x, y)

        # Should be close to 0 (less than perfect agreement)
        assert abs(result) <= 0.5

    def test_calculate_ccc_invalid_input(self):
        """Test with invalid input."""
        # Empty series
        empty_x = pd.Series([])
        empty_y = pd.Series([])
        assert calculate_ccc(empty_x, empty_y) == 0.0

        # Mismatched lengths
        x = pd.Series([1, 2, 3])
        y = pd.Series([1, 2])
        assert calculate_ccc(x, y) == 0.0

        # Constant values (zero variance)
        const_x = pd.Series([3, 3, 3, 3])
        const_y = pd.Series([3, 3, 3, 3])
        assert calculate_ccc(const_x, const_y) == 0.0


class TestGetBiasAgreementStats:
    """Test cases for get_validity_stats function."""

    def create_aligned_data(self):
        """Create test aligned data."""
        np.random.seed(42)  # For reproducible results
        return pd.DataFrame(
            {
                "heart_rate_test": np.random.normal(160, 10, 100),
                "heart_rate_ref": np.random.normal(158, 10, 100),
            }
        )

    def test_get_validity_stats_success(self):
        """Test successful bias agreement stats calculation."""
        aligned_df = self.create_aligned_data()

        # Test
        result = get_validity_stats(aligned_df, "heart_rate")

        # Verify
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "Metric" in result.columns
        assert "Value" in result.columns

        # Check that expected metrics are present
        metrics = result["Metric"].tolist()
        expected_metrics = [
            "Mean Bias",
            "Paired t-test p-value",
            "Wilcoxon signed-rank p-value",
            "Sign test p-value",
            "Cohen's d",
        ]
        for metric in expected_metrics:
            assert any(metric in m for m in metrics)

    def test_get_validity_stats_invalid_input(self):
        """Test with invalid input."""
        assert get_validity_stats(None, "heart_rate") is None
        assert get_validity_stats(pd.DataFrame(), "heart_rate") is None


class TestGetErrorMagnitudeStats:
    """Test cases for get_precision_stats function."""

    def create_aligned_data(self):
        """Create test aligned data."""
        return pd.DataFrame(
            {
                "heart_rate_test": [150, 155, 160, 165, 170],
                "heart_rate_ref": [148, 153, 158, 163, 168],
            }
        )

    def test_get_precision_stats_success(self):
        """Test successful error magnitude stats calculation."""
        aligned_df = self.create_aligned_data()

        # Test
        result = get_precision_stats(aligned_df, "heart_rate")

        # Verify
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "Metric" in result.columns
        assert "Value" in result.columns

        # Check that expected metrics are present
        metrics = result["Metric"].tolist()
        expected_metrics = ["MAE", "RMSE", "MSE", "Std of Errors"]
        assert all(metric in metrics for metric in expected_metrics)

    def test_get_precision_stats_invalid_input(self):
        """Test with invalid input."""
        assert get_precision_stats(None, "heart_rate") is None
        assert get_precision_stats(pd.DataFrame(), "heart_rate") is None


class TestGetCorrelationStats:
    """Test cases for get_reliability_stats function."""

    def create_aligned_data(self):
        """Create test aligned data."""
        return pd.DataFrame(
            {
                "heart_rate_test": [150, 155, 160, 165, 170],
                "heart_rate_ref": [148, 153, 158, 163, 168],
            }
        )

    def test_get_reliability_stats_success(self):
        """Test successful correlation stats calculation."""
        aligned_df = self.create_aligned_data()

        # Test
        result = get_reliability_stats(aligned_df, "heart_rate")

        # Verify
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "Metric" in result.columns
        assert "Value" in result.columns

        # Check that expected metrics are present
        metrics = result["Metric"].tolist()
        expected_metrics = [
            "Concordance Correlation Coefficient",
            "Pearson Correlation Coefficient",
            "Pearson Correlation P-value",
        ]
        assert all(metric in metrics for metric in expected_metrics)

    def test_get_reliability_stats_invalid_input(self):
        """Test with invalid input."""
        assert get_reliability_stats(None, "heart_rate") is None
        assert get_reliability_stats(pd.DataFrame(), "heart_rate") is None


class TestGetFileInformation:
    """Test cases for get_file_information function."""

    def create_test_data(self):
        """Create test data."""
        test_data = pd.DataFrame(
            {
                "filename": ["test1.fit", "test1.fit", "test2.fit"],
                "timestamp": pd.to_datetime(
                    [
                        "2023-01-01 10:00:00",
                        "2023-01-01 10:00:01",
                        "2023-01-01 10:00:00",
                    ]
                ),
                "heart_rate": [150, 155, 160],
            }
        )

        ref_data = pd.DataFrame(
            {
                "filename": ["ref1.fit", "ref1.fit"],
                "timestamp": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-01 10:00:01"]
                ),
                "heart_rate": [148, 153],
            }
        )

        return test_data, ref_data

    def test_get_file_information_success(self):
        """Test successful file information extraction."""
        test_data, ref_data = self.create_test_data()

        # Test
        result = get_file_information(test_data, ref_data)

        # Verify
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "device_type" in result.columns  # Correct column name
        assert "filename" in result.columns
        assert "records" in result.columns  # Correct column name
        assert "start_time" in result.columns
        assert "end_time" in result.columns

    def test_get_file_information_invalid_input(self):
        """Test with invalid input."""
        empty_df = pd.DataFrame()

        assert get_file_information(None, empty_df) is None
        assert get_file_information(empty_df, None) is None
        assert get_file_information(empty_df, empty_df) is None


class TestGetRawDataSample:
    """Test cases for get_raw_data_sample function."""

    def create_test_data(self):
        """Create test data."""
        test_data = pd.DataFrame(
            {
                "filename": ["test.fit"] * 200,
                "timestamp": pd.date_range("2023-01-01", periods=200, freq="1s"),
                "heart_rate": range(150, 350),
                "speed": np.random.uniform(0, 10, 200),
                "empty_col": [None] * 200,  # Column with all NaN/None values
                "mostly_empty_col": [1] + [None] * 199,  # Column with mostly NaN/None
            }
        )

        ref_data = pd.DataFrame(
            {
                "filename": ["ref.fit"] * 200,
                "timestamp": pd.date_range("2023-01-01", periods=200, freq="1s"),
                "heart_rate": range(148, 348),
                "speed": np.random.uniform(0, 10, 200),
                "empty_col": [None] * 200,
                "mostly_empty_col": [None] * 200,
            }
        )

        return test_data, ref_data

    def test_get_raw_data_sample_success(self):
        """Test successful raw data sampling."""
        test_data, ref_data = self.create_test_data()

        # Test
        result = get_raw_data_sample(test_data, ref_data, sample_size=50)

        # Verify
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 50
        assert "empty_col" not in result.columns  # Should be removed
        # Note: mostly_empty_col may not be removed depending on sampling

    def test_get_raw_data_sample_with_file_filter(self):
        """Test raw data sampling with file filtering."""
        test_data, ref_data = self.create_test_data()

        # Test
        result = get_raw_data_sample(
            test_data, ref_data, sample_size=50, selected_filenames=["test.fit"]
        )

        # Verify
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Should only contain test.fit data
        if not result.empty:
            assert all(result["filename"] == "test.fit")

    def test_get_raw_data_sample_empty_filter(self):
        """Test raw data sampling with empty file filter."""
        test_data, ref_data = self.create_test_data()

        # Test
        result = get_raw_data_sample(
            test_data, ref_data, sample_size=50, selected_filenames=["nonexistent.fit"]
        )

        # Verify (should return empty DataFrame)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
