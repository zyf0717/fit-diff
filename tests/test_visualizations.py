"""
Tests for visualization functions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.utils import (
    create_bland_altman_plot,
    create_error_histogram,
    create_metric_plot,
    create_rolling_error_plot,
)


class TestCreateMetricPlot:
    """Test cases for create_metric_plot function."""

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

    def test_create_metric_plot_success(self):
        """Test successful metric plot creation."""
        aligned_df = self.create_aligned_data()

        # Test
        result = create_metric_plot(aligned_df, "heart_rate")

        # Verify
        assert result is not None
        assert isinstance(result, go.Figure)
        assert len(result.data) >= 0  # Should have traces for test and reference

    def test_create_metric_plot_invalid_input(self):
        """Test with invalid input."""
        # Test with None
        assert create_metric_plot(None, "heart_rate") is None

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        assert create_metric_plot(empty_df, "heart_rate") is None

    def test_create_metric_plot_missing_columns(self):
        """Test with missing required columns."""
        # Data missing metric columns
        df = pd.DataFrame(
            {
                "elapsed_seconds_test": [0, 1, 2],
                "elapsed_seconds_ref": [0, 1, 2],
            }
        )

        result = create_metric_plot(df, "heart_rate")
        assert result is not None  # Should return empty figure
        assert isinstance(result, go.Figure)


class TestCreateErrorHistogram:
    """Test cases for create_error_histogram function."""

    def create_aligned_data(self):
        """Create test aligned data."""
        return pd.DataFrame(
            {
                "heart_rate_test": [150, 155, 160, 165, 170],
                "heart_rate_ref": [148, 153, 158, 163, 168],
            }
        )

    def test_create_error_histogram_success(self):
        """Test successful error histogram creation."""
        aligned_df = self.create_aligned_data()

        # Test
        result = create_error_histogram(aligned_df, "heart_rate")

        # Verify
        assert result is not None
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0  # Should have histogram trace

    def test_create_error_histogram_invalid_input(self):
        """Test with invalid input."""
        # Test with None
        assert create_error_histogram(None, "heart_rate") is None

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        assert create_error_histogram(empty_df, "heart_rate") is None


class TestCreateBlandAltmanPlot:
    """Test cases for create_bland_altman_plot function."""

    def create_aligned_data(self):
        """Create test aligned data."""
        return pd.DataFrame(
            {
                "heart_rate_test": [150, 155, 160, 165, 170],
                "heart_rate_ref": [148, 153, 158, 163, 168],
            }
        )

    def test_create_bland_altman_plot_success(self):
        """Test successful Bland-Altman plot creation."""
        aligned_df = self.create_aligned_data()

        # Test
        result = create_bland_altman_plot(aligned_df, "heart_rate")

        # Verify
        assert result is not None
        assert isinstance(result, go.Figure)
        assert len(result.data) > 0  # Should have scatter plot

        # Check for horizontal lines (mean, LOA)
        shapes_or_annotations = len(result.layout.shapes) + len(
            result.layout.annotations
        )
        assert shapes_or_annotations > 0  # Should have lines/annotations

    def test_create_bland_altman_plot_invalid_input(self):
        """Test with invalid input."""
        # Test with None
        assert create_bland_altman_plot(None, "heart_rate") is None

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        assert create_bland_altman_plot(empty_df, "heart_rate") is None


class TestCreateRollingErrorPlot:
    """Test cases for create_rolling_error_plot function."""

    def create_aligned_data(self):
        """Create test aligned data with more points for rolling analysis."""
        np.random.seed(42)
        n_points = 100
        return pd.DataFrame(
            {
                "heart_rate_test": np.random.normal(160, 10, n_points),
                "heart_rate_ref": np.random.normal(158, 10, n_points),
                "elapsed_seconds_test": range(n_points),
            }
        )

    def test_create_rolling_error_plot_success(self):
        """Test successful rolling error plot creation."""
        aligned_df = self.create_aligned_data()

        # Test
        result = create_rolling_error_plot(aligned_df, "heart_rate", window_size=20)

        # Verify
        assert result is not None
        assert isinstance(result, go.Figure)
        assert len(result.data) >= 1  # Should have rolling mean trace

    def test_create_rolling_error_plot_invalid_input(self):
        """Test with invalid input."""
        # Test with None
        assert create_rolling_error_plot(None, "heart_rate") is None

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        assert create_rolling_error_plot(empty_df, "heart_rate") is None

    def test_create_rolling_error_plot_small_window(self):
        """Test with small window size."""
        aligned_df = self.create_aligned_data()

        # Test with very small window
        result = create_rolling_error_plot(aligned_df, "heart_rate", window_size=5)

        # Verify
        assert result is not None
        assert isinstance(result, go.Figure)

    def test_create_rolling_error_plot_large_window(self):
        """Test with large window size."""
        aligned_df = self.create_aligned_data()

        # Test with window larger than data
        result = create_rolling_error_plot(aligned_df, "heart_rate", window_size=200)

        # Verify
        assert result is not None
        assert isinstance(result, go.Figure)
