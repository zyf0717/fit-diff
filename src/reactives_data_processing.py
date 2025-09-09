"""Data processing reactive functions for the FIT file comparison app."""

import logging

import pandas as pd
from shiny import Inputs, reactive

from src.utils import prepare_data_for_analysis, remove_outliers

logger = logging.getLogger(__name__)


def create_data_processing_reactives(inputs: Inputs, file_reactives: dict):
    """Create data processing reactive functions."""

    @reactive.Calc
    def _get_shifted_data():
        try:
            metric = _get_comparison_metric()

            # Get raw data first
            raw_data = file_reactives["_all_fit_data"]()
            if not raw_data or len(raw_data) != 2:
                return pd.DataFrame(), pd.DataFrame()

            test_raw, ref_raw = raw_data
            if test_raw.empty or ref_raw.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Prepare data for analysis
            prepared_data = prepare_data_for_analysis((test_raw, ref_raw), metric)
            if prepared_data is None:
                return pd.DataFrame(), pd.DataFrame()

            test_data, ref_data = prepared_data

            # Shift test data time if specified
            if inputs.shift_seconds() != 0:
                test_data["elapsed_seconds"] += inputs.shift_seconds()
                test_data["timestamp"] = test_data["timestamp"] + pd.to_timedelta(
                    inputs.shift_seconds(), unit="s"
                )

            return test_data, ref_data
        except Exception as e:
            logger.error("Error in _get_shifted_data: %s", e)
            return pd.DataFrame(), pd.DataFrame()

    @reactive.Calc
    def _get_elapsed_seconds_range():
        try:
            prepared_data = _get_shifted_data()
            if not prepared_data or len(prepared_data) != 2:
                return {"min": 0, "max": 60, "default": (0, 60)}

            test_data_df, ref_data_df = prepared_data

            # Handle empty DataFrames
            if test_data_df.empty and ref_data_df.empty:
                return {"min": 0, "max": 60, "default": (0, 60)}

            # Get min/max from both DataFrames, ignoring empty ones, using list comprehensions
            min_vals = [
                df["elapsed_seconds"].min()
                for df in [test_data_df, ref_data_df]
                if not df.empty and "elapsed_seconds" in df.columns
            ]
            max_vals = [
                df["elapsed_seconds"].max()
                for df in [test_data_df, ref_data_df]
                if not df.empty and "elapsed_seconds" in df.columns
            ]
            if not min_vals or not max_vals:
                return {"min": 0, "max": 60, "default": (0, 60)}

            min_value = int(min(min_vals))
            max_value = int(max(max_vals))
            default_value = (min_value, max_value)
            return {"min": min_value, "max": max_value, "default": default_value}
        except Exception as e:
            logger.error("Error in _get_elapsed_seconds_range: %s", e)
            return {"min": 0, "max": 60, "default": (0, 60)}

    @reactive.Calc
    def _get_trimmed_data():
        try:
            prepared_data = _get_shifted_data()
            if not prepared_data or len(prepared_data) != 2:
                return pd.DataFrame(), pd.DataFrame()

            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Get analysis window from numeric inputs
            start = (
                inputs.analysis_window_start()
                if hasattr(inputs, "analysis_window_start")
                else None
            )
            end = (
                inputs.analysis_window_end()
                if hasattr(inputs, "analysis_window_end")
                else None
            )

            if start is None or end is None:
                # If no analysis window is set, return the full data
                return test_data, ref_data

            # Trim data to the specified window
            test_data = test_data[
                (test_data["elapsed_seconds"] >= start)
                & (test_data["elapsed_seconds"] <= end)
            ]
            ref_data = ref_data[
                (ref_data["elapsed_seconds"] >= start)
                & (ref_data["elapsed_seconds"] <= end)
            ]

            return test_data, ref_data
        except Exception as e:
            logger.error("Error in _get_trimmed_data: %s", e)
            return pd.DataFrame(), pd.DataFrame()

    @reactive.Calc
    def _get_aligned_data_with_outlier_removal():
        """Get aligned test and reference data with outlier removal applied to differences."""
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return None

            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return None

            metric = _get_comparison_metric()
            if metric not in test_data.columns or metric not in ref_data.columns:
                return None

            # Get clean data
            test_clean = test_data[["timestamp", "elapsed_seconds", metric]].dropna()
            ref_clean = ref_data[["timestamp", "elapsed_seconds", metric]].dropna()

            # Merge on timestamp to align the data properly
            aligned_df = pd.merge(
                test_clean, ref_clean, on="timestamp", suffixes=("_test", "_ref")
            )

            if aligned_df.empty:
                return None

            # Calculate differences for outlier removal
            aligned_df["difference"] = (
                aligned_df[f"{metric}_test"] - aligned_df[f"{metric}_ref"]
            )

            # Get outlier removal methods
            outlier_methods = []
            if hasattr(inputs, "outlier_removal") and inputs.outlier_removal():
                outlier_methods = inputs.outlier_removal()

            # Apply outlier removal to differences if specified
            if outlier_methods:
                aligned_df = remove_outliers(aligned_df, "difference", outlier_methods)

            # Remove the temporary difference column
            aligned_df = aligned_df.drop(columns=["difference"])

            return aligned_df if not aligned_df.empty else None
        except Exception as e:
            logger.error("Error in _get_aligned_data_with_outlier_removal: %s", e)
            return None

    # Helper functions for error handling and data validation
    def _safe_execute(func, func_name, default_return=None):
        """
        Execute function safely with consistent error handling.

        Args:
            func: Function to execute
            func_name: Name for logging purposes
            default_return: Default value to return on error

        Returns:
            Function result or default_return on error
        """
        try:
            return func()
        except Exception as e:
            logger.error("Error in %s: %s", func_name, e)
            return default_return

    def _get_validated_aligned_data():
        """Get validated aligned data or return None if invalid."""
        aligned_data = _get_aligned_data_with_outlier_removal()
        if aligned_data is None or aligned_data.empty:
            return None
        return aligned_data

    def _get_comparison_metric():
        """Get comparison metric with fallback to 'heart_rate'."""
        return (
            inputs.comparison_metric()
            if hasattr(inputs, "comparison_metric")
            else "heart_rate"
        )

    return {
        "_get_shifted_data": _get_shifted_data,
        "_get_elapsed_seconds_range": _get_elapsed_seconds_range,
        "_get_trimmed_data": _get_trimmed_data,
        "_get_aligned_data_with_outlier_removal": _get_aligned_data_with_outlier_removal,
        "_safe_execute": _safe_execute,
        "_get_validated_aligned_data": _get_validated_aligned_data,
        "_get_comparison_metric": _get_comparison_metric,
    }
