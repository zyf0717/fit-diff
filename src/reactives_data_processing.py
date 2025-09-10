"""Data processing reactive functions for the FIT file comparison app."""

import logging

import pandas as pd
from shiny import Inputs, reactive, ui
from shiny.types import SilentException

from src.utils import (
    determine_optimal_shift,
    prepare_data_for_analysis,
    remove_outliers,
)

logger = logging.getLogger(__name__)


def create_data_processing_reactives(inputs: Inputs, file_reactives: dict):
    """Create data processing reactive functions."""

    def _safe_get_input(input_func, default=None):
        """Safely get input value, handling SilentException when input is not ready."""
        try:
            return input_func()
        except SilentException:
            return default
        except Exception as e:
            logger.warning("Error accessing input: %s, using default: %s", e, default)
            return default

    @reactive.Calc
    def _get_test_and_ref_data():
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
            return test_data, ref_data
        except Exception as e:
            logger.error("Error in _get_test_and_ref_data: %s", e, exc_info=True)
            return pd.DataFrame(), pd.DataFrame()

    @reactive.Calc
    def _get_trimmed_data():
        try:
            prepared_data = _get_test_and_ref_data()
            if not prepared_data or len(prepared_data) != 2:
                return pd.DataFrame(), pd.DataFrame()

            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Safely get trim parameters and treat non-int as 0
            trim_start = _safe_get_input(
                lambda: (
                    int(inputs.trim_from_start() or 0)
                    if hasattr(inputs, "trim_from_start")
                    else 0
                ),
                default=0,
            )
            trim_end = _safe_get_input(
                lambda: (
                    int(inputs.trim_from_end() or 0)
                    if hasattr(inputs, "trim_from_end")
                    else 0
                ),
                default=0,
            )

            if trim_start == 0 and trim_end == 0:
                # If no trimming is specified, return the full data
                return test_data, ref_data

            # Get the actual data range for each dataset
            test_min = test_data["elapsed_seconds"].min()
            test_max = test_data["elapsed_seconds"].max()
            ref_min = ref_data["elapsed_seconds"].min()
            ref_max = ref_data["elapsed_seconds"].max()

            # Calculate trim boundaries for each dataset
            test_start_boundary = test_min + trim_start
            test_end_boundary = test_max - trim_end
            ref_start_boundary = ref_min + trim_start
            ref_end_boundary = ref_max - trim_end

            # Trim data based on relative start/end positions
            test_data = test_data[
                (test_data["elapsed_seconds"] >= test_start_boundary)
                & (test_data["elapsed_seconds"] <= test_end_boundary)
            ]
            ref_data = ref_data[
                (ref_data["elapsed_seconds"] >= ref_start_boundary)
                & (ref_data["elapsed_seconds"] <= ref_end_boundary)
            ]

            return test_data, ref_data
        except Exception as e:
            logger.error("Error in _get_trimmed_data: %s", e, exc_info=True)
            return pd.DataFrame(), pd.DataFrame()

    @reactive.Effect
    def _set_optimal_shift():
        """Calculate and apply optimal shift when auto-shift method is enabled."""
        try:
            auto_shift_method = _safe_get_input(
                lambda: (
                    inputs.auto_shift_method()
                    if hasattr(inputs, "auto_shift_method")
                    else "None"
                ),
                default="None",
            )

            # If auto-shift is disabled, don't modify the input
            if auto_shift_method == "None":
                return

            test_data, ref_data = _get_trimmed_data()
            if test_data.empty or ref_data.empty:
                return

            metric = _get_comparison_metric()

            # Calculate optimal shift
            seconds_to_shift = determine_optimal_shift(
                test_data, ref_data, metric, auto_shift_method
            )
            optimal_shift = seconds_to_shift if seconds_to_shift is not None else 0

            # Update the shift_seconds input with the calculated value
            if hasattr(inputs, "shift_seconds"):
                try:
                    ui.update_numeric("shift_seconds", value=optimal_shift)
                except Exception as set_error:
                    logger.warning("Could not set shift_seconds input: %s", set_error)

        except Exception as e:
            logger.error("Error in _set_optimal_shift: %s", e, exc_info=True)

    @reactive.Calc
    def _get_shifted_data():
        # Safely get shift_seconds and treat non-int as 0
        shift_seconds = _safe_get_input(
            lambda: (
                int(inputs.shift_seconds() or 0)
                if hasattr(inputs, "shift_seconds")
                else 0
            ),
            default=0,
        )

        test_data, ref_data = _get_trimmed_data()
        if test_data.empty or ref_data.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Shift test data time if specified
        try:
            if shift_seconds != 0:
                test_data = (
                    test_data.copy()
                )  # Make a copy to avoid modifying original data
                test_data["elapsed_seconds"] += shift_seconds
                test_data["timestamp"] = test_data["timestamp"] + pd.to_timedelta(
                    shift_seconds, unit="s"
                )
            return test_data, ref_data
        except Exception as e:
            logger.error("Error in _get_shifted_data: %s", e, exc_info=True)
            return pd.DataFrame(), pd.DataFrame()

    @reactive.Calc
    def _get_aligned_data_with_outlier_removal():
        """Get aligned test and reference data with outlier removal applied to differences."""
        try:
            prepared_data = _get_shifted_data()
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
            outlier_methods = _safe_get_input(
                lambda: (
                    inputs.outlier_removal()
                    if hasattr(inputs, "outlier_removal")
                    else []
                ),
                default=[],
            )

            # Apply outlier removal to differences if specified
            if outlier_methods:
                aligned_df = remove_outliers(aligned_df, "difference", outlier_methods)

            # Remove the temporary difference column
            aligned_df = aligned_df.drop(columns=["difference"])

            return aligned_df if not aligned_df.empty else None
        except Exception as e:
            logger.error(
                "Error in _get_aligned_data_with_outlier_removal: %s", e, exc_info=True
            )
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
            logger.error("Error in %s: %s", func_name, e, exc_info=True)
            return default_return

    def _get_comparison_metric():
        """Get comparison metric with fallback to 'heart_rate'."""
        metric = _safe_get_input(
            lambda: (
                inputs.comparison_metric()
                if hasattr(inputs, "comparison_metric")
                else None
            ),
            default="heart_rate",
        )
        return metric or "heart_rate"

    return {
        "_get_shifted_data": _get_shifted_data,
        "_get_trimmed_data": _get_trimmed_data,
        "_get_aligned_data_with_outlier_removal": _get_aligned_data_with_outlier_removal,
        "_safe_execute": _safe_execute,
        "_get_comparison_metric": _get_comparison_metric,
        "_safe_get_input": _safe_get_input,
        "_set_optimal_shift": _set_optimal_shift,
        "_reset_auto_shift_on_manual_change": _reset_auto_shift_on_manual_change,
    }
