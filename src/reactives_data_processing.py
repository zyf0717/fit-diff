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

            # Comparison metric (needed for range filters)
            metric = _get_comparison_metric()

            # Metric range limiting & special HR≈cadence logic applied BEFORE shifting
            metric_range_mode = _safe_get_input(
                lambda: (
                    inputs.metric_range() if hasattr(inputs, "metric_range") else "All"
                ),
                default="All",
            )
            lower = _safe_get_input(
                lambda: (
                    inputs.metric_range_lower()
                    if hasattr(inputs, "metric_range_lower")
                    else None
                ),
                default=None,
            )
            upper = _safe_get_input(
                lambda: (
                    inputs.metric_range_upper()
                    if hasattr(inputs, "metric_range_upper")
                    else None
                ),
                default=None,
            )

            # Generic numeric range filtering on selected metric
            if (
                metric_range_mode == "Range"
                and metric in test_data.columns
                and metric in ref_data.columns
            ):
                if lower is not None or upper is not None:
                    before_counts = (len(test_data), len(ref_data))
                    if lower is not None:
                        test_data = test_data[test_data[metric] >= lower]
                        ref_data = ref_data[ref_data[metric] >= lower]
                    if upper is not None:
                        test_data = test_data[test_data[metric] <= upper]
                        ref_data = ref_data[ref_data[metric] <= upper]
                    if test_data.empty or ref_data.empty:
                        logger.info(
                            "Metric range filtering (%s) resulted in empty dataset (before counts test=%s ref=%s, bounds: %s - %s)",
                            metric,
                            before_counts[0],
                            before_counts[1],
                            lower,
                            upper,
                        )
                        return pd.DataFrame(), pd.DataFrame()

            # Special case: heart rate ≈ step cadence
            if (
                metric_range_mode == "HR ≈ step cadence"
                and "heart_rate" in test_data.columns
                and "cadence" in test_data.columns
            ):
                tol_lower = lower if lower is not None else 0
                tol_upper = upper if upper is not None else 0

                target_hr = test_data["cadence"] * 2
                mask = (test_data["heart_rate"] >= target_hr + tol_lower) & (
                    test_data["heart_rate"] <= target_hr + tol_upper
                )
                before_n = len(test_data)
                test_data = test_data[mask]
                logger.info(
                    "Applied HR ≈ 2 × cadence filter pre-shift: kept %s / %s rows (tol %s %s)",
                    len(test_data),
                    before_n,
                    tol_lower,
                    tol_upper,
                )
                if test_data.empty:
                    logger.info(
                        "Metric range filtering (HR ≈ step cadence) resulted in empty dataset"
                    )
                    return pd.DataFrame(), pd.DataFrame()

            # Time range filtering (absolute elapsed_seconds window)
            raw_start = _safe_get_input(
                lambda: (
                    inputs.time_range_start()
                    if hasattr(inputs, "time_range_start")
                    else None
                ),
                default=None,
            )
            raw_end = _safe_get_input(
                lambda: (
                    inputs.time_range_end()
                    if hasattr(inputs, "time_range_end")
                    else None
                ),
                default=None,
            )

            def _coerce(v):
                try:
                    if v is None or v == "":
                        return None
                    return int(v)
                except (TypeError, ValueError):
                    return None

            start_time = _coerce(raw_start)
            end_time = _coerce(raw_end)

            # If neither bound supplied, keep full range
            if start_time is None and end_time is None:
                return test_data, ref_data

            # Derive dataset max if only start provided etc.
            overall_min = min(
                test_data["elapsed_seconds"].min(), ref_data["elapsed_seconds"].min()
            )
            overall_max = max(
                test_data["elapsed_seconds"].max(), ref_data["elapsed_seconds"].max()
            )

            if start_time is None:
                start_time = overall_min
            if end_time is None:
                end_time = overall_max

            before_counts = (len(test_data), len(ref_data))
            test_data = test_data[
                (test_data["elapsed_seconds"] >= start_time)
                & (test_data["elapsed_seconds"] <= end_time)
            ]
            ref_data = ref_data[
                (ref_data["elapsed_seconds"] >= start_time)
                & (ref_data["elapsed_seconds"] <= end_time)
            ]
            logger.info(
                "Applied time range filter: [%s, %s] seconds (kept test %s/%s, ref %s/%s)",
                start_time,
                end_time,
                len(test_data),
                before_counts[0],
                len(ref_data),
                before_counts[1],
            )
            if test_data.empty or ref_data.empty:
                logger.info("Time range filtering resulted in empty dataset")
                return pd.DataFrame(), pd.DataFrame()
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
                    else "None (manual)"
                ),
                default="None (manual)",
            )

            if "None" in auto_shift_method:
                return

            test_data, ref_data = _get_trimmed_data()
            if test_data.empty or ref_data.empty:
                return

            metric = _get_comparison_metric()
            seconds_to_shift = determine_optimal_shift(
                test_data, ref_data, metric, auto_shift_method
            )
            optimal_shift = seconds_to_shift if seconds_to_shift is not None else 0

            if hasattr(inputs, "shift_seconds"):
                try:
                    ui.update_numeric("shift_seconds", value=optimal_shift)
                    logger.info(
                        "Auto-set shift_seconds to %s based on %s",
                        optimal_shift,
                        auto_shift_method,
                    )
                except Exception as set_error:  # noqa: BLE001
                    logger.warning("Could not set shift_seconds input: %s", set_error)
        except Exception as e:  # noqa: BLE001
            logger.error("Error in _set_optimal_shift: %s", e, exc_info=True)

    @reactive.Calc
    def _get_shifted_data():
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

        try:
            if shift_seconds != 0:
                test_data = test_data.copy()
                test_data["elapsed_seconds"] += shift_seconds
                test_data["timestamp"] = test_data["timestamp"] + pd.to_timedelta(
                    shift_seconds, unit="s"
                )
            return test_data, ref_data
        except Exception as e:  # noqa: BLE001
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
    }
