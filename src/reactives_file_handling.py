"""File handling reactive functions for the FIT file comparison app."""

import logging
from typing import List

import pandas as pd
from shiny import Inputs, Session, reactive
from shiny.types import FileInfo, SilentException

from src.utils import process_file

logger = logging.getLogger(__name__)


def create_file_handling_reactives(
    inputs: Inputs,
    session: Session,
    local_pair_override=None,
):
    """Create file handling reactive functions."""

    def _get_local_pair_override():
        if local_pair_override is None:
            return None
        pair_override = local_pair_override.get()
        return pair_override if isinstance(pair_override, dict) else None

    def _get_uploaded_files(input_attr: str) -> List[FileInfo]:
        try:
            input_obj = getattr(inputs, input_attr, None)
            if input_obj is None:
                return []
            return input_obj() or []
        except SilentException:
            return []

    def _get_override_file_data(device_type: str):
        override = _get_local_pair_override()
        if not override:
            return None

        filename = override.get(f"{device_type}_filename")
        device_df = override.get(f"{device_type}_df")
        if not filename or not isinstance(device_df, pd.DataFrame):
            return None
        return {filename: device_df.copy()}

    def _process_device_files(file_infos: List[FileInfo], selected_attr: str) -> dict:
        """Process uploaded FIT files based on a selected_attr list in input."""
        if not file_infos:
            return {}
        # Determine selected file names
        if hasattr(inputs, selected_attr) and getattr(inputs, selected_attr)():
            selected_files = getattr(inputs, selected_attr)()
        else:
            selected_files = [fi["name"] for fi in file_infos]
        device_data = {}
        for file_info in file_infos:
            if file_info["name"] not in selected_files:
                continue
            try:
                record_df = process_file(file_info["datapath"])["records"]
                device_data[file_info["name"]] = record_df
            except Exception as e:
                logger.error(
                    "Error processing %s file %s: %s",
                    selected_attr,
                    file_info["name"],
                    e,
                )
                device_data[file_info["name"]] = f"Error: {str(e)}"
        return device_data

    @reactive.Calc
    def _active_test_file_names():
        override = _get_local_pair_override()
        if override and override.get("test_filename"):
            return [override["test_filename"]]
        return [
            file_info["name"] for file_info in _get_uploaded_files("testFileUpload")
        ]

    @reactive.Calc
    def _active_ref_file_names():
        override = _get_local_pair_override()
        if override and override.get("ref_filename"):
            return [override["ref_filename"]]
        return [file_info["name"] for file_info in _get_uploaded_files("refFileUpload")]

    @reactive.Calc
    def _has_active_files():
        return bool(_active_test_file_names() and _active_ref_file_names())

    @reactive.Calc
    def _preferred_comparison_metric():
        override = _get_local_pair_override()
        if not override:
            return None
        return override.get("metric")

    @reactive.Calc
    def _preferred_auto_shift_method():
        override = _get_local_pair_override()
        if not override:
            return None
        return override.get("auto_shift_method")

    @reactive.Calc
    def _process_test_device_files():
        override_data = _get_override_file_data("test")
        if override_data is not None:
            return override_data
        try:
            return _process_device_files(inputs.testFileUpload(), "selected_test_files")
        except SilentException:
            return {}
        except Exception as e:
            logger.error("Error processing test files: %s", e, exc_info=True)
            return {}

    @reactive.Calc
    def _process_reference_device_files():
        override_data = _get_override_file_data("ref")
        if override_data is not None:
            return override_data
        try:
            return _process_device_files(inputs.refFileUpload(), "selected_ref_files")
        except SilentException:
            return {}
        except Exception as e:
            logger.error("Error processing reference files: %s", e, exc_info=True)
            return {}

    @reactive.Calc
    def _all_fit_data():
        test_data = _process_test_device_files()
        ref_data = _process_reference_device_files()

        # Collect DataFrames and ensure filename is preserved
        all_test_data = []
        for filename, df in test_data.items():
            if isinstance(df, pd.DataFrame):
                # Ensure filename column is correctly set
                df_copy = df.copy()
                df_copy["filename"] = str(
                    filename
                )  # Use the dictionary key as filename
                all_test_data.append(df_copy)

        all_ref_data = []
        for filename, df in ref_data.items():
            if isinstance(df, pd.DataFrame):
                # Ensure filename column is correctly set
                df_copy = df.copy()
                df_copy["filename"] = str(
                    filename
                )  # Use the dictionary key as filename
                all_ref_data.append(df_copy)

        if not all_test_data or not all_ref_data:
            logger.warning(
                "No FIT data available from either test or reference devices."
            )
            # Return two empty DataFrames to avoid ValueError
            return pd.DataFrame(), pd.DataFrame()

        test_data_df = pd.concat(all_test_data, ignore_index=True)
        ref_data_df = pd.concat(all_ref_data, ignore_index=True)

        return test_data_df, ref_data_df

    @reactive.Calc
    def _get_common_metrics():
        fit_data = _all_fit_data()
        if len(fit_data) != 2:
            return []

        test_df, ref_df = fit_data
        if test_df.empty or ref_df.empty:
            return []

        test_data_columns = set(test_df.columns)
        ref_data_columns = set(ref_df.columns)

        common_metrics = test_data_columns.intersection(ref_data_columns)
        if "timestamp" in common_metrics:
            common_metrics.remove("timestamp")
        if "filename" in common_metrics:
            common_metrics.remove("filename")
        return sorted(list(common_metrics))

    return {
        "_process_test_device_files": _process_test_device_files,
        "_process_reference_device_files": _process_reference_device_files,
        "_all_fit_data": _all_fit_data,
        "_get_common_metrics": _get_common_metrics,
        "_active_test_file_names": _active_test_file_names,
        "_active_ref_file_names": _active_ref_file_names,
        "_has_active_files": _has_active_files,
        "_preferred_comparison_metric": _preferred_comparison_metric,
        "_preferred_auto_shift_method": _preferred_auto_shift_method,
    }
