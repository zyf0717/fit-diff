"""File handling reactive functions for the FIT file comparison app."""

import logging
import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from shiny import Inputs, Session, reactive
from shiny.types import FileInfo, SilentException

from src.utils import process_file, process_multiple_files, read_catalogue

load_dotenv(override=True)

S3_BUCKET = os.getenv("S3_BUCKET")

logger = logging.getLogger(__name__)


def create_file_handling_reactives(inputs: Inputs, session: Session):
    """Create file handling reactive functions."""

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
    def _process_test_device_files():
        try:
            return _process_device_files(inputs.testFileUpload(), "selected_test_files")
        except SilentException:
            return {}
        except Exception as e:
            logger.error("Error processing test files: %s", e, exc_info=True)
            return {}

    @reactive.Calc
    def _process_reference_device_files():
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

    @reactive.Calc
    def _get_catalogue():
        df = read_catalogue()
        if df is None or df.empty:
            logger.warning("Catalogue DataFrame is empty or None")
            return pd.DataFrame()

        return df

    @reactive.Calc
    @reactive.event(inputs.loadBatchData)
    def _read_batch_files():
        selected_tags = (
            inputs.selectedBatchTags() if hasattr(inputs, "selectedBatchTags") else []
        )
        logger.info("Selected batch tags: %s", selected_tags)
        if not selected_tags:
            return None

        catalogue_df = _get_catalogue()
        if catalogue_df is None or catalogue_df.empty:
            logger.warning("_read_batch_files: catalogue_df is empty or None")
            return None

        # Filter catalogue by selected tags
        filtered_catalogue = catalogue_df[
            catalogue_df["tags"].isin(selected_tags)
        ].reset_index(drop=True)

        if filtered_catalogue.empty:
            logger.warning(
                "_read_batch_files: No files found for selected tags: %s",
                selected_tags,
            )
            return None

        # Read and process files in the filtered catalogue
        file_paths = [
            "s3://" + S3_BUCKET + "/" + key for key in filtered_catalogue["key"]
        ]
        s3_data = process_multiple_files(file_paths)

        return s3_data

    return {
        "_process_test_device_files": _process_test_device_files,
        "_process_reference_device_files": _process_reference_device_files,
        "_all_fit_data": _all_fit_data,
        "_get_common_metrics": _get_common_metrics,
        "_get_catalogue": _get_catalogue,
        "_read_batch_files": _read_batch_files,
    }
