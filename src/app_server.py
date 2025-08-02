import logging
from typing import List

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shinywidgets import output_widget, render_widget

from src.utils import (
    calculate_basic_stats,
    calculate_diff_stats,
    create_bland_altman_plot,
    create_combined_df_with_outlier_removal,
    create_error_histogram,
    create_metric_plot,
    create_rolling_error_plot,
    process_fit,
)

logger = logging.getLogger(__name__)


def server(input: Inputs, output: Outputs, session: Session):
    @render.ui
    def main_content():
        # Only render main content when at least one file from each type is uploaded
        test_files = input.test_file_upload()
        ref_files = input.ref_file_upload()

        if not test_files or not ref_files:
            return ui.div(
                ui.p(
                    "Please upload at least one test file and one reference file to begin."
                ),
                style="text-align: center; margin-top: 50px; color: #666;",
            )

        return ui.div(
            ui.layout_columns(
                ui.output_ui("testFileSelector"),
                ui.output_ui("refFileSelector"),
                ui.output_ui("comparisonMetricSelector"),
                ui.output_ui("outlierRemovalSelector"),
                col_widths=[3, 3, 3, 3],
            ),
            ui.card(
                ui.card_header("File Statistics"),
                ui.output_data_frame("basicStatsTable"),
            ),
            ui.card(
                ui.card_header("Metric Visualization"),
                output_widget("metricPlot"),
            ),
            ui.card(
                ui.card_header("Comparison Statistics"),
                ui.output_data_frame("diffStatsTable"),
            ),
            ui.card(
                ui.card_header("Error Distribution Histogram"),
                output_widget("errorHistogramPlot"),
            ),
            ui.card(
                ui.card_header("Bland-Altman Plot"),
                output_widget("blandAltmanPlot"),
            ),
            ui.card(
                ui.card_header("Rolling Error / Time-Varying Bias"),
                output_widget("rollingErrorPlot"),
            ),
        )

    @render.ui
    def testFileSelector():
        test_files = input.test_file_upload()
        if not test_files:
            return None

        file_choices = {
            file_info["name"]: file_info["name"] for file_info in test_files
        }
        return ui.input_selectize(
            "selected_test_files",
            "Select test files to use:",
            choices=file_choices,
            selected=list(file_choices.keys()),  # Default to all selected
            multiple=True,
        )

    @render.ui
    def refFileSelector():
        ref_files = input.ref_file_upload()
        if not ref_files:
            return None

        file_choices = {file_info["name"]: file_info["name"] for file_info in ref_files}
        return ui.input_selectize(
            "selected_ref_files",
            "Select reference files to use:",
            choices=file_choices,
            selected=list(file_choices.keys()),  # Default to all selected
            multiple=True,
        )

    @render.ui
    def comparisonMetricSelector():
        choices = _get_common_metrics()
        if not choices:
            return None
        return ui.input_select(
            "comparison_metric",
            "Select comparison metric:",
            choices=choices,
            selected="heart_rate",
        )

    @render.ui
    def outlierRemovalSelector():
        return ui.input_selectize(
            "outlier_removal",
            "Outlier removal:",
            choices={
                "remove_zeros": "Remove zero values",
                "remove_iqr": "Remove IQR outliers (1.5 × IQR)",
                "remove_zscore": "Remove Z-score outliers (|z| > 3)",
                "remove_percentile": "Remove percentile outliers (< 1% or > 99%)",
            },
            selected=[],
            multiple=True,
        )

    @reactive.Calc
    def _process_test_device_files():
        test_files: List[FileInfo] = input.test_file_upload()
        if not test_files:
            return {}

        # Get selected test files
        selected_files = []
        if hasattr(input, "selected_test_files") and input.selected_test_files():
            selected_files = input.selected_test_files()
        else:
            # Default to all files if no selection made yet
            selected_files = [file_info["name"] for file_info in test_files]

        test_device_data = {}
        for file_info in test_files:
            # Only process selected files
            if file_info["name"] not in selected_files:
                continue

            try:
                uploaded_file_path = file_info["datapath"]
                _, record_df = process_fit(uploaded_file_path)
                test_device_data[file_info["name"]] = (
                    record_df  # Use record_df for analysis
                )
            except Exception as e:
                logger.error(
                    "Error processing test device file %s: %s",
                    file_info["name"],
                    str(e),
                )
                test_device_data[file_info["name"]] = f"Error: {str(e)}"
        return test_device_data

    @reactive.Calc
    def _process_reference_device_files():
        ref_files: List[FileInfo] = input.ref_file_upload()
        if not ref_files:
            return {}

        # Get selected reference files
        selected_files = []
        if hasattr(input, "selected_ref_files") and input.selected_ref_files():
            selected_files = input.selected_ref_files()
        else:
            # Default to all files if no selection made yet
            selected_files = [file_info["name"] for file_info in ref_files]

        ref_device_data = {}
        for file_info in ref_files:
            # Only process selected files
            if file_info["name"] not in selected_files:
                continue

            try:
                uploaded_file_path = file_info["datapath"]
                _, record_df = process_fit(uploaded_file_path)
                ref_device_data[file_info["name"]] = record_df
            except Exception as e:
                logger.error(
                    "Error processing reference device file %s: %s",
                    file_info["name"],
                    str(e),
                )
                ref_device_data[file_info["name"]] = f"Error: {str(e)}"
        return ref_device_data

    @reactive.Calc
    def _all_fit_data():
        test_data = _process_test_device_files()
        ref_data = _process_reference_device_files()
        all_test_data = [
            df for _, df in test_data.items() if isinstance(df, pd.DataFrame)
        ]
        all_ref_data = [
            df for _, df in ref_data.items() if isinstance(df, pd.DataFrame)
        ]

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
    def _get_combined_df():
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        # Get outlier removal methods
        outlier_methods = []
        if hasattr(input, "outlier_removal"):
            outlier_methods = input.outlier_removal() or []

        return create_combined_df_with_outlier_removal(
            _all_fit_data(), metric, outlier_methods
        )

    @render_widget
    def metricPlot():
        combined_df = _get_combined_df()
        if combined_df is None:
            return None
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )
        return create_metric_plot(combined_df, metric)

    @render_widget
    def errorHistogramPlot():
        combined_df = _get_combined_df()
        if combined_df is None:
            return None
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        return create_error_histogram(combined_df, metric)

    @render_widget
    def blandAltmanPlot():
        combined_df = _get_combined_df()
        if combined_df is None:
            return None
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        return create_bland_altman_plot(combined_df, metric)

    @render_widget
    def rollingErrorPlot():
        combined_df = _get_combined_df()
        if combined_df is None:
            return None
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        return create_rolling_error_plot(combined_df, metric)

    @render.data_frame
    def basicStatsTable():
        combined_df = _get_combined_df()
        if combined_df is None:
            return pd.DataFrame()
        return calculate_basic_stats(combined_df, input.comparison_metric())

    @render.data_frame
    def diffStatsTable():
        combined_df = _get_combined_df()
        if combined_df is None:
            return pd.DataFrame()
        return calculate_diff_stats(combined_df, input.comparison_metric())
