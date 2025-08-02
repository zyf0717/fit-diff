import logging
from typing import List

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shinywidgets import output_widget, render_widget

from src.utils import (
    apply_outlier_removal,
    calculate_basic_stats,
    create_bland_altman_plot,
    create_error_histogram,
    create_metric_plot,
    create_rolling_error_plot,
    get_error_metrics,
    get_file_information,
    get_raw_data_sample,
    get_significance_stats,
    prepare_data_for_analysis,
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
            ui.layout_columns(
                ui.card(
                    ui.card_header("Error Metrics & Bias"),
                    ui.output_data_frame("errorMetricsTable"),
                ),
                ui.card(
                    ui.card_header("Statistical Significance"),
                    ui.output_data_frame("significanceTable"),
                ),
                col_widths=[6, 6],
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
                ui.input_slider(
                    "rolling_window_size",
                    "Rolling window size (data points)",
                    min=10,
                    max=200,
                    value=50,
                    step=10,
                ),
                output_widget("rollingErrorPlot"),
            ),
            ui.card(
                ui.card_header("File Information"),
                ui.output_data_frame("fileInfoTable"),
            ),
            ui.card(
                ui.card_header("Raw Data Sample"),
                ui.input_slider(
                    "raw_data_sample_size",
                    "Sample size",
                    min=50,
                    max=500,
                    value=100,
                    step=50,
                ),
                ui.output_data_frame("rawDataTable"),
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
    def _get_prepared_data():
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        # Get outlier removal methods
        outlier_methods = []
        if hasattr(input, "outlier_removal"):
            outlier_methods = input.outlier_removal() or []

        # Prepare data for analysis
        prepared_data = prepare_data_for_analysis(_all_fit_data(), metric)
        if prepared_data is None:
            return None

        test_data, ref_data = prepared_data

        # Apply outlier removal if specified
        if outlier_methods:
            result = apply_outlier_removal(test_data, ref_data, metric, outlier_methods)
            if result is None:
                return None
            test_data, ref_data = result

        return test_data, ref_data

    @render_widget
    def metricPlot():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return None
        test_data, ref_data = prepared_data
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )
        return create_metric_plot(test_data, ref_data, metric)

    @render_widget
    def errorHistogramPlot():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return None
        test_data, ref_data = prepared_data
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        return create_error_histogram(test_data, ref_data, metric)

    @render_widget
    def blandAltmanPlot():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return None
        test_data, ref_data = prepared_data
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        return create_bland_altman_plot(test_data, ref_data, metric)

    @render_widget
    def rollingErrorPlot():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return None
        test_data, ref_data = prepared_data
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        # Get window size from slider input
        window_size = (
            input.rolling_window_size() if hasattr(input, "rolling_window_size") else 50
        )

        return create_rolling_error_plot(test_data, ref_data, metric, window_size)

    @render.data_frame
    def basicStatsTable():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return pd.DataFrame()
        test_data, ref_data = prepared_data
        return calculate_basic_stats(test_data, ref_data, input.comparison_metric())

    @render.data_frame
    def errorMetricsTable():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return pd.DataFrame()
        test_data, ref_data = prepared_data
        return get_error_metrics(test_data, ref_data, input.comparison_metric())

    @render.data_frame
    def significanceTable():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return pd.DataFrame()
        test_data, ref_data = prepared_data
        return get_significance_stats(test_data, ref_data, input.comparison_metric())

    @render.data_frame
    def fileInfoTable():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return pd.DataFrame()
        test_data, ref_data = prepared_data
        return get_file_information(test_data, ref_data)

    @render.data_frame
    def rawDataTable():
        prepared_data = _get_prepared_data()
        if prepared_data is None:
            return pd.DataFrame()
        test_data, ref_data = prepared_data
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )
        sample_size = (
            input.raw_data_sample_size()
            if hasattr(input, "raw_data_sample_size")
            else 100
        )
        return get_raw_data_sample(test_data, ref_data, metric, sample_size)
