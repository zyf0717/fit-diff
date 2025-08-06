import logging
from typing import List

import pandas as pd
import shinyswatch
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
    get_bias_agreement_stats,
    get_correlation_stats,
    get_error_magnitude_stats,
    get_file_information,
    get_raw_data_sample,
    prepare_data_for_analysis,
    process_fit,
)

logger = logging.getLogger(__name__)


def server(input: Inputs, output: Outputs, session: Session):
    # Enable dynamic theme switching
    shinyswatch.theme_picker_server()

    @reactive.Effect
    @reactive.event(input.logout)
    async def _():
        await session.send_custom_message("logout", {})

    @render.ui
    def main_content():
        # Only render main content when at least one file from each type is uploaded
        test_files = input.testFileUpload()
        ref_files = input.refFileUpload()

        if not test_files or not ref_files:
            return ui.div(
                ui.p(
                    "Please upload at least one test file and one reference file to begin."
                ),
                style="text-align: center; margin-top: 50px; color: #666;",
            )

        return ui.div(
            # Navigation bar with different panels
            ui.navset_bar(
                ui.nav_panel(
                    "Analysis",
                    ui.card(
                        ui.card_header("Settings"),
                        ui.layout_columns(
                            ui.output_ui("testFileSelector"),
                            ui.output_ui("refFileSelector"),
                            ui.output_ui("comparisonMetricSelector"),
                            ui.output_ui("outlierRemovalSelector"),
                            col_widths=[3, 3, 3, 3],
                        ),
                        ui.layout_columns(
                            ui.input_slider(
                                id="shift_seconds",
                                label="Shift test data (seconds):",
                                min=-15,
                                max=15,
                                value=0,
                                step=1,
                            ),
                            ui.output_ui("analysisWindow"),
                            col_widths=[3, 6],
                        ),
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("File Statistics"),
                            ui.output_data_frame("basicStatsTable"),
                        ),
                        ui.card(
                            ui.card_header("Metric Visualization"),
                            output_widget("metricPlot"),
                        ),
                        col_widths=[3, 9],
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Bias & Agreement"),
                            ui.output_data_frame("biasAgreementTable"),
                        ),
                        ui.card(
                            ui.card_header("Error Magnitude"),
                            ui.output_data_frame("errorMagnitudeTable"),
                        ),
                        ui.card(
                            ui.card_header("Correlation"),
                            ui.output_data_frame("correlationTable"),
                        ),
                        col_widths=[4, 4, 4],
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Error Distribution Histogram"),
                            output_widget("errorHistogramPlot"),
                        ),
                        ui.card(
                            ui.card_header("Bland-Altman Plot"),
                            output_widget("blandAltmanPlot"),
                        ),
                        col_widths=[6, 6],
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
                ),
                ui.nav_panel(
                    "Data Info",
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
                ),
                title="FIT File Comparison Tool",
            ),
        )

    @render.ui
    def testFileSelector():
        test_files = input.testFileUpload()
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
        ref_files = input.refFileUpload()
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

    @render.ui
    def analysisWindow():
        try:
            range_info = _get_elapsed_seconds_range()
            if not range_info:
                return None
            return ui.layout_columns(
                ui.input_numeric(
                    id="analysis_window_start",
                    label="Analysis window start (elapsed seconds):",
                    value=range_info.get("min", 0),
                    min=range_info.get("min", 0),
                    max=range_info.get("max", 60),
                    step=1,
                ),
                ui.input_numeric(
                    id="analysis_window_end",
                    label="Analysis window end (elapsed seconds):",
                    value=range_info.get("max", 60),
                    min=range_info.get("min", 0),
                    max=range_info.get("max", 60),
                    step=1,
                ),
                col_widths=[6, 6],
            )
        except Exception as e:
            logger.error(f"Error in analysisWindow: {e}")
            return None

    @reactive.Calc
    def _process_test_device_files():
        test_files: List[FileInfo] = input.testFileUpload()
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
        ref_files: List[FileInfo] = input.refFileUpload()
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
    def _get_prepared_data():
        try:
            metric = (
                input.comparison_metric()
                if hasattr(input, "comparison_metric")
                else "heart_rate"
            )

            # Get outlier removal methods
            outlier_methods = []
            if hasattr(input, "outlier_removal"):
                outlier_methods = input.outlier_removal() or []

            # Get raw data first
            raw_data = _all_fit_data()
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
            if input.shift_seconds() != 0:
                test_data["elapsed_seconds"] += input.shift_seconds()
                test_data["timestamp"] = test_data["timestamp"] + pd.to_timedelta(
                    input.shift_seconds(), unit="s"
                )

            # Apply outlier removal if specified
            if outlier_methods:
                result = apply_outlier_removal(
                    test_data, ref_data, metric, outlier_methods
                )
                if result is None:
                    return pd.DataFrame(), pd.DataFrame()
                test_data, ref_data = result

            return test_data, ref_data
        except Exception as e:
            logger.error(f"Error in _get_prepared_data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    @reactive.Calc
    def _get_elapsed_seconds_range():
        try:
            prepared_data = _get_prepared_data()
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
            logger.error(f"Error in _get_elapsed_seconds_range: {e}")
            return {"min": 0, "max": 60, "default": (0, 60)}

    @reactive.Calc
    def _get_trimmed_data():
        try:
            prepared_data = _get_prepared_data()
            if not prepared_data or len(prepared_data) != 2:
                return pd.DataFrame(), pd.DataFrame()

            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return pd.DataFrame(), pd.DataFrame()

            # Get analysis window from numeric inputs
            start = (
                input.analysis_window_start()
                if hasattr(input, "analysis_window_start")
                else None
            )
            end = (
                input.analysis_window_end()
                if hasattr(input, "analysis_window_end")
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
            logger.error(f"Error in _get_trimmed_data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    @render_widget
    def metricPlot():
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return None
            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return None
            metric = (
                input.comparison_metric()
                if hasattr(input, "comparison_metric")
                else "heart_rate"
            )
            return create_metric_plot(test_data, ref_data, metric)
        except Exception as e:
            logger.error(f"Error in metricPlot: {e}")
            return None

    @render_widget
    def errorHistogramPlot():
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return None
            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return None
            metric = (
                input.comparison_metric()
                if hasattr(input, "comparison_metric")
                else "heart_rate"
            )

            return create_error_histogram(test_data, ref_data, metric)
        except Exception as e:
            logger.error(f"Error in errorHistogramPlot: {e}")
            return None

    @render_widget
    def blandAltmanPlot():
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return None
            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return None
            metric = (
                input.comparison_metric()
                if hasattr(input, "comparison_metric")
                else "heart_rate"
            )

            return create_bland_altman_plot(test_data, ref_data, metric)
        except Exception as e:
            logger.error(f"Error in blandAltmanPlot: {e}")
            return None

    @render_widget
    def rollingErrorPlot():
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return None
            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return None
            metric = (
                input.comparison_metric()
                if hasattr(input, "comparison_metric")
                else "heart_rate"
            )

            # Get window size from slider input
            window_size = (
                input.rolling_window_size()
                if hasattr(input, "rolling_window_size")
                else 50
            )

            return create_rolling_error_plot(test_data, ref_data, metric, window_size)
        except Exception as e:
            logger.error(f"Error in rollingErrorPlot: {e}")
            return None

    @render.data_frame
    def basicStatsTable():
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return pd.DataFrame()
            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return pd.DataFrame()
            return calculate_basic_stats(test_data, ref_data, input.comparison_metric())
        except Exception as e:
            logger.error(f"Error in basicStatsTable: {e}")
            return pd.DataFrame()

    @render.data_frame
    def biasAgreementTable():
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return pd.DataFrame()
            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return pd.DataFrame()
            return get_bias_agreement_stats(
                test_data, ref_data, input.comparison_metric()
            )
        except Exception as e:
            logger.error(f"Error in biasAgreementTable: {e}")
            return pd.DataFrame()

    @render.data_frame
    def errorMagnitudeTable():
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return pd.DataFrame()
            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return pd.DataFrame()
            return get_error_magnitude_stats(
                test_data, ref_data, input.comparison_metric()
            )
        except Exception as e:
            logger.error(f"Error in errorMagnitudeTable: {e}")
            return pd.DataFrame()

    @render.data_frame
    def correlationTable():
        try:
            prepared_data = _get_trimmed_data()
            if not prepared_data or len(prepared_data) != 2:
                return pd.DataFrame()
            test_data, ref_data = prepared_data
            if test_data.empty or ref_data.empty:
                return pd.DataFrame()
            return get_correlation_stats(test_data, ref_data, input.comparison_metric())
        except Exception as e:
            logger.error(f"Error in correlationTable: {e}")
            return pd.DataFrame()

    @render.data_frame
    def fileInfoTable():
        # Use raw data for file information, not prepared/filtered data
        fit_data = _all_fit_data()
        if not fit_data or (
            isinstance(fit_data, tuple) and (fit_data[0].empty or fit_data[1].empty)
        ):
            return pd.DataFrame()
        test_data, ref_data = fit_data
        result = get_file_information(test_data, ref_data)
        if result is not None:
            return render.DataGrid(result, selection_mode="rows")
        return pd.DataFrame()

    @render.data_frame
    def rawDataTable():
        # Use raw data instead of prepared/filtered data
        fit_data = _all_fit_data()
        if not fit_data or (
            isinstance(fit_data, tuple) and (fit_data[0].empty or fit_data[1].empty)
        ):
            return pd.DataFrame()

        test_data, ref_data = fit_data

        sample_size = (
            input.raw_data_sample_size()
            if hasattr(input, "raw_data_sample_size")
            else 100
        )

        # Get selected rows from file information table
        selected_filenames = None
        try:
            # Check if fileInfoTable has selections
            file_info_selection = input.fileInfoTable_selected_rows()
            if file_info_selection:
                # Get the file information data to extract filenames
                file_info_df = get_file_information(test_data, ref_data)
                if file_info_df is not None and not file_info_df.empty:
                    # Extract filenames from selected rows (0-indexed)
                    selected_rows = [int(i) for i in file_info_selection]
                    selected_filenames = file_info_df.iloc[selected_rows][
                        "filename"
                    ].tolist()
        except (AttributeError, KeyError, IndexError):
            # If no selection or error accessing selection, show all files
            selected_filenames = None

        return get_raw_data_sample(test_data, ref_data, sample_size, selected_filenames)
