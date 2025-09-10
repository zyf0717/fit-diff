"""UI reactive functions for the FIT file comparison app."""

import logging

from shiny import Inputs, reactive, render, ui
from shinywidgets import output_widget

logger = logging.getLogger(__name__)


def create_ui_reactives(inputs: Inputs, file_reactives: dict, data_reactives: dict):
    """Create UI reactive functions."""

    @render.ui
    def mainContent():
        # Only render main content when at least one file from each type is uploaded
        test_files = inputs.testFileUpload()
        ref_files = inputs.refFileUpload()

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
                    ui.layout_columns(
                        ui.output_ui("testFileSelector"),
                        ui.output_ui("refFileSelector"),
                        ui.output_ui("comparisonMetricSelector"),
                        ui.output_ui("outlierRemovalSelector"),
                        col_widths=[3, 3, 3, 3],
                    ),
                    ui.layout_columns(
                        ui.input_numeric(
                            id="trim_from_start",
                            label="Trim from start (seconds):",
                            value=0,
                            min=0,
                            step=1,
                        ),
                        ui.input_numeric(
                            id="trim_from_end",
                            label="Trim from end (seconds):",
                            value=0,
                            min=0,
                            step=1,
                        ),
                        ui.output_ui("shiftSecondsSelector"),
                        ui.input_select(
                            "auto_shift_method",
                            "Auto-shift by:",
                            choices=[
                                "None",
                                "Minimize MAE",
                                "Minimize MSE",
                                "Maximize Correlation",
                            ],
                            selected="None",
                        ),
                        col_widths=[3, 3, 3, 3],
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
                    ui.card(
                        ui.card_header("LLM Generated Summary"),
                        ui.layout_columns(
                            ui.input_action_button("llm_summary_regen", "Ask BotBot!"),
                            ui.output_markdown_stream(
                                "streamOutput", auto_scroll=False
                            ),
                            col_widths=[3, 9],
                        ),
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
        test_files = inputs.testFileUpload()
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
        ref_files = inputs.refFileUpload()
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
        choices = file_reactives["_get_common_metrics"]()
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
                "remove_iqr": "IQR outliers (1.5 × IQR)",
                "remove_zscore": "Z-score outliers (|z| > 3)",
                "remove_percentile": "Percentile outliers (< 1% or > 99%)",
            },
            selected=[],
            multiple=True,
        )

    @render.ui
    def shiftSecondsSelector():
        return ui.input_numeric(
            id="shift_seconds",
            label="Shift test data (seconds):",
            value=0,
            step=1,
        )

    return {
        "mainContent": mainContent,
        "testFileSelector": testFileSelector,
        "refFileSelector": refFileSelector,
        "comparisonMetricSelector": comparisonMetricSelector,
        "outlierRemovalSelector": outlierRemovalSelector,
        "shiftSecondsSelector": shiftSecondsSelector,
    }
