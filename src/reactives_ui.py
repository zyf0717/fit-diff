"""UI reactive functions for the FIT file comparison app."""

import logging

from shiny import Inputs, reactive, render, session, ui
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
                        ui.layout_columns(
                            ui.input_select(
                                "metric_range",
                                "Select metric range:",
                                choices=["All", "Range", "HR ≈ step cadence"],
                                selected="All",
                            ),
                            ui.input_numeric(
                                id="metric_range_lower",
                                label="Lower:",
                                value=None,
                                step=1,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                id="metric_range_upper",
                                label="Upper:",
                                value=None,
                                step=1,
                                update_on="blur",
                            ),
                            col_widths=[6, 3, 3],
                        ),
                        # ui.output_ui("outlierRemovalSelector"),
                        col_widths=[3, 3, 3, 3],
                    ),
                    ui.layout_columns(
                        ui.input_numeric(
                            id="time_range_start",
                            label="Start time (elapsed seconds):",
                            value=None,
                            min=0,
                            step=1,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            id="time_range_end",
                            label="End time (elapsed seconds):",
                            value=None,
                            min=0,
                            step=1,
                            update_on="blur",
                        ),
                        ui.output_ui("shiftSecondsSelector"),
                        ui.input_select(
                            "auto_shift_method",
                            "Auto-shift by:",
                            choices=[
                                "None (manual)",
                                "Minimize MAE",
                                "Minimize MSE",
                                "Maximize Concordance Correlation",
                                "Maximize Pearson Correlation",
                            ],
                            selected="None (manual)",
                        ),
                        col_widths=[3, 3, 3, 3],
                    ),
                    ui.hr(),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Metric Visualization"),
                            output_widget("metricPlot"),
                        ),
                        ui.card(
                            ui.card_header("Data Overview"),
                            ui.output_data_frame("basicStatsTable"),
                        ),
                        col_widths=[9, 3],
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Validity"),
                            ui.output_data_frame("validityTable"),
                        ),
                        ui.card(
                            ui.card_header("Precision"),
                            ui.output_data_frame("precisionTable"),
                        ),
                        ui.card(
                            ui.card_header("Reliability"),
                            ui.output_data_frame("reliabilityTable"),
                        ),
                        col_widths=[4, 4, 4],
                    ),
                    ui.card(
                        ui.card_header("LLM Generated Explanation"),
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
            update_on="blur",
        )

    @reactive.Effect
    async def _toggle_shift_seconds():
        sess = session.get_current_session()
        method = inputs.auto_shift_method()

        # No controlling value yet → do nothing
        if method is None:
            return

        async def send():
            await sess.send_custom_message(
                "toggle_disabled",
                {"id": "shift_seconds", "disabled": "None" not in method},
            )

        # If the widget hasn't rendered yet (conditional main UI), delay once
        if (
            getattr(inputs, "shift_seconds", None) is None
            or inputs.shift_seconds() is None
        ):
            sess.on_flushed(send, once=True)
        else:
            await send()

    # Disable and grey out metric range lower/upper inputs if 'All' is selected
    @reactive.Effect
    async def _toggle_metric_range_inputs():
        sess = session.get_current_session()
        selection = inputs.metric_range()

        if selection is None:
            return

        async def send():
            disabled = selection == "All"
            await sess.send_custom_message(
                "toggle_disabled",
                {"id": "metric_range_lower", "disabled": disabled},
            )
            await sess.send_custom_message(
                "toggle_disabled",
                {"id": "metric_range_upper", "disabled": disabled},
            )

        # Defer if the numeric inputs not yet rendered
        if (
            getattr(inputs, "metric_range_lower", None) is None
            or inputs.metric_range_lower() is None
            or getattr(inputs, "metric_range_upper", None) is None
            or inputs.metric_range_upper() is None
        ):
            sess.on_flushed(send, once=True)
        else:
            await send()

    return {
        "mainContent": mainContent,
        "testFileSelector": testFileSelector,
        "refFileSelector": refFileSelector,
        "comparisonMetricSelector": comparisonMetricSelector,
        "outlierRemovalSelector": outlierRemovalSelector,
        "shiftSecondsSelector": shiftSecondsSelector,
    }
