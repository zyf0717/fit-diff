"""UI reactive functions for the FIT file comparison app."""

from faicons import icon_svg
from shiny import Inputs, reactive, render, session, ui
from shinywidgets import output_widget

from src.utils.data_processing import load_batch_tags


def create_ui_reactives(inputs: Inputs, file_reactives: dict, data_reactives: dict):
    """Create UI reactive functions."""

    @render.ui
    def benchmarkingContent():
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
            ui.layout_columns(
                ui.output_ui("testFileSelector"),
                ui.output_ui("refFileSelector"),
                ui.output_ui("comparisonMetricSelector"),
                ui.layout_columns(
                    ui.input_select(
                        "metric_range",
                        "Select metric range:",
                        choices=[
                            "All",
                            "Range",
                            # "HR ≈ step cadence",
                        ],
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
                    ui.card_header(
                        "Metric Visualization",
                    ),
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
                    ui.card_header(
                        ui.tooltip(
                            ui.span("Bias ", icon_svg("circle-question")),
                            ui.HTML(
                                """
        <p><strong>Bias</strong>: Systematic difference between test and reference.</p>
        <hr>
        <p><b>Paired t-test p-value:</b> Chance the mean difference is zero (parametric).</p>
        <p><b>Wilcoxon signed-rank p-value:</b> Chance the median difference is zero (non-parametric, uses ranks).</p>
        <p><b>Sign test p-value:</b> Chance positives and negatives are equally likely (ignores magnitude).</p>
        <p><b>Cohen's d:</b> Standardized mean difference relative to variability.</p>
    """
                            ),
                            placement="right",
                            id="bias_tooltip",
                        )
                    ),
                    ui.output_data_frame("biasTable"),
                ),
                ui.card(
                    ui.card_header(
                        ui.tooltip(
                            ui.span(
                                "Accuracy & Precision ",
                                icon_svg("circle-question"),
                            ),
                            ui.HTML(
                                """
        <p><b>Accuracy</b>: How close measurements are to the true or reference value.</p>
        <p><b>Precision</b>: How consistent repeated measurements are with each other.</p>
        <hr>
        <p><b>Mean Absolute Error (MAE)</b>: Average size of the errors, regardless of direction.</p>
        <p><b>Root Mean Squared Error (RMSE)</b>: Average size of the errors, giving extra weight to larger errors.</p>
        <p><b>Mean Squared Error (MSE)</b>: Average of squared errors; combines both bias and variability.</p>
        <p><b>Mean Absolute Percentage Error (MAPE)</b>: Average size of the errors expressed as a percentage of the reference values.</p>
        <p><b>Standard Deviation of Errors</b>: How spread out the errors are around their mean; reflects random variability.</p>
    """
                            ),
                            placement="right",
                            id="accuracy_tooltip",
                        )
                    ),
                    ui.output_data_frame("accuracyTable"),
                ),
                ui.card(
                    ui.card_header(
                        ui.tooltip(
                            ui.span(
                                "Agreement & Reliability ",
                                icon_svg("circle-question"),
                            ),
                            ui.HTML(
                                """
        <p><b>Agreement</b>: How closely two measurement methods produce the same values.</p>
        <p><b>Reliability</b>: How consistently a method produces the same result under similar conditions.</p>
        <hr>
        <p><b>Concordance Correlation Coefficient (CCC)</b>: A single statistic that reflects both correlation and closeness in scale between test and reference values.</p>
        <p><b>Pearson Correlation Coefficient</b>: Captures how strongly test and reference move together linearly, without requiring their values to match in magnitude.</p>
        <p><b>Pearson Correlation p-value</b>: Probability of seeing the observed correlation if the true correlation were zero.</p>
        <p><b>Limits of Agreement (LoA)</b>: Range where most differences between test and reference measurements fall, shown as mean bias ± 1.96 × SD of errors.</p>
    """
                            ),
                            placement="right",
                            id="agreement_tooltip",
                        )
                    ),
                    ui.output_data_frame("agreementTable"),
                ),
                col_widths=[4, 4, 4],
            ),
            ui.card(
                ui.card_header("LLM Generated Explanation"),
                ui.layout_columns(
                    ui.input_action_button("llm_summary_regen", "Ask BotBot!"),
                    ui.output_markdown_stream("streamOutput", auto_scroll=False),
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
            # ui.card(
            #     ui.card_header("File Information"),
            #     ui.output_data_frame("fileInfoTable"),
            # ),
            # ui.card(
            #     ui.card_header("Raw Data Sample"),
            #     ui.input_slider(
            #         "raw_data_sample_size",
            #         "Sample size",
            #         min=50,
            #         max=500,
            #         value=100,
            #         step=50,
            #     ),
            #     ui.output_data_frame("rawDataTable"),
            # ),
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
        return ui.input_text(
            id="shift_seconds",
            label="Shift test data (seconds):",
            value="",
            placeholder="Enter seconds (e.g., 5 or 5, 10, 0 or [5, 10, 0] for multiple pairs)",
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

    @render.ui
    def batchTagOptions():
        """
        Simple batch tag options - loads from S3 or uses defaults.
        """
        tags = load_batch_tags()

        return ui.input_checkbox_group(
            "batchTagOptions",
            "Select batch options:",
            choices=tags,
            selected=[],
        )

    @render.ui
    def batchContent():
        return None

    return {
        "benchmarkingContent": benchmarkingContent,
        "testFileSelector": testFileSelector,
        "refFileSelector": refFileSelector,
        "comparisonMetricSelector": comparisonMetricSelector,
        "outlierRemovalSelector": outlierRemovalSelector,
        "shiftSecondsSelector": shiftSecondsSelector,
        "batchTagOptions": batchTagOptions,
        "batchContent": batchContent,
    }
