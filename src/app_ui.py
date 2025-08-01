from shiny import ui
from shinywidgets import output_widget

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "file_upload", "Upload FIT files", multiple=True, accept=[".fit"]
            ),
        ),
        ui.layout_columns(
            ui.output_ui("benchmarkSelector"),
            ui.output_ui("comparisonMetricSelector"),
            ui.output_ui("outlierRemovalSelector"),
            col_widths=[3, 3, 3],
        ),
        ui.card(
            ui.card_header("File Statistics"), ui.output_data_frame("basicStatsTable")
        ),
        ui.card(
            ui.card_header("Comparison Statistics"),
            ui.output_data_frame("diffStatsTable"),
        ),
        ui.card(
            ui.card_header("Metric Comparison Plot"),
            output_widget("metricPlot"),
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
        ui.card(
            ui.card_header("Data Tables"),
            ui.output_ui("fileSelector"),
            ui.output_data_frame("renderFitSessionDataFrame"),
            ui.output_data_frame("renderFitRecordsDataFrame"),
        ),
    )
)
