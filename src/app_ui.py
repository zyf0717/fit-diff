from shiny import ui
from shinywidgets import output_widget

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "file_upload", "Upload FIT files", multiple=True, accept=[".fit"]
            ),
        ),
        ui.output_ui("comparisonMetricSelector"),
        ui.card(
            output_widget("outputPlot"),
        ),
        ui.card(
            ui.card_header("File Statistics"), ui.output_data_frame("basicStatsTable")
        ),
        ui.card(
            ui.card_header("Comparison Statistics"),
            ui.output_data_frame("diffStatsTable"),
        ),
        ui.card(
            ui.card_header("Data Tables"),
            ui.output_ui("fileSelector"),
            ui.output_data_frame("renderFitSessionDataFrame"),
            ui.output_data_frame("renderFitRecordsDataFrame"),
        ),
    )
)
