from shiny import ui
from shinywidgets import output_widget

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "file_upload", "Upload FIT files", multiple=True, accept=[".fit"]
            ),
        ),
        output_widget("outputPlot"),
        ui.output_ui("fileSelector"),
        ui.output_data_frame("renderFitSessionDataFrame"),
        ui.output_data_frame("renderFitRecordsDataFrame"),
    )
)
