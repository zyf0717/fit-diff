from shiny import ui
from shinywidgets import output_widget

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "test_file_upload",
                "Upload test FIT files",
                multiple=True,
                accept=[".fit"],
            ),
            ui.input_file(
                "ref_file_upload",
                "Upload reference FIT files",
                multiple=True,
                accept=[".fit"],
            ),
        ),
        ui.output_ui("main_content"),
    ),
)
