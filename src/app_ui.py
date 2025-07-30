from shiny import ui

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "file_upload", "Upload FIT files", multiple=True, accept=[".fit"]
            ),
        ),
        ui.output_ui("file_selector"),
        ui.output_data_frame("render_fit_session_dataframe"),
        ui.output_data_frame("render_fit_records_dataframe"),
    )
)
