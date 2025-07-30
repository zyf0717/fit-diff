from shiny import ui

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "file_upload", "Choose FIT files", multiple=True, accept=[".fit"]
            ),
            ui.output_ui("file_summary"),
            title="Upload Files",
        ),
        ui.h1("FIT File Diff Analyzer"),
        ui.p("Upload two or more FIT files to compare their contents."),
        ui.h3("Comparison Results"),
        ui.output_data_frame("diff_table"),
    )
)
