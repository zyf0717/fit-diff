"""
Shiny Python app for comparing FIT files.
"""

from typing import List

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo

from .diff_analyzer import DiffAnalyzer
from .fit_processor import FitProcessor


def server(input: Inputs, output: Outputs, session: Session):
    fit_data = reactive.Value({})

    @reactive.Effect
    @reactive.event(input.file_upload)
    def process_uploaded_files():
        """Process uploaded FIT files when they are uploaded."""
        files: List[FileInfo] = input.file_upload()
        if not files:
            return

        processor = FitProcessor()
        data = {}

        for file_info in files:
            try:
                # Get the uploaded file path
                uploaded_file_path = file_info["datapath"]

                # Process FIT file directly from uploaded path
                fit_df = processor.process_fit_file(uploaded_file_path)
                data[file_info["name"]] = fit_df

            except Exception as e:
                # Handle processing errors
                data[file_info["name"]] = f"Error: {str(e)}"

        fit_data.set(data)

    @output
    @render.ui
    def file_summary():
        """Display summary of uploaded files."""
        data = fit_data.get()
        if not data:
            return ui.p("No files uploaded yet.")

        summaries = []
        for filename, df in data.items():
            if isinstance(df, str):  # Error case
                summaries.append(ui.div(ui.h4(filename), ui.p(df, style="color: red;")))
            else:
                summaries.append(ui.div(ui.h4(filename), ui.p(f"Records: {len(df)}")))

        return ui.div(*summaries)

    @output
    @render.data_frame
    def diff_table():
        """Display diff analysis table."""
        data = fit_data.get()
        if len(data) < 2:
            return pd.DataFrame(
                {"Message": ["Upload at least 2 FIT files to see differences"]}
            )

        # Filter out error cases
        valid_data = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
        if len(valid_data) < 2:
            return pd.DataFrame({"Message": ["At least 2 valid FIT files required"]})

        analyzer = DiffAnalyzer()
        diff_df = analyzer.compare_files(
            list(valid_data.values()), list(valid_data.keys())
        )
        return diff_df


# --- UI with sidebar for uploads ---
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

app = App(
    app_ui, server
)  # Placeholder for app.py, will move code from root/app.py here.
app = App(
    app_ui, server
)  # Placeholder for app.py, will move code from root/app.py here.
app = App(
    app_ui, server
)  # Placeholder for app.py, will move code from root/app.py here.
