from typing import List

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo

from src.diff_analyzer import DiffAnalyzer
from src.fit_processor import FitProcessor


def server(input: Inputs, output: Outputs, session: Session):
    fit_data = reactive.Value({})

    @reactive.Effect
    @reactive.event(input.file_upload)
    def process_uploaded_files():
        files: List[FileInfo] = input.file_upload()
        if not files:
            return
        processor = FitProcessor()
        data = {}
        for file_info in files:
            try:
                uploaded_file_path = file_info["datapath"]
                fit_df = processor.process_fit_file(uploaded_file_path)
                data[file_info["name"]] = fit_df
            except Exception as e:
                data[file_info["name"]] = f"Error: {str(e)}"
        fit_data.set(data)

    @output
    @render.ui
    def file_summary():
        data = fit_data.get()
        if not data:
            return ui.p("No files uploaded yet.")
        summaries = []
        for filename, df in data.items():
            if isinstance(df, str):
                summaries.append(ui.div(ui.h4(filename), ui.p(df, style="color: red;")))
            else:
                summaries.append(ui.div(ui.h4(filename), ui.p(f"Records: {len(df)}")))
        return ui.div(*summaries)

    @output
    @render.data_frame
    def diff_table():
        data = fit_data.get()
        if len(data) < 2:
            return pd.DataFrame(
                {"Message": ["Upload at least 2 FIT files to see differences"]}
            )
        valid_data = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
        if len(valid_data) < 2:
            return pd.DataFrame({"Message": ["At least 2 valid FIT files required"]})
        analyzer = DiffAnalyzer()
        diff_df = analyzer.compare_files(
            list(valid_data.values()), list(valid_data.keys())
        )
        return diff_df
