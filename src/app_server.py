from typing import List

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo

from src.utils import process_fit


def server(input: Inputs, output: Outputs, session: Session):
    fit_data = reactive.Value({})

    @render.ui
    def file_selector():
        files = list(fit_data().keys())
        if not files:
            return ui.p("No files uploaded yet.")
        return ui.input_select(
            "selected_file",
            "Select file to display:",
            choices=files,
            selected=files[0] if files else None,
        )

    @reactive.Effect
    @reactive.event(input.file_upload)
    def process_uploaded_files():
        files: List[FileInfo] = input.file_upload()
        if not files:
            return
        current = fit_data().copy()
        for file_info in files:
            try:
                uploaded_file_path = file_info["datapath"]
                fit_df = process_fit(uploaded_file_path)
                current[file_info["name"]] = fit_df
            except Exception as e:
                current[file_info["name"]] = f"Error: {str(e)}"
        fit_data.set(current)

    @render.data_frame
    def render_fit_session_dataframe():
        selected = input.selected_file()
        if not selected:
            return pd.DataFrame()
        df = fit_data().get(selected)[0]
        if df is None:
            return pd.DataFrame()
        if isinstance(df, str):
            return pd.DataFrame({"error": [df]})
        if not df.empty:
            df = df.copy()
            return df
        return pd.DataFrame()

    @render.data_frame
    def render_fit_records_dataframe():
        selected = input.selected_file()
        if not selected:
            return pd.DataFrame()
        df = fit_data().get(selected)[1]
        if df is None:
            return pd.DataFrame()
        if isinstance(df, str):
            return pd.DataFrame({"error": [df]})
        if not df.empty:
            df = df.copy()
            return df
        return pd.DataFrame()
