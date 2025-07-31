import logging
from typing import List

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shinywidgets import render_widget

from src.utils import create_heart_rate_plot, process_fit

logger = logging.getLogger(__name__)


def server(input: Inputs, output: Outputs, session: Session):
    fit_data = reactive.Value({})

    @render.ui
    def fileSelector():
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
    def _process_uploaded_files():
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
                logger.error("Error processing file %s: %s", file_info["name"], str(e))
                current[file_info["name"]] = f"Error: {str(e)}"
        fit_data.set(current)

    @render_widget
    def outputPlot():
        return create_heart_rate_plot(fit_data())

    def _render_fit_dataframe(index: int):
        selected = input.selected_file()
        if not selected:
            return pd.DataFrame()
        df_tuple = fit_data().get(selected)
        if not df_tuple or len(df_tuple) <= index:
            return pd.DataFrame()
        df = df_tuple[index]
        if isinstance(df, str):
            return pd.DataFrame({"error": [df]})
        return df.copy() if df is not None and not df.empty else pd.DataFrame()

    @render.data_frame
    def renderFitSessionDataFrame():
        return _render_fit_dataframe(0)

    @render.data_frame
    def renderFitRecordsDataFrame():
        return _render_fit_dataframe(1)
