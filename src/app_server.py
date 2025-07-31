import logging
from typing import List

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shinywidgets import render_widget

from src.utils import (
    calculate_basic_stats,
    calculate_diff_stats,
    create_combined_df,
    create_metric_plot,
    process_fit,
)

logger = logging.getLogger(__name__)


def server(input: Inputs, output: Outputs, session: Session):
    fit_data = reactive.Value({})

    @render.ui
    def comparisonMetricSelector():
        choices = _get_common_metrics()
        if not choices:
            return None
        return ui.input_select(
            "comparison_metric",
            "Select comparison metric:",
            choices=choices,
            selected="heart_rate",
        )

    @render.ui
    def fileSelector():
        files = list(fit_data().keys())
        if not files:
            return None
        return ui.input_select(
            "selected_file",
            "Select file:",
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

    @reactive.Calc
    def _get_common_metrics():
        data_dict = fit_data()
        if not data_dict:
            return []

        valid_dfs = []
        for v in data_dict.values():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                df = v[1]  # records DataFrame
                if isinstance(df, pd.DataFrame) and not df.empty:
                    valid_dfs.append(df)

        if not valid_dfs:
            return []

        # Get intersection of all column sets
        common_cols = set(valid_dfs[0].columns)
        for df in valid_dfs[1:]:
            common_cols &= set(df.columns)

        # Exclude timestamp columns
        common_cols = {col for col in common_cols if "timestamp" not in col.lower()}

        return sorted(list(common_cols))

    @reactive.Calc
    def _get_combined_df():
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )
        return create_combined_df(fit_data(), metric)

    @render_widget
    def outputPlot():
        combined_df = _get_combined_df()
        if combined_df is None or combined_df.empty:
            return None
        return create_metric_plot(combined_df, input.comparison_metric())

    @render.data_frame
    def basicStatsTable():
        combined_df = _get_combined_df()
        if combined_df is None:
            return pd.DataFrame()
        return calculate_basic_stats(combined_df, input.comparison_metric())

    @render.data_frame
    def diffStatsTable():
        combined_df = _get_combined_df()
        if combined_df is None:
            return pd.DataFrame()
        return calculate_diff_stats(combined_df, input.comparison_metric())

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
