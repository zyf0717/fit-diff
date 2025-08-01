import logging
from typing import List

import pandas as pd
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shinywidgets import render_widget

from src.utils import (
    calculate_basic_stats,
    calculate_diff_stats,
    create_bland_altman_plot,
    create_combined_df,
    create_combined_df_with_outlier_removal,
    create_error_histogram,
    create_metric_plot,
    create_rolling_error_plot,
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

    @render.ui
    def benchmarkSelector():
        files = list(fit_data().keys())
        if len(files) < 2:
            return None
        return ui.input_select(
            "benchmark_file",
            "Select reference/benchmark file:",
            choices=["Auto (first file)"] + files,
            selected="Auto (first file)",
        )

    @render.ui
    def outlierRemovalSelector():
        return ui.input_selectize(
            "outlier_removal",
            "Outlier removal:",
            choices={
                "remove_zeros": "Remove zero values",
                "remove_iqr": "Remove IQR outliers (1.5 × IQR)",
                "remove_zscore": "Remove Z-score outliers (|z| > 3)",
                "remove_percentile": "Remove percentile outliers (< 1% or > 99%)",
            },
            selected=[],
            multiple=True,
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

        # Get outlier removal methods
        outlier_methods = []
        if hasattr(input, "outlier_removal"):
            outlier_methods = input.outlier_removal() or []

        return create_combined_df_with_outlier_removal(
            fit_data(), metric, outlier_methods
        )

    @render_widget
    def metricPlot():
        combined_df = _get_combined_df()
        if combined_df is None:
            return None
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )
        return create_metric_plot(combined_df, metric)

    @render_widget
    def errorHistogramPlot():
        combined_df = _get_combined_df()
        if combined_df is None:
            return None
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        # Get benchmark file selection
        benchmark = None
        if hasattr(input, "benchmark_file"):
            selected = input.benchmark_file()
            if selected and selected != "Auto (first file)":
                benchmark = selected

        return create_error_histogram(combined_df, metric, benchmark)

    @render_widget
    def blandAltmanPlot():
        combined_df = _get_combined_df()
        if combined_df is None:
            return None
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        # Get benchmark file selection
        benchmark = None
        if hasattr(input, "benchmark_file"):
            selected = input.benchmark_file()
            if selected and selected != "Auto (first file)":
                benchmark = selected

        return create_bland_altman_plot(combined_df, metric, benchmark)

    @render_widget
    def rollingErrorPlot():
        combined_df = _get_combined_df()
        if combined_df is None:
            return None
        metric = (
            input.comparison_metric()
            if hasattr(input, "comparison_metric")
            else "heart_rate"
        )

        # Get benchmark file selection
        benchmark = None
        if hasattr(input, "benchmark_file"):
            selected = input.benchmark_file()
            if selected and selected != "Auto (first file)":
                benchmark = selected

        return create_rolling_error_plot(combined_df, metric, benchmark)

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
