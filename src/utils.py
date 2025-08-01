"""
Supporting utils and functions
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from garmin_fit_sdk import Decoder, Stream


def process_fit(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a FIT file and return:
      - meta_df: one‐row DataFrame of all non‐record messages (e.g. file_id, developer_data_id, etc.)
      - record_df: one‐row per 'record' message (timestamped samples)
    """
    stream = Stream.from_file(file_path)
    decoder = Decoder(stream)
    messages, _ = decoder.read(  # All defaults, listed for clarity
        apply_scale_and_offset=True,
        convert_datetimes_to_dates=True,
        convert_types_to_strings=True,
        enable_crc_check=True,
        expand_sub_fields=True,
        expand_components=True,
        merge_heart_rates=True,
        mesg_listener=None,
    )

    record_df = pd.json_normalize(messages.get("record_mesgs", []), sep="_")
    if record_df.empty:
        raise ValueError("No record messages found in FIT file")
    if "position_lat" in record_df.columns and "position_long" in record_df.columns:
        record_df["position_lat"] = record_df["position_lat"] * (180 / 2**31)
        record_df["position_long"] = record_df["position_long"] * (180 / 2**31)

    session_df = pd.json_normalize(messages.get("session_mesgs", []), sep="_")
    if session_df.empty:
        raise ValueError("No session messages found in FIT file")

    return session_df, record_df


def create_combined_df(
    fit_data_dict: dict, metrics: Union[str, list]
) -> Union[pd.DataFrame, None]:
    """Create combined DataFrame from fit data for specified metrics."""
    if not fit_data_dict:
        return None

    metrics = [metrics] if isinstance(metrics, str) else metrics
    required_cols = ["timestamp"] + metrics

    relevant_dfs = []
    for k, v in fit_data_dict.items():
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        df = v[1]
        if isinstance(df, str) or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        available_cols = [col for col in required_cols if col in df.columns]
        if len(available_cols) < 2:
            continue
        sub_df = df[available_cols].copy()
        sub_df["file"] = k
        relevant_dfs.append(sub_df)

    if not relevant_dfs:
        return None

    combined_df = pd.concat(relevant_dfs, ignore_index=True)
    if combined_df.empty:
        return None

    # # Find common timestamp range across all files
    # ts_min = combined_df.groupby("file")["timestamp"].min().max()
    # ts_max = combined_df.groupby("file")["timestamp"].max().min()
    # if ts_min > ts_max:
    #     return None
    # combined_df = combined_df[
    #     (combined_df["timestamp"] >= ts_min) & (combined_df["timestamp"] <= ts_max)
    # ]
    return combined_df if not combined_df.empty else None


def remove_outliers(
    df: pd.DataFrame, metric: str, removal_methods: list
) -> pd.DataFrame:
    """Remove outliers from DataFrame based on specified methods."""
    if df is None or df.empty or metric not in df.columns:
        return df

    if not removal_methods:
        return df

    df_filtered = df.copy()

    for method in removal_methods:
        if method == "remove_zeros":
            # Remove rows where the metric is zero
            df_filtered = df_filtered[df_filtered[metric] != 0]

        elif method == "remove_iqr":
            # Remove outliers using IQR method (1.5 × IQR)
            Q1 = df_filtered[metric].quantile(0.25)
            Q3 = df_filtered[metric].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_filtered = df_filtered[
                (df_filtered[metric] >= lower_bound)
                & (df_filtered[metric] <= upper_bound)
            ]

        elif method == "remove_zscore":
            # Remove outliers using Z-score method (|z| > 3)
            z_scores = np.abs(
                (df_filtered[metric] - df_filtered[metric].mean())
                / df_filtered[metric].std()
            )
            df_filtered = df_filtered[z_scores <= 3]

        elif method == "remove_percentile":
            # Remove outliers using percentile method (< 1% or > 99%)
            lower_percentile = df_filtered[metric].quantile(0.01)
            upper_percentile = df_filtered[metric].quantile(0.99)
            df_filtered = df_filtered[
                (df_filtered[metric] >= lower_percentile)
                & (df_filtered[metric] <= upper_percentile)
            ]

    return df_filtered


def create_combined_df_with_outlier_removal(
    fit_data_dict: dict, metrics: Union[str, list], outlier_removal_methods: list = None
) -> Union[pd.DataFrame, None]:
    """Create combined DataFrame from fit data with outlier removal for specified metrics."""
    # First create the basic combined DataFrame
    combined_df = create_combined_df(fit_data_dict, metrics)

    if combined_df is None or not outlier_removal_methods:
        return combined_df

    # Apply outlier removal for each metric
    metrics = [metrics] if isinstance(metrics, str) else metrics

    for metric in metrics:
        if metric in combined_df.columns:
            combined_df = remove_outliers(combined_df, metric, outlier_removal_methods)

    return combined_df if not combined_df.empty else None


def create_metric_plot(
    combined_df: pd.DataFrame, metrics: Union[str, list]
) -> Union[go.Figure, None]:
    """Create line plot from combined DataFrame for specified metrics."""
    if combined_df is None or combined_df.empty:
        return None

    metrics = [metrics] if isinstance(metrics, str) else metrics

    if len(metrics) == 1:
        return px.line(combined_df, x="timestamp", y=metrics[0], color="file")

    # Multiple metrics: melt for faceted plot
    melted_df = combined_df.melt(
        id_vars=["timestamp", "file"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    return px.line(
        melted_df, x="timestamp", y="value", color="file", facet_col="metric"
    )


def calculate_basic_stats(
    combined_df: pd.DataFrame, metrics: Union[str, list]
) -> Union[pd.DataFrame, None]:
    """Calculate basic statistics for each file and metric."""
    if combined_df is None or combined_df.empty:
        return None

    metrics = [metrics] if isinstance(metrics, str) else metrics
    available_metrics = [m for m in metrics if m in combined_df.columns]

    if not available_metrics:
        return None

    stats_list = []
    for metric in available_metrics:
        metric_stats = (
            combined_df.groupby("file")[metric]
            .agg(["count", "mean", "std", "min", "max", "median"])
            .round(2)
            .reset_index()
        )
        metric_stats["metric"] = metric
        stats_list.append(metric_stats)

    return pd.concat(stats_list, ignore_index=True)


def calculate_diff_stats(
    combined_df: pd.DataFrame, metrics: Union[str, list]
) -> Union[pd.DataFrame, None]:
    """Calculate pairwise comparison statistics between files for each metric."""
    if combined_df is None or combined_df.empty:
        return None

    files = combined_df["file"].unique()
    if len(files) < 2:
        return None

    metrics = [metrics] if isinstance(metrics, str) else metrics
    available_metrics = [m for m in metrics if m in combined_df.columns]

    if not available_metrics:
        return None

    all_stats = []
    for metric in available_metrics:
        pivot_df = combined_df.pivot_table(
            index="timestamp", columns="file", values=metric, aggfunc="first"
        ).dropna()

        if pivot_df.empty:
            continue

        for i, file1 in enumerate(files):
            for file2 in files[i + 1 :]:
                if file1 in pivot_df.columns and file2 in pivot_df.columns:
                    y1, y2 = pivot_df[file1], pivot_df[file2]
                    mae = (y1 - y2).abs().mean()
                    mse = ((y1 - y2) ** 2).mean()

                    all_stats.append(
                        {
                            "metric": metric,
                            "file1": file1,
                            "file2": file2,
                            "mae": round(mae, 2),
                            "mse": round(mse, 2),
                            "rmse": round(mse**0.5, 2),
                            "correlation": round(y1.corr(y2), 3),
                            "points": len(y1),
                        }
                    )

    return pd.DataFrame(all_stats) if all_stats else None


def create_error_histogram(
    combined_df: pd.DataFrame, metric: str, benchmark_file: str = None
) -> Union[go.Figure, None]:
    """Create error distribution histogram from combined DataFrame."""
    if combined_df is None or combined_df.empty:
        return None

    files = combined_df["file"].unique()
    if len(files) < 2:
        return None

    # Pivot to get each file as a column
    pivot_df = combined_df.pivot_table(
        index="timestamp", columns="file", values=metric, aggfunc="first"
    ).dropna()

    if pivot_df.empty:
        return None

    # Calculate pairwise errors
    fig = go.Figure()

    # If benchmark is specified, compare all others to it
    if benchmark_file and benchmark_file in files:
        for file in files:
            if file != benchmark_file and file in pivot_df.columns:
                errors = pivot_df[file] - pivot_df[benchmark_file]
                fig.add_trace(
                    go.Histogram(
                        x=errors,
                        name=f"{file} - {benchmark_file} (ref)",
                        opacity=0.7,
                        nbinsx=30,
                    )
                )
    else:
        # Default: all pairwise comparisons (first file is implicit reference)
        for i, file1 in enumerate(files):
            for file2 in files[i + 1 :]:
                if file1 in pivot_df.columns and file2 in pivot_df.columns:
                    errors = pivot_df[file2] - pivot_df[file1]
                    ref_indicator = " (ref)" if i == 0 else ""
                    fig.add_trace(
                        go.Histogram(
                            x=errors,
                            name=f"{file2} - {file1}{ref_indicator}",
                            opacity=0.7,
                            nbinsx=30,
                        )
                    )

    title_suffix = (
        f" (vs {benchmark_file})" if benchmark_file else " (first file as reference)"
    )
    fig.update_layout(
        title=f"Error Distribution for {metric}{title_suffix}",
        xaxis_title="Error (measured - reference)",
        yaxis_title="Frequency",
        barmode="overlay",
    )

    return fig


def create_bland_altman_plot(
    combined_df: pd.DataFrame, metric: str, benchmark_file: str = None
) -> Union[go.Figure, None]:
    """Create Bland-Altman plot from combined DataFrame."""
    if combined_df is None or combined_df.empty:
        return None

    files = combined_df["file"].unique()
    if len(files) < 2:
        return None

    # Pivot to get each file as a column
    pivot_df = combined_df.pivot_table(
        index="timestamp", columns="file", values=metric, aggfunc="first"
    ).dropna()

    if pivot_df.empty:
        return None

    fig = go.Figure()

    # If benchmark is specified, compare all others to it
    if benchmark_file and benchmark_file in files:
        for file in files:
            if file != benchmark_file and file in pivot_df.columns:
                x_vals = pivot_df[[benchmark_file, file]].mean(axis=1)  # Mean
                y_vals = pivot_df[file] - pivot_df[benchmark_file]  # Difference

                # Calculate limits of agreement
                diff_mean = y_vals.mean()
                diff_std = y_vals.std()
                upper_loa = diff_mean + 1.96 * diff_std
                lower_loa = diff_mean - 1.96 * diff_std

                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        name=f"{file} vs {benchmark_file}",
                        opacity=0.7,
                    )
                )

                # Add mean difference line
                fig.add_hline(
                    y=diff_mean,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {diff_mean:.2f}",
                )

                # Add limits of agreement
                fig.add_hline(
                    y=upper_loa,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text=f"+1.96 SD: {upper_loa:.2f}",
                )
                fig.add_hline(
                    y=lower_loa,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text=f"-1.96 SD: {lower_loa:.2f}",
                )
    else:
        # Default: compare all others to first file
        reference_file = files[0]
        for file in files[1:]:
            if file in pivot_df.columns and reference_file in pivot_df.columns:
                x_vals = pivot_df[[reference_file, file]].mean(axis=1)  # Mean
                y_vals = pivot_df[file] - pivot_df[reference_file]  # Difference

                # Calculate limits of agreement
                diff_mean = y_vals.mean()
                diff_std = y_vals.std()
                upper_loa = diff_mean + 1.96 * diff_std
                lower_loa = diff_mean - 1.96 * diff_std

                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        name=f"{file} vs {reference_file} (ref)",
                        opacity=0.7,
                    )
                )

                # Add mean difference line (only for first comparison to avoid duplicates)
                if file == files[1]:
                    fig.add_hline(
                        y=diff_mean,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {diff_mean:.2f}",
                    )

                    # Add limits of agreement
                    fig.add_hline(
                        y=upper_loa,
                        line_dash="dot",
                        line_color="orange",
                        annotation_text=f"+1.96 SD: {upper_loa:.2f}",
                    )
                    fig.add_hline(
                        y=lower_loa,
                        line_dash="dot",
                        line_color="orange",
                        annotation_text=f"-1.96 SD: {lower_loa:.2f}",
                    )

    title_suffix = (
        f" (vs {benchmark_file})" if benchmark_file else " (vs first file as reference)"
    )
    fig.update_layout(
        title=f"Bland-Altman Plot for {metric}{title_suffix}",
        xaxis_title=f"Mean of {metric}",
        yaxis_title=f"Difference in {metric} (measured - reference)",
        showlegend=True,
    )

    return fig


def create_rolling_error_plot(
    combined_df: pd.DataFrame,
    metric: str,
    benchmark_file: str = None,
    window_size: int = 50,
) -> Union[go.Figure, None]:
    """Create rolling error / time-varying bias plot from combined DataFrame."""
    if combined_df is None or combined_df.empty:
        return None

    files = combined_df["file"].unique()
    if len(files) < 2:
        return None

    # Pivot to get each file as a column
    pivot_df = combined_df.pivot_table(
        index="timestamp", columns="file", values=metric, aggfunc="first"
    ).dropna()

    if pivot_df.empty:
        return None

    # Sort by timestamp to ensure proper rolling calculation
    pivot_df = pivot_df.sort_index()

    fig = go.Figure()

    # If benchmark is specified, compare all others to it
    if benchmark_file and benchmark_file in files:
        for file in files:
            if file != benchmark_file and file in pivot_df.columns:
                # Calculate differences
                differences = pivot_df[file] - pivot_df[benchmark_file]

                # Calculate rolling mean and std
                rolling_mean = differences.rolling(
                    window=window_size, center=True
                ).mean()
                rolling_std = differences.rolling(window=window_size, center=True).std()

                # Add rolling mean line
                fig.add_trace(
                    go.Scatter(
                        x=pivot_df.index,
                        y=rolling_mean,
                        mode="lines",
                        name=f"{file} - {benchmark_file} (rolling mean)",
                        line=dict(width=2),
                    )
                )

                # Add confidence bands (±1 std)
                upper_band = rolling_mean + rolling_std
                lower_band = rolling_mean - rolling_std

                fig.add_trace(
                    go.Scatter(
                        x=pivot_df.index,
                        y=upper_band,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=pivot_df.index,
                        y=lower_band,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=f"rgba(0,100,80,0.2)",
                        name=f"{file} ±1σ band",
                        hoverinfo="skip",
                    )
                )
    else:
        # Default: compare all others to first file
        reference_file = files[0]
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, file in enumerate(files[1:]):
            if file in pivot_df.columns and reference_file in pivot_df.columns:
                # Calculate differences
                differences = pivot_df[file] - pivot_df[reference_file]

                # Calculate rolling mean and std
                rolling_mean = differences.rolling(
                    window=window_size, center=True
                ).mean()
                rolling_std = differences.rolling(window=window_size, center=True).std()

                color = colors[i % len(colors)]

                # Add rolling mean line
                fig.add_trace(
                    go.Scatter(
                        x=pivot_df.index,
                        y=rolling_mean,
                        mode="lines",
                        name=f"{file} - {reference_file} (ref)",
                        line=dict(width=2, color=color),
                    )
                )

                # Add confidence bands (±1 std)
                upper_band = rolling_mean + rolling_std
                lower_band = rolling_mean - rolling_std

                fig.add_trace(
                    go.Scatter(
                        x=pivot_df.index,
                        y=upper_band,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=pivot_df.index,
                        y=lower_band,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=f"rgba({i*50},{100-i*20},{80+i*30},0.2)",
                        name=f"{file} ±1σ band",
                        hoverinfo="skip",
                    )
                )

    # Add horizontal line at zero for reference
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        line_width=1,
        annotation_text="Zero bias",
    )

    title_suffix = (
        f" (vs {benchmark_file})" if benchmark_file else " (vs first file as reference)"
    )
    fig.update_layout(
        title=f"Rolling Error / Time-Varying Bias for {metric}{title_suffix}",
        xaxis_title="Time",
        yaxis_title=f"Rolling Error in {metric} (window={window_size})",
        showlegend=True,
        hovermode="x unified",
    )

    return fig
