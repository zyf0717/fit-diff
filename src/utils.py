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
    all_fit_data: tuple, metrics: Union[str, list]
) -> Union[pd.DataFrame, None]:
    """Create combined DataFrame from fit data for specified metrics."""
    if (
        not all_fit_data
        or not isinstance(all_fit_data, tuple)
        or len(all_fit_data) != 2
    ):
        return pd.DataFrame()

    test_data_df, ref_data_df = all_fit_data
    if test_data_df.empty or ref_data_df.empty:
        return pd.DataFrame()

    metrics = [metrics] if isinstance(metrics, str) else metrics
    required_cols = ["timestamp"] + metrics

    test_data_df = test_data_df[required_cols].copy()
    test_data_df["source"] = "test"

    ref_data_df = ref_data_df[required_cols].copy()
    ref_data_df["source"] = "reference"

    combined_df = pd.concat([test_data_df, ref_data_df], ignore_index=True)
    if combined_df.empty:
        return pd.DataFrame()
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
    all_fit_data: tuple, metrics: Union[str, list], outlier_removal_methods: list = None
) -> Union[pd.DataFrame, None]:
    """Create combined DataFrame from fit data with outlier removal for specified metrics."""
    # First create the basic combined DataFrame
    combined_df = create_combined_df(all_fit_data, metrics)

    if combined_df is None or not outlier_removal_methods:
        return combined_df

    # Apply outlier removal for each metric
    metrics = [metrics] if isinstance(metrics, str) else metrics

    for metric in metrics:
        if metric in combined_df.columns:
            combined_df = remove_outliers(combined_df, metric, outlier_removal_methods)

    return combined_df if not combined_df.empty else pd.DataFrame()


def create_metric_plot(
    combined_df: pd.DataFrame, metrics: Union[str, list]
) -> Union[go.Figure, None]:
    """Create line plot from combined DataFrame for specified metrics."""
    if combined_df is None or combined_df.empty:
        return None

    metrics = [metrics] if isinstance(metrics, str) else metrics

    if len(metrics) == 1:
        return px.line(combined_df, x="timestamp", y=metrics[0], color="source")

    # Multiple metrics: melt for faceted plot
    melted_df = combined_df.melt(
        id_vars=["timestamp", "source"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    return px.line(
        melted_df, x="timestamp", y="value", color="source", facet_col="metric"
    )


def calculate_basic_stats(
    combined_df: pd.DataFrame, metrics: Union[str, list]
) -> Union[pd.DataFrame, None]:
    """Calculate basic statistics for test and reference data."""
    if combined_df is None or combined_df.empty:
        return None

    metrics = [metrics] if isinstance(metrics, str) else metrics

    stats_list = []
    for metric in metrics:
        if metric not in combined_df.columns:
            continue

        # Calculate stats for test data
        test_data = combined_df[combined_df["source"] == "test"][metric]
        if not test_data.empty:
            test_stats = {
                "device": "test",
                "metric": metric,
                "count": test_data.count(),
                "mean": test_data.mean(),
                "std": test_data.std(),
                "min": test_data.min(),
                "max": test_data.max(),
                "median": test_data.median(),
            }
            stats_list.append(test_stats)

        # Calculate stats for reference data
        ref_data = combined_df[combined_df["source"] == "reference"][metric]
        if not ref_data.empty:
            ref_stats = {
                "device": "reference",
                "metric": metric,
                "count": ref_data.count(),
                "mean": ref_data.mean(),
                "std": ref_data.std(),
                "min": ref_data.min(),
                "max": ref_data.max(),
                "median": ref_data.median(),
            }
            stats_list.append(ref_stats)

    if not stats_list:
        return None

    stats_df = pd.DataFrame(stats_list)
    return stats_df.round(2)


def calculate_diff_stats(
    combined_df: pd.DataFrame, metrics: Union[str, list]
) -> Union[pd.DataFrame, None]:
    """Calculate comparison statistics between test and reference data."""
    if combined_df is None or combined_df.empty:
        return None

    metrics = [metrics] if isinstance(metrics, str) else metrics

    all_stats = []
    for metric in metrics:
        if metric not in combined_df.columns:
            continue

        # Get test and reference data
        test_data = combined_df[combined_df["source"] == "test"][
            ["timestamp", metric]
        ].dropna()
        ref_data = combined_df[combined_df["source"] == "reference"][
            ["timestamp", metric]
        ].dropna()

        # Merge on timestamp to align the data
        aligned_df = pd.merge(
            test_data, ref_data, on="timestamp", suffixes=("_test", "_ref")
        )

        if aligned_df.empty:
            continue

        test_aligned = aligned_df[f"{metric}_test"]
        ref_aligned = aligned_df[f"{metric}_ref"]

        # Calculate error metrics
        errors = test_aligned - ref_aligned
        mae = errors.abs().mean()
        mse = (errors**2).mean()
        rmse = mse**0.5
        correlation = test_aligned.corr(ref_aligned)

        # Calculate bias and limits of agreement for Bland-Altman
        bias = errors.mean()
        std_errors = errors.std()
        loa_upper = bias + 1.96 * std_errors
        loa_lower = bias - 1.96 * std_errors

        all_stats.append(
            {
                "metric": metric,
                "comparison": "test vs reference",
                "mae": round(mae, 3),
                "mse": round(mse, 3),
                "rmse": round(rmse, 3),
                "bias": round(bias, 3),
                "loa_upper": round(loa_upper, 3),
                "loa_lower": round(loa_lower, 3),
                "correlation": round(correlation, 3),
                "points": len(aligned_df),
            }
        )

    return pd.DataFrame(all_stats) if all_stats else None


def create_error_histogram(
    combined_df: pd.DataFrame, metric: str
) -> Union[go.Figure, None]:
    """Create error distribution histogram from combined DataFrame."""
    if combined_df is None or combined_df.empty:
        return None

    if metric not in combined_df.columns:
        return None

    # Get test and reference data
    test_data = combined_df[combined_df["source"] == "test"][
        ["timestamp", metric]
    ].dropna()
    ref_data = combined_df[combined_df["source"] == "reference"][
        ["timestamp", metric]
    ].dropna()

    # Merge on timestamp to align the data
    aligned_df = pd.merge(
        test_data, ref_data, on="timestamp", suffixes=("_test", "_ref")
    )

    if aligned_df.empty:
        return None

    # Calculate errors (test - reference)
    errors = aligned_df[f"{metric}_test"] - aligned_df[f"{metric}_ref"]

    fig = go.Figure()

    # Add histogram of errors
    fig.add_trace(
        go.Histogram(
            x=errors,
            name="Test - Reference",
            opacity=0.7,
            nbinsx=30,
        )
    )

    fig.update_layout(
        title=f"Error Distribution for {metric} (Test vs Reference)",
        xaxis_title="Error (test - reference)",
        yaxis_title="Frequency",
        barmode="overlay",
    )

    return fig


def create_bland_altman_plot(
    combined_df: pd.DataFrame, metric: str
) -> Union[go.Figure, None]:
    """Create Bland-Altman plot from combined DataFrame."""
    if combined_df is None or combined_df.empty:
        return None

    if metric not in combined_df.columns:
        return None

    # Get test and reference data
    test_data = combined_df[combined_df["source"] == "test"][
        ["timestamp", metric]
    ].dropna()
    ref_data = combined_df[combined_df["source"] == "reference"][
        ["timestamp", metric]
    ].dropna()

    # Merge on timestamp to align the data
    aligned_df = pd.merge(
        test_data, ref_data, on="timestamp", suffixes=("_test", "_ref")
    )

    if aligned_df.empty:
        return None

    # Calculate mean and difference for Bland-Altman plot
    x_vals = (aligned_df[f"{metric}_test"] + aligned_df[f"{metric}_ref"]) / 2  # Mean
    y_vals = (
        aligned_df[f"{metric}_test"] - aligned_df[f"{metric}_ref"]
    )  # Difference (test - reference)

    # Calculate limits of agreement
    diff_mean = y_vals.mean()
    diff_std = y_vals.std()
    upper_loa = diff_mean + 1.96 * diff_std
    lower_loa = diff_mean - 1.96 * diff_std

    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name="Test vs Reference",
            opacity=0.7,
        )
    )

    # Add mean difference line
    fig.add_hline(
        y=diff_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean bias: {diff_mean:.3f}",
    )

    # Add limits of agreement
    fig.add_hline(
        y=upper_loa,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"+1.96 SD: {upper_loa:.3f}",
    )
    fig.add_hline(
        y=lower_loa,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"-1.96 SD: {lower_loa:.3f}",
    )

    fig.update_layout(
        title=f"Bland-Altman Plot for {metric} (Test vs Reference)",
        xaxis_title=f"Mean of Test and Reference {metric}",
        yaxis_title=f"Difference (Test - Reference) {metric}",
        showlegend=True,
    )

    return fig


def create_rolling_error_plot(
    combined_df: pd.DataFrame,
    metric: str,
    window_size: int = 50,
) -> Union[go.Figure, None]:
    """Create rolling error / time-varying bias plot from combined DataFrame."""
    if combined_df is None or combined_df.empty:
        return None

    if metric not in combined_df.columns:
        return None

    # Get test and reference data
    test_data = combined_df[combined_df["source"] == "test"][
        ["timestamp", metric]
    ].dropna()
    ref_data = combined_df[combined_df["source"] == "reference"][
        ["timestamp", metric]
    ].dropna()

    # Merge on timestamp to align the data
    aligned_df = pd.merge(
        test_data, ref_data, on="timestamp", suffixes=("_test", "_ref")
    )

    if aligned_df.empty:
        return None

    # Sort by timestamp
    aligned_df = aligned_df.sort_values("timestamp")

    # Calculate differences (test - reference)
    differences = aligned_df[f"{metric}_test"] - aligned_df[f"{metric}_ref"]

    # Calculate rolling statistics
    rolling_mean = differences.rolling(window=window_size, center=True).mean()
    rolling_std = differences.rolling(window=window_size, center=True).std()

    fig = go.Figure()

    # Add rolling mean line
    fig.add_trace(
        go.Scatter(
            x=aligned_df["timestamp"],
            y=rolling_mean,
            mode="lines",
            name=f"Rolling mean error (window={window_size})",
            line=dict(width=2, color="blue"),
        )
    )

    # Add rolling confidence bands
    upper_band = rolling_mean + 1.96 * rolling_std
    lower_band = rolling_mean - 1.96 * rolling_std

    fig.add_trace(
        go.Scatter(
            x=aligned_df["timestamp"],
            y=upper_band,
            mode="lines",
            name="Upper 95% CI",
            line=dict(width=1, color="lightblue", dash="dot"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=aligned_df["timestamp"],
            y=lower_band,
            mode="lines",
            name="Lower 95% CI",
            line=dict(width=1, color="lightblue", dash="dot"),
            fill="tonexty",
            fillcolor="rgba(173, 216, 230, 0.2)",
            showlegend=True,
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

    fig.update_layout(
        title=f"Rolling Error / Time-Varying Bias for {metric} (Test vs Reference)",
        xaxis_title="Time",
        yaxis_title=f"Rolling Error in {metric} (window={window_size})",
        showlegend=True,
        hovermode="x unified",
    )

    return fig
