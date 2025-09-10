"""
Visualization utilities using Plotly.
"""

from typing import Union

import pandas as pd
import plotly.graph_objects as go


def create_metric_plot(aligned_df: pd.DataFrame, metric: str) -> Union[go.Figure, None]:
    """Create line plot from aligned test and reference data for specified metric."""
    if aligned_df is None or aligned_df.empty:
        return None

    fig = go.Figure()

    # Track if averaging occurred for either test or reference
    averaging_happened = False

    # Add trend lines for both test and reference data
    for source_name, col_suffix in [("test", "_test"), ("reference", "_ref")]:
        metric_col = f"{metric}{col_suffix}"
        elapsed_col = f"elapsed_seconds{col_suffix}"

        if metric_col in aligned_df.columns and elapsed_col in aligned_df.columns:
            temp_df = aligned_df[[elapsed_col, metric_col]].dropna()
            if not temp_df.empty:
                # Check if any elapsed_seconds value is duplicated (i.e., averaging will occur)
                rounded = temp_df[elapsed_col].round()
                if rounded.duplicated().any():
                    averaging_happened = True
                mean_per_second = (
                    temp_df.groupby(rounded)[metric_col].mean().reset_index()
                )
                mean_per_second.rename(
                    columns={elapsed_col: "elapsed_seconds"}, inplace=True
                )

                if not mean_per_second.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=mean_per_second["elapsed_seconds"],
                            y=mean_per_second[metric_col],
                            mode="lines",
                            name=f"{source_name}",
                            line=dict(width=2),
                            opacity=0.75,
                        )
                    )

    if averaging_happened:
        yaxis_title = f"Mean {metric.replace('_', ' ').title()}"
    else:
        yaxis_title = metric.replace("_", " ").title()
    fig.update_layout(
        xaxis_title="Elapsed Seconds",
        yaxis_title=yaxis_title,
        hovermode="x unified",
    )
    return fig


def create_error_histogram(
    aligned_df: pd.DataFrame, metric: str
) -> Union[go.Figure, None]:
    """Create error distribution histogram from test and reference data."""
    if aligned_df is None or aligned_df.empty:
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
        xaxis_title="Error (test - reference)",
        yaxis_title="Frequency",
        barmode="overlay",
    )

    return fig


def create_bland_altman_plot(
    aligned_df: pd.DataFrame, metric: str
) -> Union[go.Figure, None]:
    """Create Bland-Altman plot from test and reference data."""
    if aligned_df is None or aligned_df.empty:
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
        xaxis_title=f"Mean of Test and Reference {metric}",
        yaxis_title=f"Difference (Test - Reference) {metric}",
        showlegend=False,
    )

    return fig


def create_rolling_error_plot(
    aligned_df: pd.DataFrame,
    metric: str,
    window_size: int = 50,
) -> Union[go.Figure, None]:
    """Create rolling error / time-varying bias plot from test and reference data."""
    if aligned_df is None or aligned_df.empty:
        return None

    # Sort by elapsed_seconds for proper time ordering in plot
    aligned_df = aligned_df.sort_values("elapsed_seconds_test")

    # Calculate differences (test - reference)
    differences = aligned_df[f"{metric}_test"] - aligned_df[f"{metric}_ref"]

    # Calculate rolling statistics
    rolling_mean = differences.rolling(window=window_size, center=True).mean()
    rolling_std = differences.rolling(window=window_size, center=True).std()

    fig = go.Figure()

    # Add rolling mean line
    fig.add_trace(
        go.Scatter(
            x=aligned_df["elapsed_seconds_test"],
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
            x=aligned_df["elapsed_seconds_test"],
            y=lower_band,
            mode="lines",
            name="Lower 95% CI",
            line=dict(width=2, color="#1E90FF", dash="dot"),  # DodgerBlue, thicker
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=aligned_df["elapsed_seconds_test"],
            y=upper_band,
            mode="lines",
            name="Upper 95% CI",
            line=dict(width=2, color="#1E90FF", dash="dot"),  # DodgerBlue, thicker
            fill="tonexty",
            fillcolor="rgba(30, 144, 255, 0.35)",  # More saturated blue, less transparent
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
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title=f"Rolling Error in {metric} (window={window_size})",
        showlegend=True,
        hovermode="x unified",
    )

    return fig
