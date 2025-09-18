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

    # Get unique start_datetime to plot each pair separately
    if "start_datetime" in aligned_df.columns:
        start_datetimes = aligned_df["start_datetime"].unique()

        for start_datetime in start_datetimes:
            # Filter data for this specific start_datetime
            file_data = aligned_df[aligned_df["start_datetime"] == start_datetime]

            if not file_data.empty:
                # Plot test data for this start_datetime
                test_col = f"{metric}_test"
                ref_col = f"{metric}_ref"
                elapsed_test_col = f"elapsed_seconds_test"
                elapsed_ref_col = f"elapsed_seconds_ref"

                # Add test line
                if (
                    test_col in file_data.columns
                    and elapsed_test_col in file_data.columns
                ):
                    test_data = file_data[[elapsed_test_col, test_col]].dropna()
                    if not test_data.empty:
                        test_data = test_data.sort_values(elapsed_test_col)
                        fig.add_trace(
                            go.Scatter(
                                x=test_data[elapsed_test_col],
                                y=test_data[test_col],
                                mode="lines",
                                name=f"{start_datetime} (test)",
                                opacity=0.8,
                                line=dict(width=2),
                            )
                        )

                # Add reference line
                if (
                    ref_col in file_data.columns
                    and elapsed_ref_col in file_data.columns
                ):
                    ref_data = file_data[[elapsed_ref_col, ref_col]].dropna()
                    if not ref_data.empty:
                        ref_data = ref_data.sort_values(elapsed_ref_col)
                        fig.add_trace(
                            go.Scatter(
                                x=ref_data[elapsed_ref_col],
                                y=ref_data[ref_col],
                                mode="lines",
                                name=f"{start_datetime} (ref)",
                                opacity=0.8,
                                line=dict(width=2),
                            )
                        )

    fig.update_layout(
        xaxis_title="Elapsed Seconds",
        yaxis_title=metric.replace("_", " ").title(),
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
