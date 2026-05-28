"""Cloud Storage plotting helpers."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.utils import apply_plotly_theme, get_plotly_theme


RANGE_PLOT_SPECS = [
    {
        "metric_name": "Mean Bias",
        "card_title": "Mean Bias",
        "output_id": "cloudMeanBiasRangePlot",
        "stats_output_id": "cloudMeanBiasRangeStats",
        "benchmark_indicator": 10.0,
        "benchmark_direction": "above",
    },
    {
        "metric_name": "MAE",
        "card_title": "Mean Absolute Error",
        "output_id": "cloudMaeRangePlot",
        "stats_output_id": "cloudMaeRangeStats",
        "benchmark_indicator": 10.0,
        "benchmark_direction": "above",
    },
    {
        "metric_name": "MSE",
        "card_title": "Mean Squared Error",
        "output_id": "cloudMseRangePlot",
        "stats_output_id": "cloudMseRangeStats",
        "benchmark_indicator": None,
        "benchmark_direction": None,
    },
    {
        "metric_name": "MAPE (%)",
        "card_title": "Mean Absolute Percentage Error (%)",
        "output_id": "cloudMapeRangePlot",
        "stats_output_id": "cloudMapeRangeStats",
        "benchmark_indicator": 10.0,
        "benchmark_direction": "above",
    },
    {
        "metric_name": "CCC",
        "card_title": "Concordance Correlation Coefficient",
        "output_id": "cloudCccRangePlot",
        "stats_output_id": "cloudCccRangeStats",
        "benchmark_indicator": 0.90,
        "benchmark_direction": "below",
    },
    {
        "metric_name": "Pearson Corr",
        "card_title": "Pearson Correlation",
        "output_id": "cloudPearsonCorrRangePlot",
        "stats_output_id": "cloudPearsonCorrRangeStats",
        "benchmark_indicator": 0.90,
        "benchmark_direction": "below",
    },
    {
        "metric_name": "LoA Lower",
        "card_title": "Lower Limit of Agreement",
        "output_id": "cloudLoaLowerRangePlot",
        "stats_output_id": "cloudLoaLowerRangeStats",
        "benchmark_indicator": -10.0,
        "benchmark_direction": "below",
    },
    {
        "metric_name": "LoA Upper",
        "card_title": "Upper Limit of Agreement",
        "output_id": "cloudLoaUpperRangePlot",
        "stats_output_id": "cloudLoaUpperRangeStats",
        "benchmark_indicator": 10.0,
        "benchmark_direction": "above",
    },
]


def summarize_cloud_metric_range_stats(
    results_df: pd.DataFrame,
    metric_name: str,
    benchmark_indicator: float | None = None,
    benchmark_direction: str | None = None,
):
    """Summarize one cloud metric distribution for the side panel."""
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return None

    if "Status" not in results_df.columns or metric_name not in results_df.columns:
        return None

    plot_df = results_df[results_df["Status"] == "OK"].copy()
    if plot_df.empty:
        return None

    metric_values = pd.to_numeric(plot_df[metric_name], errors="coerce")
    finite_values = metric_values[np.isfinite(metric_values)]
    if finite_values.empty:
        return None

    summary = {
        "count": int(finite_values.size),
        "mean": float(finite_values.mean()),
        "median": float(finite_values.median()),
        "sd": float(finite_values.std(ddof=1)) if finite_values.size > 1 else 0.0,
    }

    if benchmark_indicator is None or not np.isfinite(benchmark_indicator):
        return summary

    benchmark_value = float(benchmark_indicator)
    summary["benchmark_value"] = benchmark_value
    summary["benchmark_direction"] = benchmark_direction
    if benchmark_direction == "below":
        bad_mask = finite_values < benchmark_value
        summary["benchmark_label"] = "Below threshold"
    else:
        bad_mask = finite_values > benchmark_value
        summary["benchmark_label"] = "Above threshold"

    bad_count = int(bad_mask.sum())
    summary["benchmark_exceed_count"] = bad_count
    summary["benchmark_exceed_pct"] = (
        float((bad_count / finite_values.size) * 100.0) if finite_values.size else 0.0
    )
    return summary


def create_cloud_metric_range_plot(
    results_df: pd.DataFrame,
    metric_name: str,
    benchmark_indicator: float | None = None,
    benchmark_direction: str | None = None,
    theme_settings=None,
    selected_pair_id: str | None = None,
    empty_message: str | None = None,
):
    """Create one horizontal range-strip plot for a single cloud summary metric."""

    def _empty_figure():
        fig = go.Figure()
        fig.update_layout(
            height=150,
            margin={"l": 40, "r": 20, "t": 10, "b": 40},
        )
        fig.update_xaxes(visible=False, fixedrange=True)
        fig.update_yaxes(visible=False, fixedrange=True)
        if empty_message:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text=str(empty_message).replace("\n", "<br>"),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                align="center",
                font={"size": 12},
            )
        return apply_plotly_theme(fig, theme_settings)

    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return _empty_figure()

    if "Status" not in results_df.columns or metric_name not in results_df.columns:
        return _empty_figure()

    plot_df = results_df[results_df["Status"] == "OK"].copy()
    if plot_df.empty:
        return _empty_figure()

    hover_text = (
        "Group: "
        + plot_df["Group"].astype(str)
        + "<br>Date: "
        + plot_df["Date"].astype(str)
        + "<br>Test: "
        + plot_df["Test File"].astype(str)
        + "<br>Ref: "
        + plot_df["Ref File"].astype(str)
    )

    metric_values = pd.to_numeric(plot_df[metric_name], errors="coerce")
    finite_values = metric_values[np.isfinite(metric_values)]

    benchmark_value = None
    if benchmark_indicator is not None and np.isfinite(benchmark_indicator):
        benchmark_value = float(benchmark_indicator)
    theme = get_plotly_theme(theme_settings)
    theme_mode = (
        str(theme_settings.get("mode", "light")).lower()
        if hasattr(theme_settings, "get")
        else "light"
    )
    default_marker_color = "#2c7fb8"
    selected_marker_color = "#f8f9fa" if theme_mode == "dark" else "#212529"
    default_marker_line_color = "#ffffff" if theme_mode == "dark" else "#1f4e6d"
    selected_marker_line_color = "#212529" if theme_mode == "dark" else "#f8f9fa"

    if "pair_id" in plot_df.columns:
        pair_ids = plot_df["pair_id"].tolist()
        marker_colors = [
            (
                selected_marker_color
                if pair_id == selected_pair_id
                else default_marker_color
            )
            for pair_id in pair_ids
        ]
        marker_sizes = [
            12 if pair_id == selected_pair_id else 10 for pair_id in pair_ids
        ]
        marker_line_widths = [
            2 if pair_id == selected_pair_id else 1 for pair_id in pair_ids
        ]
        marker_line_colors = [
            (
                selected_marker_line_color
                if pair_id == selected_pair_id
                else default_marker_line_color
            )
            for pair_id in pair_ids
        ]
    else:
        marker_colors = default_marker_color
        marker_sizes = 10
        marker_line_widths = 1
        marker_line_colors = default_marker_line_color

    axis_values = finite_values.tolist()
    if benchmark_value is not None:
        axis_values.append(benchmark_value)

    if not axis_values:
        line_min, line_max = -1.0, 1.0
    else:
        line_min = float(min(axis_values))
        line_max = float(max(axis_values))
        if line_min == line_max:
            padding = max(abs(line_min) * 0.1, 1.0)
            line_min -= padding
            line_max += padding
        if benchmark_value is not None:
            edge_padding = max((line_max - line_min) * 0.05, 0.05)
            if np.isclose(benchmark_value, line_min):
                line_min -= edge_padding
            if np.isclose(benchmark_value, line_max):
                line_max += edge_padding
    tick_values = np.linspace(line_min, line_max, 10).tolist()

    fig = go.Figure()
    if benchmark_value is not None:
        if benchmark_direction == "below":
            shade_start, shade_end = line_min, benchmark_value
        else:
            shade_start, shade_end = benchmark_value, line_max
        if shade_end > shade_start:
            fig.add_vrect(
                x0=shade_start,
                x1=shade_end,
                fillcolor="rgba(214, 39, 40, 0.12)",
                line_width=0,
                layer="below",
            )
    fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[0, 0],
            mode="lines",
            line={"color": theme["muted_color"], "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if not finite_values.empty:
        fig.add_trace(
            go.Box(
                x=finite_values,
                y=[0] * len(finite_values),
                orientation="h",
                width=0.45,
                whiskerwidth=0,
                boxpoints=False,
                fillcolor=theme["box_fill_color"],
                line={"color": theme["box_line_color"], "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=metric_values,
            y=[0] * len(plot_df),
            mode="markers",
            marker={
                "size": marker_sizes,
                "color": marker_colors,
                "line": {
                    "width": marker_line_widths,
                    "color": marker_line_colors,
                },
            },
            customdata=plot_df["pair_id"] if "pair_id" in plot_df.columns else None,
            text=hover_text,
            hovertemplate="%{text}<br>" + metric_name + ": %{x}<extra></extra>",
            showlegend=False,
        )
    )
    if benchmark_value is not None:
        fig.add_trace(
            go.Scatter(
                x=[benchmark_value, benchmark_value],
                y=[-0.5, 0.5],
                mode="lines",
                line={"color": "#d62728", "width": 2, "dash": "dot"},
                hovertemplate=("Benchmark: %{x:.3f}<extra></extra>"),
                showlegend=False,
            )
        )

    fig.update_layout(
        height=150,
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
    )
    fig.update_yaxes(
        visible=False,
        range=[-0.6, 0.6],
        fixedrange=True,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_values,
        tickformat=".3f",
        ticks="outside",
        ticklen=8,
        tickwidth=1,
        showline=True,
        linewidth=1,
        linecolor="#7f7f7f",
        mirror=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )
    return apply_plotly_theme(fig, theme_settings)
