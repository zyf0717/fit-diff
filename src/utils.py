"""
Supporting utils and functions
"""

from typing import Tuple, Union

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
        df = v[1]  # records DataFrame
        if isinstance(df, str) or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        available_cols = [col for col in required_cols if col in df.columns]
        if len(available_cols) < 2:  # At least timestamp + 1 metric
            continue
        sub_df = df[available_cols].copy()
        sub_df["file"] = k
        relevant_dfs.append(sub_df)

    if not relevant_dfs:
        return None

    combined_df = pd.concat(relevant_dfs, ignore_index=True)
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


# def plot_continuous_line(
#     df: pd.DataFrame,
#     time_col: str,  # Column should already be in datetime type
#     value_col: Union[str, list[str]],
#     value_label: str = None,
# ) -> px.line:
#     if df.empty:
#         return None

#     # 1. Normalize to list
#     cols = [time_col] + (
#         value_col if isinstance(value_col, (list, tuple)) else [value_col]
#     )

#     # 2. Clean NaNs/inf
#     before = len(df)
#     df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
#     dropped = before - len(df)
#     if dropped:
#         logger.warning("Dropped %d rows due to NaN/inf in %s", dropped, cols)

#     if df.empty or len(cols) < 2:
#         return None

#     # 3. Melt always
#     df_long = df.melt(
#         id_vars=[time_col],
#         value_vars=cols[1:],
#         var_name="variable",
#         value_name="value",
#     )

#     # 4. If only one value column, use px.line as before (no mean line)
#     if len(cols) == 2:
#         fig = px.line(
#             df_long,
#             x=time_col,
#             y="value",
#             color=None,
#             markers=True,
#             line_shape="linear",
#         )
#         fig.update_layout(
#             xaxis_title="Time",
#             yaxis_title=value_label,
#         )
#         fig.update_xaxes(dtick=3600 * 1000)
#         return fig

#     # 5. If multiple value columns, use go.Figure for independent y-axes (no mean lines)
#     fig = go.Figure()
#     yaxis_names = []
#     colors = px.colors.qualitative.Plotly
#     for i, var in enumerate(cols[1:]):
#         color = colors[i % len(colors)]
#         yaxis = "y" if i == 0 else f"y{i+1}"
#         yaxis_names.append(yaxis)
#         fig.add_trace(
#             go.Scatter(
#                 x=df[time_col],
#                 y=df[var],
#                 name=var,
#                 marker_color=color,
#                 yaxis=yaxis,
#                 mode="lines+markers",
#             )
#         )
#     # Layout for multiple y-axes: hide all y-axes and their titles/units
#     layout = dict(
#         xaxis=dict(title="Time", dtick=3600 * 1000),
#         legend=dict(title="Variable"),
#     )
#     # Hide all y-axes and their titles
#     for i, var in enumerate(cols[1:]):
#         yaxis = "yaxis" if i == 0 else f"yaxis{i+1}"
#         layout[yaxis] = dict(
#             title=None,
#             showticklabels=False,
#             showgrid=False,
#             visible=False,
#             anchor="x",
#             overlaying="y" if i > 0 else None,
#         )
#     fig.update_layout(**layout)
#     return fig
