"""Cloud Storage reactive functions for manifest-driven S3 pair analysis."""

import logging
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import Inputs, reactive, render, ui
from shiny.types import SilentException
from shinywidgets import output_widget, render_widget

from src.utils import (
    apply_plotly_theme,
    calculate_ccc,
    determine_optimal_shift,
    get_cached_cloud_pair_common_metrics,
    get_cached_cloud_pair_summary,
    get_cloud_cache_db_path,
    get_plotly_theme,
    init_cloud_cache,
    prepare_data_for_analysis,
    process_file,
    put_cached_cloud_pair_common_metrics,
    put_cached_cloud_pair_summary,
    read_catalogue,
)

logger = logging.getLogger(__name__)

PAIR_MANIFEST_COLUMNS = [
    "pair_id",
    "pair_label",
    "pairing_group",
    "pair_index",
    "date",
    "paired_overlap_pct",
    "test_etag",
    "test_filename",
    "test_s3_key",
    "ref_etag",
    "ref_filename",
    "ref_s3_key",
]

PAIR_SELECTION_TABLE_COLUMNS = [
    "Group",
    "Date",
    "Test File",
    "Ref File",
    "Overlap (%)",
]
RANGE_PLOT_SPECS = [
    {
        "metric_name": "Mean Bias",
        "card_title": "Mean Bias",
        "output_id": "cloudMeanBiasRangePlot",
        "benchmark_indicator": 10.0,
    },
    {
        "metric_name": "MAE",
        "card_title": "Mean Absolute Error",
        "output_id": "cloudMaeRangePlot",
        "benchmark_indicator": 10.0,
    },
    {
        "metric_name": "MSE",
        "card_title": "Mean Squared Error",
        "output_id": "cloudMseRangePlot",
        "benchmark_indicator": 25.0,
    },
    {
        "metric_name": "MAPE (%)",
        "card_title": "Mean Absolute Percentage Error (%)",
        "output_id": "cloudMapeRangePlot",
        "benchmark_indicator": 5.0,
    },
    {
        "metric_name": "CCC",
        "card_title": "Concordance Correlation Coefficient",
        "output_id": "cloudCccRangePlot",
        "benchmark_indicator": 0.90,
    },
    {
        "metric_name": "Pearson Corr",
        "card_title": "Pearson Correlation",
        "output_id": "cloudPearsonCorrRangePlot",
        "benchmark_indicator": 0.90,
    },
    {
        "metric_name": "LoA Lower",
        "card_title": "Lower Limit of Agreement",
        "output_id": "cloudLoaLowerRangePlot",
        "benchmark_indicator": -10.0,
    },
    {
        "metric_name": "LoA Upper",
        "card_title": "Upper Limit of Agreement",
        "output_id": "cloudLoaUpperRangePlot",
        "benchmark_indicator": 10.0,
    },
]


def _coerce_manifest_date(value):
    """Coerce manifest dates to date objects for filtering."""
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _coerce_input_date(value):
    """Coerce Shiny date inputs to date objects."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def get_cloud_manifest_groups(pair_df: pd.DataFrame) -> list[str]:
    """Return available pairing groups from the manifest."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return []
    return sorted(
        group
        for group in pair_df["pairing_group"].dropna().astype(str).unique().tolist()
        if group
    )


def filter_cloud_pair_manifest(
    pair_df: pd.DataFrame,
    selected_groups=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Filter manifest pairs by group and inclusive date range."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    filtered_df = pair_df.copy()
    filtered_df["_pair_date"] = filtered_df["date"].apply(_coerce_manifest_date)

    if selected_groups is not None:
        groups = [str(group) for group in selected_groups if group]
        if not groups:
            return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)
        filtered_df = filtered_df[filtered_df["pairing_group"].astype(str).isin(groups)]

    start = _coerce_input_date(start_date)
    if start is not None:
        filtered_df = filtered_df[
            filtered_df["_pair_date"].notna() & (filtered_df["_pair_date"] >= start)
        ]

    end = _coerce_input_date(end_date)
    if end is not None:
        filtered_df = filtered_df[
            filtered_df["_pair_date"].notna() & (filtered_df["_pair_date"] <= end)
        ]

    return (
        filtered_df.drop(columns="_pair_date")
        .sort_values(
            ["pairing_group", "date", "pair_index", "test_filename", "ref_filename"]
        )
        .reset_index(drop=True)
    )


def get_cloud_manifest_date_bounds(pair_df: pd.DataFrame):
    """Return inclusive date bounds for the filtered manifest."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return None, None

    pair_dates = pair_df["date"].apply(_coerce_manifest_date).dropna()
    if pair_dates.empty:
        return None, None

    return pair_dates.min(), pair_dates.max()


def build_cloud_pair_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Build one manifest row per test/reference pair."""
    if not isinstance(manifest_df, pd.DataFrame) or manifest_df.empty:
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    required_columns = {
        "etag",
        "paired_etag",
        "pairing_group",
        "pair_index",
        "date",
        "filename",
        "device_type",
        "s3_key",
    }
    if not required_columns.issubset(manifest_df.columns):
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    test_rows = manifest_df[
        (manifest_df["device_type"] == "test") & manifest_df["paired_etag"].notna()
    ].copy()
    if test_rows.empty:
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    ref_rows = (
        manifest_df[manifest_df["device_type"] == "ref"][["etag", "filename", "s3_key"]]
        .rename(
            columns={
                "etag": "ref_etag",
                "filename": "ref_filename",
                "s3_key": "ref_s3_key",
            }
        )
        .copy()
    )

    pair_df = test_rows.merge(
        ref_rows, left_on="paired_etag", right_on="ref_etag", how="inner"
    )
    if pair_df.empty:
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    pair_df = pair_df.rename(
        columns={
            "etag": "test_etag",
            "filename": "test_filename",
            "s3_key": "test_s3_key",
        }
    )
    pair_df["pair_id"] = pair_df.apply(
        lambda row: (
            f"{row['pairing_group']}:{int(row['pair_index']) if pd.notna(row['pair_index']) else 'na'}:"
            f"{row['test_etag']}:{row['ref_etag']}"
        ),
        axis=1,
    )
    pair_df["pair_label"] = pair_df.apply(
        lambda row: (
            f"{row['pairing_group']} | {row['date']} | "
            f"{row['test_filename']} <> {row['ref_filename']}"
        ),
        axis=1,
    )
    if "paired_overlap_pct" not in pair_df.columns:
        pair_df["paired_overlap_pct"] = np.nan
    pair_df = pair_df.loc[:, PAIR_MANIFEST_COLUMNS].sort_values(
        ["pairing_group", "date", "pair_index", "test_filename", "ref_filename"]
    )
    pair_df = pair_df.reset_index(drop=True)
    return pair_df


def build_cloud_pair_selection_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Build a sortable display table for cloud pair selection."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return pd.DataFrame(columns=PAIR_SELECTION_TABLE_COLUMNS)

    overlap_pct = (
        (pair_df["paired_overlap_pct"] * 100).round(2)
        if "paired_overlap_pct" in pair_df.columns
        else np.nan
    )
    selection_df = pd.DataFrame(
        {
            "Group": pair_df["pairing_group"],
            "Date": pair_df["date"],
            "Test File": pair_df["test_filename"],
            "Ref File": pair_df["ref_filename"],
            "Overlap (%)": overlap_pct,
        }
    )
    return selection_df.reset_index(drop=True)


def get_selected_cloud_pair_ids(pair_df: pd.DataFrame, selected_rows) -> list[str]:
    """Map selected grid row indices back to pair ids."""
    if (
        not isinstance(pair_df, pd.DataFrame)
        or pair_df.empty
        or not selected_rows
        or "pair_id" not in pair_df.columns
    ):
        return []

    selected_indices = []
    for row_index in selected_rows:
        try:
            selected_indices.append(int(row_index))
        except (TypeError, ValueError):
            continue

    valid_indices = [
        row_index for row_index in selected_indices if 0 <= row_index < len(pair_df)
    ]
    if not valid_indices:
        return []

    return pair_df.iloc[valid_indices]["pair_id"].tolist()


def compute_pair_summary(aligned_df: pd.DataFrame, metric: str) -> dict:
    """Compute compact summary metrics for one aligned pair."""
    test_col = f"{metric}_test"
    ref_col = f"{metric}_ref"
    errors = aligned_df[test_col] - aligned_df[ref_col]
    bias = errors.mean()
    mae = errors.abs().mean()
    mse = (errors**2).mean()

    nonzero_ref = aligned_df[ref_col] != 0
    mape = (
        (errors[nonzero_ref].abs() / aligned_df.loc[nonzero_ref, ref_col].abs()).mean()
        * 100
        if nonzero_ref.any()
        else np.nan
    )

    ccc = calculate_ccc(aligned_df[test_col], aligned_df[ref_col])
    pearson_corr = (
        aligned_df[test_col].corr(aligned_df[ref_col])
        if len(aligned_df) >= 2
        else np.nan
    )
    std_err = errors.std()
    loa_lower = bias - 1.96 * std_err
    loa_upper = bias + 1.96 * std_err

    return {
        "Samples": int(len(aligned_df)),
        "Mean Bias": round(float(bias), 3),
        "MAE": round(float(mae), 3),
        "MSE": round(float(mse), 3),
        "MAPE (%)": round(float(mape), 3) if not np.isnan(mape) else np.nan,
        "CCC": round(float(ccc), 3),
        "Pearson Corr": (
            round(float(pearson_corr), 3) if not np.isnan(pearson_corr) else np.nan
        ),
        "LoA Lower": round(float(loa_lower), 3),
        "LoA Upper": round(float(loa_upper), 3),
    }


def load_cloud_pair_data(
    pair_df: pd.DataFrame,
    selected_pair_ids: list[str],
    bucket_name: str | None,
    aws_profile: str | None,
) -> dict:
    """Load selected cloud pair data from S3 for uncached computations."""
    if (
        not isinstance(pair_df, pd.DataFrame)
        or pair_df.empty
        or not selected_pair_ids
        or not bucket_name
    ):
        return {}

    selected_pairs = pair_df[pair_df["pair_id"].isin(selected_pair_ids)]
    pair_data = {}
    for row in selected_pairs.itertuples(index=False):
        try:
            test_url = f"s3://{bucket_name}/{row.test_s3_key}"
            ref_url = f"s3://{bucket_name}/{row.ref_s3_key}"
            test_df = process_file(test_url, aws_profile=aws_profile)["records"].copy()
            ref_df = process_file(ref_url, aws_profile=aws_profile)["records"].copy()
            test_df["filename"] = row.test_filename
            ref_df["filename"] = row.ref_filename

            common_metrics = sorted(
                set(test_df.columns).intersection(ref_df.columns)
                - {"timestamp", "filename"}
            )
            pair_data[row.pair_id] = {
                "meta": row._asdict(),
                "test_df": test_df,
                "ref_df": ref_df,
                "common_metrics": common_metrics,
            }
        except Exception as exc:
            logger.error("Error processing cloud pair %s: %s", row.pair_id, exc)
            pair_data[row.pair_id] = {
                "meta": row._asdict(),
                "error": str(exc),
                "common_metrics": [],
            }
    return pair_data


def load_cached_or_compute_common_metrics(
    pair_df: pd.DataFrame,
    selected_pair_ids: list[str],
    bucket_name: str | None,
    aws_profile: str | None,
    db_path,
) -> list[str]:
    """Return shared common metrics for selected pairs using cache first."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty or not selected_pair_ids:
        return []

    selected_pairs_df = pair_df[pair_df["pair_id"].isin(selected_pair_ids)].copy()
    if selected_pairs_df.empty:
        return []

    metric_sets = []
    missing_pair_ids = []
    for row in selected_pairs_df.itertuples(index=False):
        cached_metrics = get_cached_cloud_pair_common_metrics(
            test_etag=row.test_etag,
            ref_etag=row.ref_etag,
            db_path=db_path,
        )
        if cached_metrics is None:
            missing_pair_ids.append(row.pair_id)
            continue
        metric_sets.append(set(cached_metrics))

    if missing_pair_ids:
        missing_pair_data = load_cloud_pair_data(
            pair_df,
            missing_pair_ids,
            bucket_name,
            aws_profile,
        )
        for pair_id, entry in missing_pair_data.items():
            common_metrics = entry.get("common_metrics", [])
            if common_metrics:
                meta = entry["meta"]
                put_cached_cloud_pair_common_metrics(
                    test_etag=meta["test_etag"],
                    ref_etag=meta["ref_etag"],
                    common_metrics=common_metrics,
                    db_path=db_path,
                )
                metric_sets.append(set(common_metrics))

    if not metric_sets:
        return []
    return sorted(set.intersection(*metric_sets))


def build_cloud_pair_result_row(
    pair_id: str,
    entry: dict,
    metric: str,
    auto_shift_method: str,
) -> dict:
    """Build one per-pair result row for the cloud summary and plots."""
    meta = entry["meta"]
    base_row = {
        "pair_id": pair_id,
        "Group": meta["pairing_group"],
        "Date": meta["date"],
        "Test File": meta["test_filename"],
        "Ref File": meta["ref_filename"],
        "Metric": metric,
        "Auto-shift": auto_shift_method,
    }

    if entry.get("error"):
        return {**base_row, "Status": f"Error: {entry['error']}"}

    if metric not in entry["common_metrics"]:
        return {**base_row, "Status": "Metric not available for pair"}

    aligned_df, applied_shift = align_pair_data(
        entry["test_df"],
        entry["ref_df"],
        metric,
        auto_shift_method=auto_shift_method,
    )
    if aligned_df.empty:
        return {**base_row, "Status": "No aligned overlapping data"}

    return {
        **base_row,
        "Status": "OK",
        "Applied Shift (s)": applied_shift,
        **compute_pair_summary(aligned_df, metric),
    }


def build_cloud_pair_results(
    pair_data: dict,
    metric: str,
    auto_shift_method: str,
) -> pd.DataFrame:
    """Build per-pair result rows for the cloud summary and range plots."""
    if not pair_data or not metric:
        return pd.DataFrame()

    rows = []
    for pair_id, entry in pair_data.items():
        rows.append(
            build_cloud_pair_result_row(pair_id, entry, metric, auto_shift_method)
        )

    return pd.DataFrame(rows)


def create_cloud_metric_range_plot(
    results_df: pd.DataFrame,
    metric_name: str,
    benchmark_indicator: float | None = None,
    theme_settings=None,
    selected_pair_id: str | None = None,
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
    selected_marker_color = "#8ec5ff" if theme_mode == "dark" else "#084298"

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
    else:
        marker_colors = default_marker_color
        marker_sizes = 10
        marker_line_widths = 1

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
                "line": {"width": marker_line_widths},
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


def _normalize_shift_value(shift_value):
    """Normalize single/list shift outputs to one numeric shift value."""
    if isinstance(shift_value, list):
        if not shift_value:
            return 0
        shift_value = shift_value[0]
    if shift_value is None or pd.isna(shift_value):
        return 0
    return float(shift_value)


def align_pair_data(
    test_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    metric: str,
    auto_shift_method: str = "None (manual)",
):
    """Apply the Local Files preparation/alignment path to one test/reference pair."""
    prepared = prepare_data_for_analysis((test_df, ref_df), metric)
    if prepared is None:
        return pd.DataFrame(), 0

    test_data, ref_data = prepared
    if test_data.empty or ref_data.empty:
        return pd.DataFrame(), 0

    applied_shift = 0
    if auto_shift_method and "None" not in auto_shift_method:
        applied_shift = _normalize_shift_value(
            determine_optimal_shift(test_data, ref_data, metric, auto_shift_method)
        )
        if applied_shift != 0:
            test_data = test_data.copy()
            test_data["timestamp"] = test_data["timestamp"] + applied_shift
            if "elapsed_seconds" in test_data.columns:
                test_data["elapsed_seconds"] = (
                    test_data["elapsed_seconds"] + applied_shift
                )

    test_clean = (
        test_data[["timestamp", "filename", metric]].dropna().reset_index(drop=True)
    )
    ref_clean = ref_data[["timestamp", metric]].dropna().reset_index(drop=True)

    aligned_df = pd.merge(
        test_clean,
        ref_clean,
        on="timestamp",
        suffixes=("_test", "_ref"),
    ).reset_index(drop=True)
    return aligned_df, applied_shift


def create_cloud_storage_reactives(
    inputs: Inputs, session=None, local_pair_override=None
):
    """Create Cloud Storage reactive functions."""

    bucket_name = os.getenv("S3_BUCKET")
    aws_profile = os.getenv("AWS_PROFILE")
    manifest_key = os.getenv("CATALOGUE_CSV_KEY")
    cloud_cache_db_path = get_cloud_cache_db_path()
    init_cloud_cache(cloud_cache_db_path)
    cloud_analysis_request = reactive.Value(None)
    selected_cloud_plot_pair_id = reactive.Value(None)

    def _safe_input(input_name, default=None):
        try:
            input_obj = getattr(inputs, input_name, None)
            if input_obj is None:
                return default
            return input_obj()
        except SilentException:
            return default

    @reactive.Calc
    def _cloud_manifest():
        manifest_df = read_catalogue()
        if not isinstance(manifest_df, pd.DataFrame):
            return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)
        return build_cloud_pair_manifest(manifest_df)

    @render.ui
    def cloudGroupSelector():
        pair_df = _cloud_manifest()
        if pair_df.empty:
            return None

        choices = get_cloud_manifest_groups(pair_df)
        return ui.input_selectize(
            "selected_cloud_groups",
            "Filter groups:",
            choices=choices,
            selected=choices,
            multiple=True,
        )

    @render.ui
    def cloudDateRangeSelector():
        group_filtered_df = filter_cloud_pair_manifest(
            _cloud_manifest(),
            selected_groups=_safe_input("selected_cloud_groups"),
        )
        start_date, end_date = get_cloud_manifest_date_bounds(group_filtered_df)
        if start_date is None or end_date is None:
            return None

        selected_range = _safe_input("selected_cloud_date_range")
        selected_start = start_date
        selected_end = end_date
        if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
            selected_start = _coerce_input_date(selected_range[0]) or start_date
            selected_end = _coerce_input_date(selected_range[1]) or end_date

        return ui.input_date_range(
            "selected_cloud_date_range",
            "Filter date range:",
            start=selected_start,
            end=selected_end,
            min=start_date,
            max=end_date,
        )

    @reactive.Calc
    def _filtered_cloud_manifest():
        selected_range = _safe_input("selected_cloud_date_range")
        start_date = (
            selected_range[0]
            if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2
            else None
        )
        end_date = (
            selected_range[1]
            if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2
            else None
        )
        return filter_cloud_pair_manifest(
            _cloud_manifest(),
            selected_groups=_safe_input("selected_cloud_groups"),
            start_date=start_date,
            end_date=end_date,
        )

    @render.data_frame
    def cloudPairSelectionTable():
        pair_df = _filtered_cloud_manifest()
        if pair_df.empty:
            return pd.DataFrame(columns=PAIR_SELECTION_TABLE_COLUMNS)

        return render.DataGrid(
            build_cloud_pair_selection_table(pair_df),
            selection_mode="rows",
        )

    @reactive.Calc
    def _selected_cloud_pair_ids():
        return get_selected_cloud_pair_ids(
            _filtered_cloud_manifest(),
            _safe_input("cloudPairSelectionTable_selected_rows", []),
        )

    @reactive.Effect
    @reactive.event(inputs.cloudRefreshAnalysis)
    def _capture_cloud_analysis_request():
        cloud_analysis_request.set(
            {
                "pair_df": _filtered_cloud_manifest().copy(),
                "selected_pair_ids": _selected_cloud_pair_ids(),
                "metric": _safe_input("cloud_comparison_metric", "heart_rate"),
                "auto_shift_method": _safe_input(
                    "cloud_auto_shift_method", "Minimize MAE"
                ),
            }
        )

    @reactive.Calc
    def _cloud_analysis_request():
        return cloud_analysis_request.get()

    def _build_local_pair_override(pair_id: str, request: dict):
        pair_data = load_cloud_pair_data(
            request["pair_df"],
            [pair_id],
            bucket_name,
            aws_profile,
        )
        entry = pair_data.get(pair_id)
        if not entry or entry.get("error"):
            return None

        return {
            "pair_id": pair_id,
            "metric": request["metric"],
            "auto_shift_method": request["auto_shift_method"],
            "test_filename": entry["meta"]["test_filename"],
            "ref_filename": entry["meta"]["ref_filename"],
            "test_df": entry["test_df"].copy(),
            "ref_df": entry["ref_df"].copy(),
        }

    @reactive.Calc
    def _selected_cloud_pair_data():
        request = _cloud_analysis_request()
        if request is None:
            return {}
        return load_cloud_pair_data(
            request["pair_df"],
            request["selected_pair_ids"],
            bucket_name,
            aws_profile,
        )

    @reactive.Calc
    def _cloud_common_metrics():
        return load_cached_or_compute_common_metrics(
            _filtered_cloud_manifest(),
            _selected_cloud_pair_ids(),
            bucket_name,
            aws_profile,
            cloud_cache_db_path,
        )

    @render.ui
    def cloudMetricSelector():
        choices = _cloud_common_metrics()
        if not choices:
            choices = ["heart_rate"]

        selected = "heart_rate" if "heart_rate" in choices else choices[0]
        return ui.input_select(
            "cloud_comparison_metric",
            "Select comparison metric:",
            choices=choices,
            selected=selected,
        )

    @render.ui
    def cloudAutoShiftSelector():
        return ui.input_select(
            "cloud_auto_shift_method",
            "Auto-shift by:",
            choices=[
                "None (manual)",
                "Minimize MAE",
                "Minimize MSE",
                "Maximize Concordance Correlation",
                "Maximize Pearson Correlation",
            ],
            selected="Minimize MAE",
        )

    @reactive.Calc
    def _cloud_pair_results():
        request = _cloud_analysis_request()
        if request is None:
            return pd.DataFrame()
        pair_df = request["pair_df"]
        selected_pair_ids = request["selected_pair_ids"]
        metric = request["metric"]
        auto_shift_method = request["auto_shift_method"]

        if pair_df.empty or not selected_pair_ids or not metric:
            return pd.DataFrame()

        selected_pairs_df = pair_df[pair_df["pair_id"].isin(selected_pair_ids)].copy()
        if selected_pairs_df.empty:
            return pd.DataFrame()

        cached_rows_by_pair_id = {}
        missing_pair_ids = []
        for row in selected_pairs_df.itertuples(index=False):
            cached_row = get_cached_cloud_pair_summary(
                test_etag=row.test_etag,
                ref_etag=row.ref_etag,
                metric=metric,
                auto_shift_method=auto_shift_method,
                db_path=cloud_cache_db_path,
            )
            if cached_row is None:
                missing_pair_ids.append(row.pair_id)
            else:
                cached_rows_by_pair_id[row.pair_id] = cached_row

        computed_rows_by_pair_id = {}
        if missing_pair_ids:
            missing_pair_data = load_cloud_pair_data(
                pair_df,
                missing_pair_ids,
                bucket_name,
                aws_profile,
            )
            for pair_id, entry in missing_pair_data.items():
                result_row = build_cloud_pair_result_row(
                    pair_id,
                    entry,
                    metric,
                    auto_shift_method,
                )
                computed_rows_by_pair_id[pair_id] = result_row
                if result_row.get("Status") == "OK":
                    meta = entry["meta"]
                    put_cached_cloud_pair_summary(
                        test_etag=meta["test_etag"],
                        ref_etag=meta["ref_etag"],
                        metric=metric,
                        auto_shift_method=auto_shift_method,
                        result_row=result_row,
                        common_metrics=entry.get("common_metrics"),
                        db_path=cloud_cache_db_path,
                    )

        ordered_rows = []
        for row in selected_pairs_df.itertuples(index=False):
            if row.pair_id in cached_rows_by_pair_id:
                ordered_rows.append(cached_rows_by_pair_id[row.pair_id])
            elif row.pair_id in computed_rows_by_pair_id:
                ordered_rows.append(computed_rows_by_pair_id[row.pair_id])

        return pd.DataFrame(ordered_rows)

    @reactive.Effect
    def _clear_selected_cloud_pair_if_missing():
        results_df = _cloud_pair_results()
        current_pair_id = selected_cloud_plot_pair_id.get()
        if current_pair_id is None:
            return
        if (
            results_df.empty
            or current_pair_id not in results_df.get("pair_id", []).tolist()
        ):
            selected_cloud_plot_pair_id.set(None)

    @render.data_frame
    def cloudPairSummaryTable():
        result_df = _cloud_pair_results()
        if result_df.empty:
            return pd.DataFrame()
        return render.DataGrid(
            result_df.drop(columns=["pair_id"], errors="ignore"),
            selection_mode="none",
        )

    @render.ui
    def cloudMetricRangePlotGrid():
        cards = []
        for spec in RANGE_PLOT_SPECS:
            cards.append(
                ui.card(
                    ui.card_header(spec["card_title"]),
                    output_widget(
                        spec["output_id"],
                        height="150px",
                        fill=True,
                    ),
                )
            )
        return ui.layout_columns(*cards, col_widths=[6] * len(cards))

    def _make_cloud_metric_plot_renderer(
        output_id: str,
        metric_name: str,
        benchmark_indicator: float | None = None,
    ):
        def _plot():
            results_df = _cloud_pair_results()
            request = _cloud_analysis_request()
            figure = create_cloud_metric_range_plot(
                results_df,
                metric_name,
                benchmark_indicator=benchmark_indicator,
                theme_settings=_safe_input("plotly_theme"),
                selected_pair_id=selected_cloud_plot_pair_id.get(),
            )

            figure_widget = go.FigureWidget(figure)
            if "Status" not in results_df.columns:
                plot_df = pd.DataFrame()
            else:
                plot_df = results_df[results_df["Status"] == "OK"].reset_index(
                    drop=True
                )
            marker_trace = next(
                (
                    trace
                    for trace in figure_widget.data
                    if getattr(trace, "type", None) == "scatter"
                    and getattr(trace, "mode", None) == "markers"
                ),
                None,
            )

            if marker_trace is not None and not plot_df.empty and request is not None:

                def _handle_click(_trace, points, _selector):
                    if not points.point_inds:
                        return

                    point_index = points.point_inds[0]
                    if point_index >= len(plot_df):
                        return

                    pair_id = plot_df.iloc[point_index].get("pair_id")
                    if not pair_id:
                        return

                    selected_cloud_plot_pair_id.set(pair_id)
                    if local_pair_override is not None:
                        pair_override = _build_local_pair_override(pair_id, request)
                        if pair_override is not None:
                            local_pair_override.set(pair_override)
                    if session is not None:
                        ui.update_navs(
                            "mainNavset",
                            selected="Benchmarking",
                            session=session,
                        )

                marker_trace.on_click(_handle_click)

            return figure_widget

        _plot.__name__ = output_id
        return render_widget(_plot)

    cloud_metric_plot_renderers = {
        spec["output_id"]: _make_cloud_metric_plot_renderer(
            spec["output_id"],
            spec["metric_name"],
            benchmark_indicator=spec["benchmark_indicator"],
        )
        for spec in RANGE_PLOT_SPECS
    }

    return {
        "_cloud_manifest": _cloud_manifest,
        "_filtered_cloud_manifest": _filtered_cloud_manifest,
        "_cloud_analysis_request": _cloud_analysis_request,
        "_selected_cloud_pair_ids": _selected_cloud_pair_ids,
        "_selected_cloud_pair_data": _selected_cloud_pair_data,
        "_cloud_common_metrics": _cloud_common_metrics,
        "_cloud_pair_results": _cloud_pair_results,
        "cloudGroupSelector": cloudGroupSelector,
        "cloudDateRangeSelector": cloudDateRangeSelector,
        "cloudPairSelectionTable": cloudPairSelectionTable,
        "cloudMetricSelector": cloudMetricSelector,
        "cloudAutoShiftSelector": cloudAutoShiftSelector,
        "cloudMetricRangePlotGrid": cloudMetricRangePlotGrid,
        **cloud_metric_plot_renderers,
        "cloudPairSummaryTable": cloudPairSummaryTable,
    }
