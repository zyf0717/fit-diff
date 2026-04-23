"""Cloud Storage reactive functions for manifest-driven S3 pair analysis."""

import logging
import os

import numpy as np
import pandas as pd
from shiny import Inputs, reactive, render, ui
from shiny.types import SilentException

from src.utils import calculate_ccc, prepare_data_for_analysis, process_file, read_catalogue

logger = logging.getLogger(__name__)

PAIR_MANIFEST_COLUMNS = [
    "pair_id",
    "pair_label",
    "pairing_group",
    "pair_index",
    "date",
    "test_etag",
    "test_filename",
    "test_s3_key",
    "ref_etag",
    "ref_filename",
    "ref_s3_key",
]


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
        manifest_df[manifest_df["device_type"] == "ref"][
            ["etag", "filename", "s3_key"]
        ]
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
    pair_df = pair_df.loc[:, PAIR_MANIFEST_COLUMNS].sort_values(
        ["pairing_group", "date", "pair_index", "test_filename", "ref_filename"]
    )
    pair_df = pair_df.reset_index(drop=True)
    return pair_df


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


def align_pair_data(test_df: pd.DataFrame, ref_df: pd.DataFrame, metric: str):
    """Apply the Local Files preparation/alignment path to one test/reference pair."""
    prepared = prepare_data_for_analysis((test_df, ref_df), metric)
    if prepared is None:
        return pd.DataFrame()

    test_data, ref_data = prepared
    if test_data.empty or ref_data.empty:
        return pd.DataFrame()

    test_clean = test_data[["timestamp", "filename", metric]].dropna().reset_index(
        drop=True
    )
    ref_clean = ref_data[["timestamp", metric]].dropna().reset_index(drop=True)

    aligned_df = pd.merge(
        test_clean,
        ref_clean,
        on="timestamp",
        suffixes=("_test", "_ref"),
    ).reset_index(drop=True)
    return aligned_df


def create_cloud_storage_reactives(inputs: Inputs):
    """Create Cloud Storage reactive functions."""

    bucket_name = os.getenv("S3_BUCKET")
    aws_profile = os.getenv("AWS_PROFILE")
    manifest_key = os.getenv("CATALOGUE_CSV_KEY")

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
    def cloudManifestStatus():
        pair_df = _cloud_manifest()
        if pair_df.empty:
            return ui.p(
                "No paired manifest entries available from the configured cloud storage.",
                style="color: #666;",
            )

        manifest_location = (
            f"s3://{bucket_name}/{manifest_key}" if bucket_name and manifest_key else None
        )
        status_text = (
            f"Loaded {len(pair_df)} manifest-defined pairs from {manifest_location}."
            if manifest_location
            else f"Loaded {len(pair_df)} manifest-defined pairs."
        )
        return ui.p(status_text, style="color: #666;")

    @render.ui
    def cloudPairSelector():
        pair_df = _cloud_manifest()
        if pair_df.empty:
            return None

        choices = dict(zip(pair_df["pair_id"], pair_df["pair_label"]))
        return ui.input_selectize(
            "selected_cloud_pairs",
            "Select manifest pair(s):",
            choices=choices,
            selected=[],
            multiple=True,
        )

    @reactive.Calc
    def _selected_cloud_pair_data():
        pair_df = _cloud_manifest()
        selected_pair_ids = _safe_input("selected_cloud_pairs", []) or []
        if pair_df.empty or not selected_pair_ids or not bucket_name:
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
                    (
                        set(test_df.columns).intersection(ref_df.columns)
                        - {"timestamp", "filename"}
                    )
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

    @reactive.Calc
    def _cloud_common_metrics():
        pair_data = _selected_cloud_pair_data()
        metric_sets = [
            set(entry["common_metrics"])
            for entry in pair_data.values()
            if entry.get("common_metrics")
        ]
        if not metric_sets:
            return []
        return sorted(set.intersection(*metric_sets))

    @render.ui
    def cloudMetricSelector():
        choices = _cloud_common_metrics()
        if not choices:
            return None

        selected = "heart_rate" if "heart_rate" in choices else choices[0]
        return ui.input_select(
            "cloud_comparison_metric",
            "Select comparison metric:",
            choices=choices,
            selected=selected,
        )

    @render.data_frame
    def cloudPairSummaryTable():
        pair_data = _selected_cloud_pair_data()
        metric = _safe_input("cloud_comparison_metric")
        if not pair_data or not metric:
            return pd.DataFrame()

        rows = []
        for pair_id, entry in pair_data.items():
            meta = entry["meta"]
            base_row = {
                "Pair": meta["pair_label"],
                "Group": meta["pairing_group"],
                "Date": meta["date"],
                "Test File": meta["test_filename"],
                "Ref File": meta["ref_filename"],
                "Metric": metric,
            }

            if entry.get("error"):
                rows.append({**base_row, "Status": f"Error: {entry['error']}"})
                continue

            if metric not in entry["common_metrics"]:
                rows.append({**base_row, "Status": "Metric not available for pair"})
                continue

            aligned_df = align_pair_data(entry["test_df"], entry["ref_df"], metric)
            if aligned_df.empty:
                rows.append({**base_row, "Status": "No aligned overlapping data"})
                continue

            rows.append(
                {
                    **base_row,
                    "Status": "OK",
                    **compute_pair_summary(aligned_df, metric),
                }
            )

        result_df = pd.DataFrame(rows)
        if result_df.empty:
            return pd.DataFrame()
        return render.DataGrid(result_df, selection_mode="none")

    return {
        "_cloud_manifest": _cloud_manifest,
        "_selected_cloud_pair_data": _selected_cloud_pair_data,
        "_cloud_common_metrics": _cloud_common_metrics,
        "cloudManifestStatus": cloudManifestStatus,
        "cloudPairSelector": cloudPairSelector,
        "cloudMetricSelector": cloudMetricSelector,
        "cloudPairSummaryTable": cloudPairSummaryTable,
    }
