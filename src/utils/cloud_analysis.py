"""Cloud pair loading, alignment, caching, and summary helpers."""

import logging

import numpy as np
import pandas as pd

from src.utils import (
    calculate_ccc,
    determine_optimal_shift,
    get_cached_cloud_pair_common_metrics,
    prepare_data_for_analysis,
    process_file,
    put_cached_cloud_pair_common_metrics,
)

logger = logging.getLogger(__name__)


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
        for _pair_id, entry in missing_pair_data.items():
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


def load_cached_common_metrics(
    pair_df: pd.DataFrame,
    selected_pair_ids: list[str],
    db_path,
) -> list[str]:
    """Return shared common metrics for selected pairs using only local cache."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty or not selected_pair_ids:
        return []

    selected_pairs_df = pair_df[pair_df["pair_id"].isin(selected_pair_ids)].copy()
    if selected_pairs_df.empty:
        return []

    metric_sets = []
    for row in selected_pairs_df.itertuples(index=False):
        cached_metrics = get_cached_cloud_pair_common_metrics(
            test_etag=row.test_etag,
            ref_etag=row.ref_etag,
            db_path=db_path,
        )
        if cached_metrics is None:
            continue
        metric_sets.append(set(cached_metrics))

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
        "Group": meta.get("group", meta["pairing_group"]),
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
