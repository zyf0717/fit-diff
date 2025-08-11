"""
Statistical analysis utilities.
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats


def calculate_basic_stats(
    aligned_df: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Calculate basic statistics for test and reference data using aligned data."""
    if aligned_df is None or aligned_df.empty:
        return None

    stats_list = []

    # Calculate stats for test data (from aligned data)
    test_metric = aligned_df[f"{metric}_test"]
    if not test_metric.empty:
        test_stats = {
            "device": "test",
            "metric": metric,
            "count": test_metric.count(),
            "mean": test_metric.mean(),
            "std": test_metric.std(),
            "min": test_metric.min(),
            "max": test_metric.max(),
            "median": test_metric.median(),
        }
        stats_list.append(test_stats)

    # Calculate stats for reference data (from aligned data)
    ref_metric = aligned_df[f"{metric}_ref"]
    if not ref_metric.empty:
        ref_stats = {
            "device": "reference",
            "metric": metric,
            "count": ref_metric.count(),
            "mean": ref_metric.mean(),
            "std": ref_metric.std(),
            "min": ref_metric.min(),
            "max": ref_metric.max(),
            "median": ref_metric.median(),
        }
        stats_list.append(ref_stats)

    if not stats_list:
        return None

    df = pd.DataFrame(stats_list).round(2)

    # Set 'device' as columns, 'stat' as index
    df_pivot = df.set_index("device").T
    df_pivot = df_pivot.reset_index().rename(columns={"index": "stat"})
    # Ensure columns are in order: stat, test, reference
    cols = ["stat"]
    for dev in ["test", "reference"]:
        if dev in df_pivot.columns:
            cols.append(dev)
    df_pivot = df_pivot[cols]
    return df_pivot


def get_bias_agreement_stats(
    aligned_df: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Get bias, agreement and normality-aware test selection."""
    if aligned_df is None or aligned_df.empty:
        return None

    # Compute errors
    errors = aligned_df[f"{metric}_test"] - aligned_df[f"{metric}_ref"]
    n_points = len(errors)

    # Descriptive moments on errors
    bias = errors.mean()
    std_err = errors.std()

    # Normality test: Shapiro–Wilk
    _, sw_p = stats.shapiro(errors)

    # Select inferential test
    if sw_p > 0.05:
        test_name = "Paired t-test"
        t_stat, p_val = stats.ttest_1samp(errors, 0.0)
    else:
        test_name = "Wilcoxon signed-rank"
        # zero_method='wilcox' drops zero-differences for scipy ≥1.7
        t_stat, p_val = stats.wilcoxon(errors, zero_method="wilcox")

    # Effect size
    cohens_d = bias / std_err if std_err > 0 else np.nan

    # Assemble results
    rows = [
        ("Mean Bias", round(bias, 6)),
        ("Shapiro–Wilk p-value", round(sw_p, 6)),
        (f"{test_name} statistic", round(t_stat, 6)),
        (f"{test_name} p-value", round(p_val, 8)),
        ("Cohen's d", round(cohens_d, 6) if not np.isnan(cohens_d) else "N/A"),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    return df


def get_error_magnitude_stats(
    aligned_df: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Get error magnitude statistics."""
    if aligned_df is None or aligned_df.empty:
        return None

    test_aligned = aligned_df[f"{metric}_test"]
    ref_aligned = aligned_df[f"{metric}_ref"]
    errors = test_aligned - ref_aligned

    # Calculate error magnitude metrics
    mae = errors.abs().mean()
    mse = (errors**2).mean()
    rmse = mse**0.5
    std_errors = errors.std()

    error_magnitude_stats = {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "MSE": round(mse, 6),
        "Std of Errors": round(std_errors, 6),
    }

    # Convert to transposed DataFrame
    df = pd.DataFrame(list(error_magnitude_stats.items()), columns=["Metric", "Value"])
    return df


def get_correlation_stats(
    aligned_df: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Get correlation statistics."""
    if aligned_df is None or aligned_df.empty:
        return None

    test_aligned = aligned_df[f"{metric}_test"]
    ref_aligned = aligned_df[f"{metric}_ref"]

    # Calculate correlation metrics
    correlation = test_aligned.corr(ref_aligned)
    _, r_p_value = stats.pearsonr(test_aligned, ref_aligned)

    correlation_stats = {
        "Correlation Coefficient": round(correlation, 6),
        "Correlation P-value": round(r_p_value, 8),
    }

    # Convert to transposed DataFrame
    df = pd.DataFrame(list(correlation_stats.items()), columns=["Metric", "Value"])
    return df
