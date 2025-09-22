"""
Statistical analysis utilities.
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats


def format_p_value(p_value: float) -> str:
    """
    Format p-value according to standard conventions:
    - If p < 0.001: return "<0.001"
    - Otherwise: return p-value with 3 decimal places

    Args:
        p_value: The p-value to format

    Returns:
        Formatted p-value string
    """
    if np.isnan(p_value):
        return "N/A"
    if p_value < 0.001:
        return "<0.001"
    else:
        return f"{p_value:.3f}"


def format_number(value: float, decimal_places: int) -> str:
    """
    Format a number with fixed decimal places, padding with zeros if necessary.

    Args:
        value: The number to format
        decimal_places: Number of decimal places to show

    Returns:
        Formatted number string with zero padding
    """
    if np.isnan(value):
        return "N/A"
    return f"{value:.{decimal_places}f}"


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

    df = pd.DataFrame(stats_list).round(3)

    # Set 'device' as columns, 'stat' as index
    df_pivot = df.set_index("device").T
    df_pivot = df_pivot.reset_index().rename(columns={"index": metric})
    # Ensure columns are in order: stat, test, reference
    cols = [metric]
    for dev in ["test", "reference"]:
        if dev in df_pivot.columns:
            cols.append(dev)
    df_pivot = df_pivot[cols]
    return df_pivot


def get_bias_stats(aligned_df: pd.DataFrame, metric: str) -> Union[pd.DataFrame, None]:
    """Get bias and agreement statistics using multiple statistical tests."""
    if aligned_df is None or aligned_df.empty:
        return None

    # Compute errors
    errors = aligned_df[f"{metric}_test"] - aligned_df[f"{metric}_ref"]
    n_points = len(errors)

    # Descriptive moments on errors
    bias = errors.mean()
    std_err = errors.std()

    # Two-sample K–S test
    test_metric = aligned_df[f"{metric}_test"]
    ref_metric = aligned_df[f"{metric}_ref"]
    ks_stat, ks_p_val = stats.ks_2samp(test_metric, ref_metric)

    # Paired t-test
    t_stat, t_p_val = stats.ttest_1samp(errors, 0.0)

    # Wilcoxon signed-rank test
    # zero_method='wilcox' drops zero-differences for scipy ≥1.7
    w_stat, w_p_val = stats.wilcoxon(errors, zero_method="wilcox")

    # Sign test
    # Count positive and negative differences (excluding zeros)
    non_zero_errors = errors[errors != 0]
    n_positive = (non_zero_errors > 0).sum()
    n_negative = (non_zero_errors < 0).sum()
    n_total = len(non_zero_errors)

    if n_total > 0:
        sign_p_val = stats.binomtest(
            n_positive, n_total, 0.5, alternative="two-sided"
        ).pvalue
    else:
        sign_p_val = np.nan

    # Effect size
    cohens_d = bias / std_err if std_err > 0 else np.nan

    # Assemble results
    rows = [
        ("Mean Bias", format_number(bias, 3)),
        # ("Two-sample K–S test p-value", format_p_value(ks_p_val)),
        ("Paired t-test p-value", format_p_value(t_p_val)),
        ("Wilcoxon signed-rank p-value", format_p_value(w_p_val)),
        (
            "Sign test p-value",
            format_p_value(sign_p_val),
        ),
        ("Cohen's d", format_number(cohens_d, 3) if not np.isnan(cohens_d) else "N/A"),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    return df


def get_accuracy_stats(
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
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by masking zero reference values
    nonzero_ref = ref_aligned != 0
    if nonzero_ref.any():
        mape = (errors[nonzero_ref].abs() / ref_aligned[nonzero_ref].abs()).mean() * 100
        mape = format_number(mape, 3)
    else:
        mape = None

    error_magnitude_stats = {
        "MAE": format_number(mae, 3),
        "RMSE": format_number(rmse, 3),
        "MSE": format_number(mse, 3),
        "MAPE (%)": mape,
        "Std of Errors": format_number(std_errors, 3),
    }

    # Convert to transposed DataFrame
    df = pd.DataFrame(list(error_magnitude_stats.items()), columns=["Metric", "Value"])
    return df


def calculate_ccc(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate the Concordance Correlation Coefficient (CCC) using Lin's formula.

    The CCC measures the agreement between two continuous variables.
    CCC = 2 * r * σx * σy / (σx² + σy² + (μx - μy)²)

    Args:
        x: First series of values
        y: Second series of values

    Returns:
        CCC value between -1 and 1, where 1 indicates perfect agreement
    """
    if x.empty or y.empty or len(x) != len(y):
        return 0.0

    # Calculate means and variances
    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var()
    var_y = y.var()

    # Pearson correlation coefficient
    pearson_r = x.corr(y)

    # Handle edge cases
    if pd.isna(pearson_r) or var_x == 0 or var_y == 0:
        return 0.0

    # CCC formula: CCC = 2 * r * σx * σy / (σx² + σy² + (μx - μy)²)
    numerator = 2 * pearson_r * np.sqrt(var_x) * np.sqrt(var_y)
    denominator = var_x + var_y + (mean_x - mean_y) ** 2

    return numerator / denominator if denominator != 0 else 0.0


def get_agreement_stats(
    aligned_df: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Get correlation statistics."""
    if aligned_df is None or aligned_df.empty:
        return None

    test_aligned = aligned_df[f"{metric}_test"]
    ref_aligned = aligned_df[f"{metric}_ref"]

    # Calculate Concordance Correlation Coefficient (CCC)
    ccc = calculate_ccc(test_aligned, ref_aligned)

    # Pearson correlation coefficient
    pearson_r = test_aligned.corr(ref_aligned)

    # Calculate p-value for Pearson correlation (for reference)
    _, r_p_value = stats.pearsonr(test_aligned, ref_aligned)

    # Calculate Limits of Agreement (LoA)
    errors = test_aligned - ref_aligned
    bias = errors.mean()
    std_err = errors.std()
    loa_lower = bias - 1.96 * std_err
    loa_upper = bias + 1.96 * std_err

    agreement_stats = {
        "Concordance Correlation Coefficient": format_number(ccc, 3),
        "Pearson Correlation Coefficient": format_number(pearson_r, 3),
        "Pearson Correlation P-value": format_p_value(r_p_value),
        "LoA Lower": format_number(loa_lower, 3),
        "LoA Upper": format_number(loa_upper, 3),
    }

    # Convert to transposed DataFrame
    df = pd.DataFrame(list(agreement_stats.items()), columns=["Metric", "Value"])
    return df
