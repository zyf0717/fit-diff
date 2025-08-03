"""
Supporting utils and functions
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from garmin_fit_sdk import Decoder, Stream
from scipy import stats


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
    record_df["filename"] = str(Path(file_path).name)

    session_df = pd.json_normalize(messages.get("session_mesgs", []), sep="_")
    if session_df.empty:
        raise ValueError("No session messages found in FIT file")
    session_df["filename"] = str(Path(file_path).name)

    return session_df, record_df


def prepare_data_for_analysis(
    all_fit_data: tuple, metric: str
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], None]:
    """Prepare test and reference data for analysis, keeping them separate."""
    if (
        not all_fit_data
        or not isinstance(all_fit_data, tuple)
        or len(all_fit_data) != 2
    ):
        return None

    test_data_df, ref_data_df = all_fit_data
    if test_data_df.empty or ref_data_df.empty:
        return None

    required_cols = ["timestamp", "filename", metric]

    # Filter to required columns
    test_data_df = test_data_df[required_cols].copy()
    ref_data_df = ref_data_df[required_cols].copy()

    # Ensure filename is string type
    test_data_df["filename"] = test_data_df["filename"].astype(str)
    ref_data_df["filename"] = ref_data_df["filename"].astype(str)

    # Find common timestamps between test and reference data
    test_timestamps = set(test_data_df["timestamp"])
    ref_timestamps = set(ref_data_df["timestamp"])
    common_timestamps = test_timestamps.intersection(ref_timestamps)

    if not common_timestamps:
        return None

    # Filter to only common timestamps
    test_data_df = test_data_df[
        test_data_df["timestamp"].isin(common_timestamps)
    ].copy()
    ref_data_df = ref_data_df[ref_data_df["timestamp"].isin(common_timestamps)].copy()

    # Generate elapsed_seconds on a per-file basis
    # Each file starts at elapsed_seconds = 0 from its own first timestamp (after filtering)
    for df in [test_data_df, ref_data_df]:
        df["elapsed_seconds"] = df.groupby("filename")["timestamp"].transform(
            lambda x: (x - x.min()).dt.total_seconds()
        )

    return test_data_df, ref_data_df


def get_aligned_data(
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Get aligned test and reference data by timestamp."""
    if test_data is None or test_data.empty or ref_data is None or ref_data.empty:
        return None

    if metric not in test_data.columns or metric not in ref_data.columns:
        return None

    # Get clean data
    test_clean = test_data[["timestamp", "elapsed_seconds", metric]].dropna()
    ref_clean = ref_data[["timestamp", "elapsed_seconds", metric]].dropna()

    # Merge on timestamp to align the data properly
    aligned_df = pd.merge(
        test_clean, ref_clean, on="timestamp", suffixes=("_test", "_ref")
    )

    return aligned_df if not aligned_df.empty else None


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


def apply_outlier_removal(
    test_data: pd.DataFrame,
    ref_data: pd.DataFrame,
    metric: str,
    outlier_removal_methods: list = None,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], None]:
    """Apply outlier removal to test and reference data."""
    if not outlier_removal_methods:
        return test_data, ref_data

    # Apply outlier removal to each dataset separately
    if metric in test_data.columns:
        test_data = remove_outliers(test_data, metric, outlier_removal_methods)
    if metric in ref_data.columns:
        ref_data = remove_outliers(ref_data, metric, outlier_removal_methods)

    if test_data.empty or ref_data.empty:
        return None

    return test_data, ref_data


def create_metric_plot(
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str
) -> Union[go.Figure, None]:
    """Create line plot from test and reference data for specified metric."""
    if test_data is None or test_data.empty or ref_data is None or ref_data.empty:
        return None

    fig = go.Figure()

    # Add only mean trend lines (no raw data points)
    for source_name, source_data in [("test", test_data), ("reference", ref_data)]:
        if not source_data.empty and metric in source_data.columns:
            # Calculate mean per second for trend line
            mean_per_second = (
                source_data.groupby(source_data["elapsed_seconds"].round())[metric]
                .mean()
                .reset_index()
            )

            if not mean_per_second.empty:
                fig.add_trace(
                    go.Scatter(
                        x=mean_per_second["elapsed_seconds"],
                        y=mean_per_second[metric],
                        mode="lines",
                        name=f"{source_name}",
                        line=dict(width=2),
                    )
                )

    fig.update_layout(
        # title=f"{metric} over Time",
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title=metric,
        hovermode="x unified",
    )
    return fig


def calculate_basic_stats(
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Calculate basic statistics for test and reference data using aligned data."""
    # Get aligned data to ensure equal counts
    aligned_df = get_aligned_data(test_data, ref_data, metric)
    if aligned_df is None:
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
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Get bias and agreement statistics."""
    aligned_df = get_aligned_data(test_data, ref_data, metric)
    if aligned_df is None:
        return None

    test_aligned = aligned_df[f"{metric}_test"]
    ref_aligned = aligned_df[f"{metric}_ref"]
    errors = test_aligned - ref_aligned
    n_points = len(aligned_df)

    # Calculate bias and agreement metrics
    bias = errors.mean()
    std_errors = errors.std()
    loa_upper = bias + 1.96 * std_errors
    loa_lower = bias - 1.96 * std_errors

    # Statistical tests for bias
    t_stat, p_value = stats.ttest_1samp(errors, 0)
    cohens_d = bias / std_errors if std_errors > 0 else np.nan

    # 95% Confidence interval for bias
    confidence_interval = stats.t.interval(
        0.95, n_points - 1, loc=bias, scale=stats.sem(errors)
    )

    bias_agreement_stats = {
        "Mean Bias": round(bias, 6),
        "95% CI Lower": round(confidence_interval[0], 6),
        "95% CI Upper": round(confidence_interval[1], 6),
        "T-statistic": round(t_stat, 6),
        "P-value": round(p_value, 8),
        "Cohen's d": round(cohens_d, 6) if not np.isnan(cohens_d) else "N/A",
        "LoA Upper": round(loa_upper, 6),
        "LoA Lower": round(loa_lower, 6),
    }

    # Convert to transposed DataFrame
    df = pd.DataFrame(list(bias_agreement_stats.items()), columns=["Metric", "Value"])
    return df


def get_error_magnitude_stats(
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Get error magnitude statistics."""
    aligned_df = get_aligned_data(test_data, ref_data, metric)
    if aligned_df is None:
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
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str
) -> Union[pd.DataFrame, None]:
    """Get correlation statistics."""
    aligned_df = get_aligned_data(test_data, ref_data, metric)
    if aligned_df is None:
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


def create_error_histogram(
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str
) -> Union[go.Figure, None]:
    """Create error distribution histogram from test and reference data."""
    aligned_df = get_aligned_data(test_data, ref_data, metric)
    if aligned_df is None:
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
        # title=f"Error Distribution for {metric} (Test vs Reference)",
        xaxis_title="Error (test - reference)",
        yaxis_title="Frequency",
        barmode="overlay",
    )

    return fig


def create_bland_altman_plot(
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str
) -> Union[go.Figure, None]:
    """Create Bland-Altman plot from test and reference data."""
    aligned_df = get_aligned_data(test_data, ref_data, metric)
    if aligned_df is None:
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
        # title=f"Bland-Altman Plot for {metric} (Test vs Reference)",
        xaxis_title=f"Mean of Test and Reference {metric}",
        yaxis_title=f"Difference (Test - Reference) {metric}",
        showlegend=False,
    )

    return fig


def create_rolling_error_plot(
    test_data: pd.DataFrame,
    ref_data: pd.DataFrame,
    metric: str,
    window_size: int = 50,
) -> Union[go.Figure, None]:
    """Create rolling error / time-varying bias plot from test and reference data."""
    aligned_df = get_aligned_data(test_data, ref_data, metric)
    if aligned_df is None:
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
        # title=f"Rolling Error / Time-Varying Bias for {metric} (Test vs Reference)",
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title=f"Rolling Error in {metric} (window={window_size})",
        showlegend=True,
        hovermode="x unified",
    )

    return fig


def get_file_information(
    test_data: pd.DataFrame, ref_data: pd.DataFrame
) -> Union[pd.DataFrame, None]:
    """Get file information for test and reference data showing raw data before filtering."""
    if test_data is None or test_data.empty or ref_data is None or ref_data.empty:
        return None

    def extract_file_info(df, device_type):
        info_list = []
        for filename in df["filename"].unique():
            file_subset = df[df["filename"] == filename]
            if not file_subset.empty and "timestamp" in file_subset.columns:
                all_metrics = [
                    col
                    for col in file_subset.columns
                    if col not in ["timestamp", "filename", "elapsed_seconds"]
                    and file_subset[col].notna().any()
                ]
                file_info = {
                    "filename": str(filename),
                    "device_type": device_type,
                    "records": len(file_subset),
                    "start_time": file_subset["timestamp"].min(),
                    "end_time": file_subset["timestamp"].max(),
                    "duration_minutes": round(
                        (
                            file_subset["timestamp"].max()
                            - file_subset["timestamp"].min()
                        ).total_seconds()
                        / 60,
                        1,
                    ),
                    "sampling_rate_hz": None,
                    "available_metrics": ", ".join(sorted(all_metrics)),
                    "metric_count": len(all_metrics),
                }
                duration_sec = (
                    file_subset["timestamp"].max() - file_subset["timestamp"].min()
                ).total_seconds()
                if duration_sec > 0:
                    file_info["sampling_rate_hz"] = round(
                        len(file_subset) / duration_sec, 2
                    )
                info_list.append(file_info)
        return info_list

    file_info_list = extract_file_info(test_data, "test") + extract_file_info(
        ref_data, "reference"
    )
    if not file_info_list:
        return None
    return pd.DataFrame(file_info_list)


def get_raw_data_sample(
    test_data: pd.DataFrame, ref_data: pd.DataFrame, metric: str, sample_size: int = 100
) -> Union[pd.DataFrame, None]:
    """Get a sample of raw aligned data for inspection."""
    aligned_df = get_aligned_data(test_data, ref_data, metric)
    if aligned_df is None:
        return None

    # Sort by timestamp and take a sample
    aligned_df = aligned_df.sort_values("timestamp")

    # Take evenly spaced samples if data is large
    if len(aligned_df) > sample_size:
        step = len(aligned_df) // sample_size
        sample_df = aligned_df.iloc[::step][:sample_size]
    else:
        sample_df = aligned_df

    # Reorder columns for better readability
    columns_order = [
        "timestamp",
        "elapsed_seconds_test",
        "elapsed_seconds_ref",
        f"{metric}_test",
        f"{metric}_ref",
    ]

    # Only include columns that exist
    available_columns = [col for col in columns_order if col in sample_df.columns]
    sample_df = sample_df[available_columns].copy()

    # Add difference column
    sample_df["difference"] = sample_df[f"{metric}_test"] - sample_df[f"{metric}_ref"]

    # Round numeric values for better display
    numeric_columns = sample_df.select_dtypes(include=[np.number]).columns
    sample_df[numeric_columns] = sample_df[numeric_columns].round(3)

    return sample_df
