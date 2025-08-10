"""
Supporting utils and functions
"""

import json
import os
from pathlib import Path
from typing import Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from garmin_fit_sdk import Decoder, Stream
from scipy import stats

load_dotenv(override=True)
API_KEY_ID = os.getenv("API_KEY_ID", "")
API_KEY_SECRET = os.getenv("API_KEY_SECRET", "")


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
        if method == "remove_iqr":
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


def create_metric_plot(aligned_df: pd.DataFrame, metric: str) -> Union[go.Figure, None]:
    """Create line plot from aligned test and reference data for specified metric."""
    if aligned_df is None or aligned_df.empty:
        return None

    fig = go.Figure()

    # Add trend lines for both test and reference data
    for source_name, col_suffix in [("test", "_test"), ("reference", "_ref")]:
        metric_col = f"{metric}{col_suffix}"
        elapsed_col = f"elapsed_seconds{col_suffix}"

        if metric_col in aligned_df.columns and elapsed_col in aligned_df.columns:
            # Calculate mean per second for trend line
            temp_df = aligned_df[[elapsed_col, metric_col]].dropna()
            if not temp_df.empty:
                mean_per_second = (
                    temp_df.groupby(temp_df[elapsed_col].round())[metric_col]
                    .mean()
                    .reset_index()
                )

                if not mean_per_second.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=mean_per_second[elapsed_col],
                            y=mean_per_second[metric_col],
                            mode="lines",
                            name=f"{source_name}",
                            line=dict(width=2),
                        )
                    )

    fig.update_layout(
        xaxis_title="Elapsed Time (seconds)",
        yaxis_title=metric,
        hovermode="x unified",
    )
    return fig


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
    # skewness = stats.skew(errors)
    # kurt = stats.kurtosis(errors, fisher=True)  # Fisher’s definition

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

    # # 95% CI for bias
    # ci_low, ci_high = stats.t.interval(
    #     0.95, df=n_points - 1, loc=bias, scale=stats.sem(errors)
    # )

    # # Limits of Agreement
    # loa_upper = bias + 1.96 * std_err
    # loa_lower = bias - 1.96 * std_err

    # Effect size
    cohens_d = bias / std_err if std_err > 0 else np.nan

    # Assemble results
    rows = [
        # ("Count", n_points),
        ("Mean Bias", round(bias, 6)),
        # ("Skewness", round(skewness, 6)),
        # ("Kurtosis", round(kurt, 6)),
        ("Shapiro–Wilk p-value", round(sw_p, 6)),
        (f"{test_name} statistic", round(t_stat, 6)),
        (f"{test_name} p-value", round(p_val, 8)),
        # ("95% CI Lower", round(ci_low, 6)),
        # ("95% CI Upper", round(ci_high, 6)),
        # ("LoA Upper", round(loa_upper, 6)),
        # ("LoA Lower", round(loa_lower, 6)),
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
        # title=f"Error Distribution for {metric} (Test vs Reference)",
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
        # title=f"Bland-Altman Plot for {metric} (Test vs Reference)",
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
    test_data: pd.DataFrame,
    ref_data: pd.DataFrame,
    sample_size: int = 100,
    selected_filenames: list = None,
) -> Union[pd.DataFrame, None]:
    """Get a sample of raw data for inspection, optionally filtered by selected files."""
    # Combine both test and reference data
    combined_data = pd.concat([test_data, ref_data], ignore_index=True)

    # Filter data by selected filenames if provided
    if selected_filenames:
        combined_data = combined_data[
            combined_data["filename"].isin(selected_filenames)
        ]

    # If no data remains after filtering, return empty DataFrame
    if combined_data.empty:
        return pd.DataFrame()

    # Sort by timestamp and take a sample
    combined_data = combined_data.sort_values("timestamp")

    # Take evenly spaced samples if data is large
    if len(combined_data) > sample_size:
        step = len(combined_data) // sample_size
        sample_df = combined_data.iloc[::step][:sample_size]
    else:
        sample_df = combined_data.copy()

    # Remove empty columns more thoroughly
    columns_to_drop = []

    for col in sample_df.columns:
        if sample_df[col].dtype == "object":
            # For object columns, check if all values are empty/whitespace after dropping NaN
            non_na_values = sample_df[col].dropna()
            if non_na_values.empty:
                columns_to_drop.append(col)
                continue
            # Convert to string and check for empty/whitespace
            str_values = non_na_values.astype(str).str.strip()
            if str_values.empty or (str_values == "").all():
                columns_to_drop.append(col)
        else:
            # For numeric columns, check if all values are NaN
            if sample_df[col].isna().all():
                columns_to_drop.append(col)

    # Drop all identified empty columns at once
    sample_df = sample_df.drop(columns=columns_to_drop)

    # Round numeric values for better display
    numeric_columns = sample_df.select_dtypes(include=[np.number]).columns
    sample_df[numeric_columns] = sample_df[numeric_columns].round(3)

    return sample_df


async def generate_llm_summary(
    metric: str,
    bias_stats: pd.DataFrame,
    error_stats: pd.DataFrame,
    correlation_stats: pd.DataFrame,
) -> str:
    """
    Generate a summary for the LLM based on the provided statistics.
    """
    # Only proceed if all three stats are present and non-empty
    if (
        bias_stats is not None
        and not bias_stats.empty
        and error_stats is not None
        and not error_stats.empty
        and correlation_stats is not None
        and not correlation_stats.empty
    ):
        records = {
            "benchmark_metric": metric,
            "bias_agreement": bias_stats.to_dict(orient="records"),
            "error_magnitude": error_stats.to_dict(orient="records"),
            "correlation": correlation_stats.to_dict(orient="records"),
        }
        response = await api_call_to_llm(records)
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        return "Insufficient statistics: all of bias, error, and correlation stats must be present and non-empty."


async def api_call_to_llm(records: dict) -> dict:
    """
    Make an async API call to the LLM with the provided records.
    """
    url = "https://llm-hrpc.paperclips.dev/v1/chat/completions"
    headers = {
        "CF-Access-Client-Id": API_KEY_ID,
        "CF-Access-Client-Secret": API_KEY_SECRET,
        "Content-Type": "application/json",
    }
    prompt = f"""
    Reason logically. Interpret the following JSON benchmark stats for non-technical readers, and explain in layman terms.
    - Context: benchmarking of wearable devices regarding {records.get("benchmark_metric", "")}.
    - Mainly focus on the key statistics that materially influence your verdict.
    - Preserve all numerical values exactly as given.
    - Always caveat any speculations not supported by the data.
    - Always caveat any generic disclaimers.
    - Output is meant for dashboard use.
    - End with a a single-sentence verdict.
    - Output valid HTML only, using <p>, <ul>, <li>, <strong>, and <em> tags for structure.
    - No <html>, <head>, or <body> wrappers — only the HTML snippet for embedding.

    JSON:
    {json.dumps(records, indent=2)}
    """
    data = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=data, timeout=30
            ) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        print(f"Error occurred: {e}")
        return {"error": str(e)}
