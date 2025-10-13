"""
Data processing utilities for FIT and CSV files.
"""

import logging
import math
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream

from .statistics import calculate_ccc

logger = logging.getLogger(__name__)

# Optional S3 support
try:
    import os
    import tempfile

    import boto3

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


def process_file(file_path: str, aws_profile: str = None) -> dict:
    """
    Read a FIT or CSV file from local filesystem or S3 and return dictionary with:
    - session_df: one‐row DataFrame of all session messages
    - record_df: one‐row per 'record' message (timestamped samples)
    - file_id_df: one‐row of file ID messages
    - device_info_df: one‐row of device info messages

    Args:
        file_path: Local file path or S3 URL (s3://bucket/key)
        aws_profile: AWS profile name for S3 access (optional)
    """
    # Check if it's an S3 URL
    if file_path.startswith("s3://"):
        return _process_s3_file(file_path, aws_profile)

    # Handle local file
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".csv":
        return _process_csv(file_path)
    elif file_path.suffix.lower() == ".fit":
        return _process_fit_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _read_fit_file(file_path: Path) -> dict:
    """Read and decode a FIT file, returning raw messages."""
    stream = Stream.from_file(str(file_path))
    decoder = Decoder(stream)
    messages, _ = decoder.read(
        apply_scale_and_offset=True,
        convert_datetimes_to_dates=False,  # Get raw FIT epoch timestamps
        convert_types_to_strings=True,
        enable_crc_check=True,
        expand_sub_fields=True,
        expand_components=True,
        merge_heart_rates=True,
        mesg_listener=None,
    )
    return messages


def _process_fit_messages(messages: dict, filename: str) -> dict:
    """Process FIT messages into session and record DataFrames."""
    ### Record data - one row per record message ###
    record_df = pd.json_normalize(messages.get("record_mesgs", []), sep="_")
    if record_df.empty:
        raise ValueError("No record messages found in FIT file")

    # Convert GPS coordinates from semicircles to degrees
    if "position_lat" in record_df.columns and "position_long" in record_df.columns:
        record_df["position_lat"] = record_df["position_lat"] * (180 / 2**31)
        record_df["position_long"] = record_df["position_long"] * (180 / 2**31)

    # Convert timestamp to epoch (Unix timestamp)
    if "timestamp" in record_df.columns:
        # FIT epochs need to be converted to Unix epochs
        # FIT_EPOCH_S = 631065600 (seconds between Unix Epoch and FIT Epoch)
        FIT_EPOCH_S = 631065600
        record_df["timestamp"] = record_df["timestamp"] + FIT_EPOCH_S

    record_df["filename"] = filename

    ### Session data - one row per file ###
    session_df = pd.json_normalize(messages.get("session_mesgs", []), sep="_")
    if session_df.empty:
        raise ValueError("No session messages found in FIT file")
    session_df["filename"] = filename

    ### Metadata - one row per file ###
    file_id_df = pd.json_normalize(messages.get("file_id_mesgs", []), sep="_")
    device_info_df = pd.json_normalize(messages.get("device_info_mesgs", []), sep="_")

    if file_id_df.empty and device_info_df.empty:
        raise ValueError("No file_id_mesgs or device_info_mesgs found in FIT file")
    file_id_df["filename"] = filename
    device_info_df["filename"] = filename

    return {
        "session": session_df,
        "records": record_df,
        "file_id": file_id_df,
        "device_info": device_info_df,
    }


def _process_fit_file(file_path: Path) -> dict:
    """Process a FIT file by reading and then processing the messages."""
    messages = _read_fit_file(file_path)
    return _process_fit_messages(messages, str(file_path.name))


def _process_csv(file_path: Path) -> dict:
    """Process a CSV file, converting it to the same format as FIT files."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    if df.empty:
        raise ValueError("CSV file is empty")

    # Add filename to the dataframe
    df["filename"] = str(file_path.name)

    # Try to identify timestamp column
    timestamp_cols = [
        col
        for col in df.columns
        if col.lower() in ["timestamp", "datetime", "time_stamp", "date_time", "time"]
    ]

    if not timestamp_cols:
        raise ValueError("No timestamp column found")

    # Use the first timestamp column found
    timestamp_col = timestamp_cols[0]

    # Convert timestamp column to epoch (Unix timestamp)
    try:
        # First, try to parse as datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")

        # Check if timezone info is present or if we need to assume GMT+8
        if df[timestamp_col].dt.tz is None:
            # No timezone info: assume GMT+8 and convert to UTC
            df[timestamp_col] = (
                df[timestamp_col].dt.tz_localize("Asia/Singapore").dt.tz_convert("UTC")
            )
        else:
            # Timezone info present - convert to UTC
            df[timestamp_col] = df[timestamp_col].dt.tz_convert("UTC")

        # Convert to epoch (Unix timestamp in seconds)
        df[timestamp_col] = df[timestamp_col].astype("int64") // 10**9

    except Exception as e:
        raise ValueError(f"Error parsing timestamp column '{timestamp_col}': {e}")

    # Rename timestamp column to standard name
    if timestamp_col != "timestamp":
        df = df.rename(columns={timestamp_col: "timestamp"})

    # Create record_df (the main data)
    record_df = df.copy()

    # Create session_df (summary data - one row per file)
    session_data = {
        "filename": str(file_path.name),
        "start_time": df["timestamp"].min(),  # Min epoch timestamp
        "end_time": df["timestamp"].max(),  # Max epoch timestamp
        "total_records": len(df),
        "duration_seconds": df["timestamp"].max()
        - df["timestamp"].min(),  # Epoch difference
    }

    # Add summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "timestamp":  # Skip timestamp if it's numeric
            session_data[f"{col}_avg"] = df[col].mean()
            session_data[f"{col}_max"] = df[col].max()
            session_data[f"{col}_min"] = df[col].min()

    session_df = pd.DataFrame([session_data])

    return {
        "session": session_df,
        "records": record_df,
        "file_id": pd.DataFrame(),
        "device_info": pd.DataFrame(),
    }


def _read_s3_fit_file(s3_url: str, aws_profile: str = None) -> dict:
    """Read and decode a FIT file from S3, returning raw messages."""
    if not S3_AVAILABLE:
        raise ImportError(
            "boto3 is required for S3 support. Install with: pip install boto3"
        )

    # Parse S3 URL
    if not s3_url.startswith("s3://"):
        raise ValueError("S3 URL must start with s3://")

    url_parts = s3_url[5:].split("/", 1)  # Remove s3:// prefix
    if len(url_parts) != 2:
        raise ValueError("Invalid S3 URL format. Expected: s3://bucket/key")

    bucket_name, key = url_parts

    # Create S3 client with optional profile
    session = (
        boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    )
    s3_client = session.client("s3")

    # Download to temporary file for FIT processing (garmin_fit_sdk needs file path)
    with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tmp_file:
        try:
            logger.info(f"Downloading {s3_url} to temporary file...")
            s3_client.download_file(bucket_name, key, tmp_file.name)

            # Read FIT messages from temporary file
            messages = _read_fit_file(Path(tmp_file.name))
            return messages

        except Exception as e:
            logger.error(f"Error reading S3 FIT file {s3_url}: {e}")
            raise
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file.name)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")


def _process_s3_file(s3_url: str, aws_profile: str = None) -> dict:
    """
    Process a FIT or CSV file from S3.

    Args:
        s3_url: S3 URL in format s3://bucket/key
        aws_profile: AWS profile name (optional)

    Returns:
        Dictionary with keys:
          - 'session': session DataFrame
          - 'records': record DataFrame
          - 'file_id': file ID DataFrame
          - 'device_info': device info DataFrame
    """
    if not S3_AVAILABLE:
        raise ImportError(
            "boto3 is required for S3 support. Install with: pip install boto3"
        )

    # Parse S3 URL to get key and determine file type
    url_parts = s3_url[5:].split("/", 1)  # Remove s3:// prefix
    if len(url_parts) != 2:
        raise ValueError("Invalid S3 URL format. Expected: s3://bucket/key")

    bucket_name, key = url_parts
    file_extension = Path(key).suffix.lower()
    original_filename = Path(key).name

    if file_extension not in [".fit", ".csv"]:
        raise ValueError(f"Unsupported file format: {file_extension}")

    try:
        if file_extension == ".fit":
            # Process FIT file
            messages = _read_s3_fit_file(s3_url, aws_profile)
            df_dict = _process_fit_messages(messages, original_filename)
        else:  # .csv
            # For CSV files, still need to download to temp file as pandas needs file path
            session = (
                boto3.Session(profile_name=aws_profile)
                if aws_profile
                else boto3.Session()
            )
            s3_client = session.client("s3")

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                try:
                    logger.info(f"Downloading {s3_url} to temporary file...")
                    s3_client.download_file(bucket_name, key, tmp_file.name)
                    df_dict = _process_csv(Path(tmp_file.name))

                    # Update filename to use the original S3 key basename
                    df_dict["session"]["filename"] = original_filename
                    df_dict["records"]["filename"] = original_filename

                finally:
                    try:
                        os.unlink(tmp_file.name)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Failed to cleanup temporary file: {cleanup_error}"
                        )

        return df_dict

    except Exception as e:
        logger.error(f"Error processing S3 file {s3_url}: {e}")
        raise


def _get_required_columns(df: pd.DataFrame, metric: str) -> list:
    """Get list of required columns for analysis, including supplementary columns."""
    required_cols = ["timestamp", "filename", metric]

    # Supplementary cadence column needed by downstream filters (e.g. HR ≈ 2 × cadence)
    supplementary = []
    if "cadence" in df.columns and metric == "heart_rate":
        supplementary.append("cadence")

    # Build final column list guarding for missing columns
    cols = []
    for c in required_cols + supplementary:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def _prepare_single_dataframe(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Prepare a single DataFrame for analysis."""
    data_df = df.copy()

    # Select required columns
    required_cols = _get_required_columns(data_df, metric)
    data_df = data_df[required_cols].copy()

    # Ensure filename is string type
    data_df["filename"] = data_df["filename"].astype(str)

    # Add elapsed seconds based on first timestamp per file
    data_df["elapsed_seconds"] = data_df.groupby("filename")["timestamp"].transform(
        lambda x: x - x.min()
    )

    return data_df


def _prepare_comparison_dataframes(
    test_df: pd.DataFrame, ref_df: pd.DataFrame, metric: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare two DataFrames for comparison analysis with common timestamp alignment."""
    if test_df.empty or ref_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Select required columns for both dataframes
    test_cols = _get_required_columns(test_df, metric)
    ref_cols = _get_required_columns(ref_df, metric)

    test_data_df = test_df[test_cols].copy()
    ref_data_df = ref_df[ref_cols].copy()

    # Ensure filename is string type
    test_data_df["filename"] = test_data_df["filename"].astype(str)
    ref_data_df["filename"] = ref_data_df["filename"].astype(str)

    # Find common timestamps between test and reference data
    test_timestamps = set(test_data_df["timestamp"])
    ref_timestamps = set(ref_data_df["timestamp"])
    common_timestamps = test_timestamps.intersection(ref_timestamps)

    if not common_timestamps:
        return pd.DataFrame(), pd.DataFrame()

    # Create file pairings based on overlapping timestamps
    # Find which test and reference files have overlapping timestamps
    test_file_timestamps = {}
    ref_file_timestamps = {}

    for filename in test_data_df["filename"].unique():
        file_timestamps = set(
            test_data_df[test_data_df["filename"] == filename]["timestamp"]
        )
        test_file_timestamps[filename] = file_timestamps

    for filename in ref_data_df["filename"].unique():
        file_timestamps = set(
            ref_data_df[ref_data_df["filename"] == filename]["timestamp"]
        )
        ref_file_timestamps[filename] = file_timestamps

    # Create pairs based on overlapping timestamps between test and ref files
    file_pairs = []
    pair_index = 0

    for test_file, test_ts in test_file_timestamps.items():
        for ref_file, ref_ts in ref_file_timestamps.items():
            if test_ts.intersection(ref_ts):  # Files have overlapping timestamps
                file_pairs.append((test_file, ref_file, pair_index))
                pair_index += 1
                break  # Assume one-to-one pairing; move to next test file

    # Create mapping from filename to pair_index
    test_file_to_pair = {}
    ref_file_to_pair = {}

    for test_file, ref_file, pair_idx in file_pairs:
        test_file_to_pair[test_file] = pair_idx
        ref_file_to_pair[ref_file] = pair_idx

    # Add pair_index and elapsed_seconds to both dataframes
    for df, file_to_pair in [
        (test_data_df, test_file_to_pair),
        (ref_data_df, ref_file_to_pair),
    ]:
        # Map filename to pair_index
        df["pair_index"] = df["filename"].map(file_to_pair).astype("Int64")

        # Elapsed seconds based on first common timestamp per file
        df["elapsed_seconds"] = df.groupby("filename")["timestamp"].transform(
            lambda x: x - x[x.isin(common_timestamps)].min()
        )

    return test_data_df, ref_data_df


def process_multiple_files(file_paths: list, aws_profile: str = None) -> list:
    """
    Process multiple files (local or S3) and return list of (session_df, record_df) tuples.

    Args:
        file_paths: List of file paths or S3 URLs
        aws_profile: AWS profile name for S3 files (optional)

    Returns:
        Dictionary with keys:
          - 'session': session DataFrame
          - 'records': record DataFrame
          - 'file_id': file ID DataFrame
          - 'device_info': device info DataFrame
        for each successfully processed file.
    """
    results = []
    failed_files = []

    for i, file_path in enumerate(file_paths, 1):
        try:
            logger.info(f"Processing file {i}/{len(file_paths)}: {file_path}")
            df_dict = process_file(file_path, aws_profile)
            results.append(df_dict)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            failed_files.append(file_path)
            continue

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")

    logger.info(f"Successfully processed {len(results)}/{len(file_paths)} files")
    return results


def prepare_data_for_analysis(
    all_fit_data: Union[tuple, pd.DataFrame], metric: str
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame, None]:
    """
    Prepare data for analysis.

    Args:
        all_fit_data: Either a tuple of (test_data_df, ref_data_df) for comparison analysis,
                     or a single DataFrame for single-file analysis
        metric: The metric column name to analyze

    Returns:
        - For tuple input: (test_data_df, ref_data_df) with common timestamps and elapsed_seconds
        - For DataFrame input: Single processed DataFrame with elapsed_seconds
        - None if invalid input
    """
    # Handle single DataFrame input
    if isinstance(all_fit_data, pd.DataFrame):
        if all_fit_data.empty:
            return None
        return _prepare_single_dataframe(all_fit_data, metric)

    # Handle tuple input for comparison analysis
    if not isinstance(all_fit_data, tuple) or len(all_fit_data) != 2:
        return None

    test_data_df, ref_data_df = all_fit_data

    return _prepare_comparison_dataframes(test_data_df, ref_data_df, metric)


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
                        )
                        / 60,
                        1,
                    ),
                    "sampling_rate_hz": None,
                    "available_metrics": ", ".join(sorted(all_metrics)),
                    "metric_count": len(all_metrics),
                }
                duration_sec = (
                    file_subset["timestamp"].max() - file_subset["timestamp"].min()
                )
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


def determine_optimal_shift(test_data, ref_data, metric, auto_shift_method):
    """Determine optimal time shift in seconds to align test data to reference data."""
    if test_data is None or test_data.empty or ref_data is None or ref_data.empty:
        return None
    if "None" in auto_shift_method:
        return 0

    # Check if required columns exist
    if metric not in test_data.columns or metric not in ref_data.columns:
        return 0
    if "timestamp" not in test_data.columns or "timestamp" not in ref_data.columns:
        return 0

    # Set max shift to 300 seconds (5 minutes)
    max_shift_seconds = 300

    # Default step size is 1 second
    step_size = 1

    ### Determine adaptive step size based on data sampling intervals ###
    test_dt = test_data["timestamp"].diff().dropna().astype(int)
    ref_dt = ref_data["timestamp"].diff().dropna().astype(int)

    # keep only positive gaps
    test_dt = test_dt[test_dt > 0]
    ref_dt = ref_dt[ref_dt > 0]

    if not test_dt.empty and not ref_dt.empty:
        # Base step per stream: gcd of all intervals
        def base_step(diffs: np.ndarray) -> int:
            g = 0
            for v in diffs:
                g = math.gcd(g, int(v))
                if g == 1:  # early exit; can't get smaller than 1
                    break
            return max(g, 1)

        test_base = base_step(test_dt.values)
        ref_base = base_step(ref_dt.values)

        # Validate that every interval is a multiple of the base (guards weirdness)
        if np.all((test_dt.values % test_base) == 0) and np.all(
            (ref_dt.values % ref_base) == 0
        ):
            step_size = math.gcd(test_base, ref_base)
            logger.info("Adaptive step size set at %s seconds", step_size)

    best_shift = 0
    best_score = None

    # Function to calculate metric score for a given shift
    def calculate_score(shift_seconds):
        # Create shifted test data
        test_shifted = test_data.copy()
        test_shifted["timestamp"] = test_shifted["timestamp"] + shift_seconds

        # Merge on timestamp to align data
        merged = pd.merge(
            test_shifted[["timestamp", metric]],
            ref_data[["timestamp", metric]],
            on="timestamp",
            suffixes=("_test", "_ref"),
        )

        if merged.empty:
            return None

        # Clean data - remove NaN values
        merged_clean = merged.dropna(subset=[f"{metric}_test", f"{metric}_ref"])
        if merged_clean.empty:
            return None

        test_values = merged_clean[f"{metric}_test"]
        ref_values = merged_clean[f"{metric}_ref"]

        if auto_shift_method == "Minimize MAE":
            # Mean Absolute Error (lower is better)
            return np.mean(np.abs(test_values - ref_values))
        elif auto_shift_method == "Minimize MSE":
            # Mean Squared Error (lower is better)
            return np.mean((test_values - ref_values) ** 2)
        elif auto_shift_method == "Maximize Concordance Correlation":
            return -calculate_ccc(test_values, ref_values)
        elif auto_shift_method == "Maximize Pearson Correlation":
            # Higher is better, return negative for consistency
            if len(test_values) < 2:
                return None
            corr = np.corrcoef(test_values, ref_values)[0, 1]
            return -corr if not np.isnan(corr) else None

    # Start from 0 and expand outwards using adaptive step size
    shifts_to_try = [0]  # Start with no shift
    current_step = step_size
    while current_step <= max_shift_seconds:
        shifts_to_try.extend(
            [current_step, -current_step]
        )  # Add positive and negative shifts
        current_step += step_size

    # Find the best shift
    for shift in shifts_to_try:
        score = calculate_score(shift)

        if score is not None:
            if (
                best_score is None or score < best_score
            ):  # Lower is better for all our metrics
                best_score = score
                best_shift = shift

    return best_shift
