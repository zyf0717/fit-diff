"""
Data processing utilities for FIT and CSV files.
"""

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream


def process_file(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a FIT or CSV file and return:
      - session_df: one‐row DataFrame of all session messages
      - record_df: one‐row per 'record' message (timestamped samples)
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        return _process_csv(file_path)
    elif file_path.suffix.lower() == ".fit":
        return _process_fit_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _process_fit_file(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process a FIT file."""
    stream = Stream.from_file(str(file_path))
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
    record_df["filename"] = str(file_path.name)

    session_df = pd.json_normalize(messages.get("session_mesgs", []), sep="_")
    if session_df.empty:
        raise ValueError("No session messages found in FIT file")
    session_df["filename"] = str(file_path.name)

    return session_df, record_df


def _process_csv(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        col for col in df.columns if "time" in col.lower() or "timestamp" in col.lower()
    ]
    if not timestamp_cols:
        # Look for common datetime patterns in column names
        datetime_cols = [
            col
            for col in df.columns
            if any(
                pattern in col.lower()
                for pattern in ["date", "time", "timestamp", "datetime"]
            )
        ]
        if datetime_cols:
            timestamp_cols = datetime_cols
        else:
            raise ValueError("No timestamp column found in CSV file")

    # Use the first timestamp column found
    timestamp_col = timestamp_cols[0]

    # Convert timestamp column to datetime
    try:
        # First, try to parse as datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], dayfirst=True)

        # Check if timezone info is present or if we need to assume GMT+8
        if df[timestamp_col].dt.tz is None:
            # No timezone info - assume GMT+8 and convert to UTC
            df[timestamp_col] = (
                df[timestamp_col].dt.tz_localize("Asia/Singapore").dt.tz_convert("UTC")
            )
        else:
            # Timezone info present - convert to UTC
            df[timestamp_col] = df[timestamp_col].dt.tz_convert("UTC")

        # Remove timezone info to keep as naive UTC datetime
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)

        # Convert to string format matching FIT files (ISO format with 'Z' suffix)
        timestamps_dt = df[timestamp_col].copy()  # Keep for calculations
        df[timestamp_col] = df[timestamp_col].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )  # ISO format, standardized with FIT files

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
        "start_time": df["timestamp"].min(),  # Keep as string for consistency with FIT
        "end_time": df["timestamp"].max(),  # Keep as string for consistency with FIT
        "total_records": len(df),
        "duration_seconds": (timestamps_dt.max() - timestamps_dt.min()).total_seconds(),
    }

    # Add summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "timestamp":  # Skip timestamp if it's numeric
            session_data[f"{col}_avg"] = df[col].mean()
            session_data[f"{col}_max"] = df[col].max()
            session_data[f"{col}_min"] = df[col].min()

    session_df = pd.DataFrame([session_data])

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
            lambda x: (pd.to_datetime(x) - pd.to_datetime(x.min())).dt.total_seconds()
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
    return sample_df
