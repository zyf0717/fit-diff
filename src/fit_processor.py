"""
FIT file processor module.
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fitparse import FitFile


class FitProcessor:
    """Processes FIT files and extracts data into pandas DataFrames."""

    def __init__(self):
        self.supported_messages = ["record", "session", "lap", "device_info", "file_id"]

    def process_fit_file(self, file_path: str) -> pd.DataFrame:
        """
        Process a FIT file and return a DataFrame with key metrics.

        Args:
            file_path: Path to the FIT file

        Returns:
            DataFrame with processed FIT data
        """
        fitfile = FitFile(file_path)

        records = []

        for message in fitfile.get_messages():
            if message.name in self.supported_messages:
                record = {"message_type": message.name}

                for field in message.fields:
                    if field.value is not None:
                        record[field.name] = field.value

                records.append(record)

        if not records:
            raise ValueError("No supported data found in FIT file")

        df = pd.DataFrame(records)

        # Basic processing
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def extract_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract summary statistics from processed FIT data."""
        stats = {}

        # Record counts by message type
        if "message_type" in df.columns:
            stats["message_counts"] = df["message_type"].value_counts().to_dict()

        # Time range
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].dropna()
            if not timestamps.empty:
                stats["start_time"] = timestamps.min()
                stats["end_time"] = timestamps.max()
                stats["duration"] = timestamps.max() - timestamps.min()

        # Numeric field summaries
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if col in ["heart_rate", "speed", "power", "cadence", "distance"]:
                values = df[col].dropna()
                if not values.empty:
                    stats[f"{col}_mean"] = values.mean()
                    stats[f"{col}_max"] = values.max()
                    stats[f"{col}_min"] = values.min()

        return stats
        return stats
