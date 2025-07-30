"""
Diff analyzer module for comparing FIT file data.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class DiffAnalyzer:
    """Analyzes differences between multiple FIT file datasets."""

    def compare_files(
        self, dataframes: List[pd.DataFrame], filenames: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple FIT file DataFrames and return differences.

        Args:
            dataframes: List of processed FIT DataFrames
            filenames: List of corresponding filenames

        Returns:
            DataFrame summarizing differences between files
        """
        if len(dataframes) != len(filenames):
            raise ValueError("Number of dataframes must match number of filenames")

        comparison_results = []

        # Compare basic metrics
        for i, (df, filename) in enumerate(zip(dataframes, filenames)):
            stats = self._extract_basic_stats(df)
            stats["filename"] = filename
            stats["file_index"] = i
            comparison_results.append(stats)

        comparison_df = pd.DataFrame(comparison_results)

        # Add difference calculations
        diff_df = self._calculate_differences(comparison_df)

        return diff_df

    def _extract_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic statistics from a FIT DataFrame."""
        stats = {}

        # Record counts
        stats["total_records"] = len(df)

        if "message_type" in df.columns:
            message_counts = df["message_type"].value_counts()
            for msg_type, count in message_counts.items():
                stats[f"{msg_type}_records"] = count

        # Time-based metrics
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].dropna()
            if not timestamps.empty:
                stats["duration_seconds"] = (
                    timestamps.max() - timestamps.min()
                ).total_seconds()
                stats["start_time"] = timestamps.min()
                stats["end_time"] = timestamps.max()

        # Performance metrics
        numeric_metrics = [
            "heart_rate",
            "speed",
            "power",
            "cadence",
            "distance",
            "altitude",
        ]
        for metric in numeric_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if not values.empty:
                    stats[f"{metric}_avg"] = values.mean()
                    stats[f"{metric}_max"] = values.max()
                    stats[f"{metric}_min"] = values.min()

        return stats

    def _calculate_differences(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate differences between files."""
        diff_results = []

        # Identify numeric columns for comparison
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "file_index"]

        # Calculate differences between first file and others
        base_row = comparison_df.iloc[0] if len(comparison_df) > 0 else None

        for idx in range(len(comparison_df)):
            row = comparison_df.iloc[idx]
            diff_row = {"filename": row["filename"]}

            for col in numeric_cols:
                if col in comparison_df.columns:
                    if idx == 0 or len(comparison_df) == 1:
                        diff_row[f"{col}_diff"] = 0  # Base file or single file
                    else:
                        diff_value = (
                            row[col] - base_row[col]
                            if pd.notna(row[col]) and pd.notna(base_row[col])
                            else None
                        )
                        diff_row[f"{col}_diff"] = diff_value

            diff_results.append(diff_row)

        return pd.DataFrame(diff_results)
