import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.data_processing import process_file
from shared.utils.fit_pairing import (
    build_group_pairings,
    build_pair_candidate,
    build_time_bounds,
    format_duration_seconds,
)

logger = logging.getLogger(__name__)


def read_fit_record_timestamps(file_path):
    """Return record timestamps from a FIT file as Unix epoch seconds."""
    logger.info("Reading FIT file for overlap: %s", os.path.basename(str(file_path)))
    try:
        record_df = process_file(str(file_path))["records"]
    except Exception as exc:
        logger.warning(
            "Error reading FIT file %s for overlap: %s",
            os.path.basename(str(file_path)),
            exc,
        )
        return None

    if "timestamp" not in record_df.columns:
        return set()

    return set(record_df["timestamp"].dropna().astype("int64").tolist())


def compare_fit_files_by_timestamp(file_path_a, file_path_b):
    """Compare two FIT files and return overlap metrics based on common timestamps."""
    fit_data_a = _build_fit_overlap_data(file_path_a)
    fit_data_b = _build_fit_overlap_data(file_path_b)
    if fit_data_a is None or fit_data_b is None:
        return None

    candidate = build_pair_candidate(fit_data_a, fit_data_b, id_key="path")
    if candidate is None:
        return None
    return _candidate_to_overlap_metrics(candidate)


def build_folder_fit_overlap_metrics(file_paths):
    """
    Build overlap metrics for FIT files in the same folder.

    Files are paired within each folder using FIT time bounds, then overlap metrics
    are reported from the matched record timestamps when available.
    """
    fit_items = []
    for file_path in file_paths:
        if not str(file_path).lower().endswith(".fit"):
            continue
        fit_data = _build_fit_overlap_data(str(file_path))
        if fit_data is not None:
            fit_items.append(fit_data)

    overlap_by_path = {}
    for pairing in build_group_pairings(fit_items, group_key="folder", id_key="path"):
        metrics = _candidate_to_overlap_metrics(pairing)
        path_a = pairing["item_a_id"]
        path_b = pairing["item_b_id"]
        overlap_by_path[path_a] = {**metrics, "paired_file_path": path_b}
        overlap_by_path[path_b] = {**metrics, "paired_file_path": path_a}

    return overlap_by_path


def _build_fit_overlap_data(file_path):
    timestamps = read_fit_record_timestamps(file_path)
    if timestamps is None:
        return None

    return {
        "path": str(file_path),
        "folder": os.path.dirname(str(file_path)),
        "timestamps": timestamps,
        **build_time_bounds(timestamps),
    }


def _candidate_to_overlap_metrics(candidate):
    overlap_datapoints = candidate.get("overlap_datapoints") or 0
    overlap_duration_seconds = candidate.get("overlap_duration_seconds") or 0
    return {
        "overlap_duration": format_duration_seconds(overlap_duration_seconds),
        "overlap_datapoints": overlap_datapoints,
    }
