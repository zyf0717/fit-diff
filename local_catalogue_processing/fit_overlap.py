import logging
import os
from itertools import combinations

from garmin_fit_sdk import Decoder, Stream

logger = logging.getLogger(__name__)

FIT_EPOCH_S = 631065600


def read_fit_record_timestamps(file_path):
    """Return record timestamps from a FIT file as Unix epoch seconds."""
    logger.info("Reading FIT file for overlap: %s", os.path.basename(str(file_path)))
    try:
        stream = Stream.from_file(str(file_path))
        decoder = Decoder(stream)
        messages, _ = decoder.read(
            apply_scale_and_offset=True,
            convert_datetimes_to_dates=False,
            convert_types_to_strings=True,
            enable_crc_check=True,
            expand_sub_fields=True,
            expand_components=True,
            merge_heart_rates=True,
            mesg_listener=None,
        )
    except Exception as exc:
        logger.warning(
            "Error reading FIT file %s for overlap: %s",
            os.path.basename(str(file_path)),
            exc,
        )
        return None

    timestamps = set()
    for record in messages.get("record_mesgs", []):
        timestamp = record.get("timestamp")
        if timestamp is None:
            continue
        try:
            timestamps.add(int(timestamp) + FIT_EPOCH_S)
        except (TypeError, ValueError):
            logger.warning(
                "Skipping non-numeric timestamp in %s: %r",
                os.path.basename(str(file_path)),
                timestamp,
            )
    return timestamps


def compare_fit_files_by_timestamp(file_path_a, file_path_b):
    """Compare two FIT files and return overlap metrics based on common timestamps."""
    fit_data_a = _build_fit_overlap_data(file_path_a)
    fit_data_b = _build_fit_overlap_data(file_path_b)

    if fit_data_a is None or fit_data_b is None:
        return None

    return _build_overlap_metrics(fit_data_a, fit_data_b)


def build_folder_fit_overlap_metrics(file_paths):
    """
    Build overlap metrics for FIT files in the same folder.

    Files are paired within each folder using parsed FIT start/end timestamps.
    After pairing, overlap metrics are computed from the matching record timestamps.
    """
    fit_data_by_folder = {}
    for file_path in file_paths:
        if not str(file_path).lower().endswith(".fit"):
            continue
        folder_path = os.path.dirname(str(file_path))
        fit_data = _build_fit_overlap_data(str(file_path))
        if fit_data is None:
            continue
        fit_data_by_folder.setdefault(folder_path, []).append(fit_data)

    overlap_by_path = {}
    for fit_data_list in fit_data_by_folder.values():
        for fit_data_a, fit_data_b in _pair_fit_files_by_bounds(fit_data_list):
            overlap_metrics = _build_overlap_metrics(fit_data_a, fit_data_b)
            overlap_by_path[fit_data_a["path"]] = {
                **overlap_metrics,
                "paired_file_path": fit_data_b["path"],
            }
            overlap_by_path[fit_data_b["path"]] = {
                **overlap_metrics,
                "paired_file_path": fit_data_a["path"],
            }

    return overlap_by_path


def _build_fit_overlap_data(file_path):
    timestamps = read_fit_record_timestamps(file_path)
    if timestamps is None:
        return None
    if timestamps:
        start_timestamp = min(timestamps)
        end_timestamp = max(timestamps)
    else:
        start_timestamp = None
        end_timestamp = None
    return {
        "path": str(file_path),
        "timestamps": timestamps,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
    }


def _pair_fit_files_by_bounds(fit_data_list):
    candidates = []
    for fit_data_a, fit_data_b in combinations(fit_data_list, 2):
        overlap_seconds = _calculate_interval_overlap_seconds(fit_data_a, fit_data_b)
        if overlap_seconds is None or overlap_seconds < 0:
            continue
        start_delta = abs(
            fit_data_a["start_timestamp"] - fit_data_b["start_timestamp"]
        )
        candidates.append((overlap_seconds, -start_delta, fit_data_a, fit_data_b))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2]["start_timestamp"],
            item[3]["start_timestamp"],
        ),
        reverse=True,
    )

    paired_paths = set()
    selected_pairs = []
    for _, _, fit_data_a, fit_data_b in candidates:
        if fit_data_a["path"] in paired_paths or fit_data_b["path"] in paired_paths:
            continue
        paired_paths.add(fit_data_a["path"])
        paired_paths.add(fit_data_b["path"])
        selected_pairs.append((fit_data_a, fit_data_b))

    return selected_pairs


def _calculate_interval_overlap_seconds(fit_data_a, fit_data_b):
    if (
        fit_data_a["start_timestamp"] is None
        or fit_data_a["end_timestamp"] is None
        or fit_data_b["start_timestamp"] is None
        or fit_data_b["end_timestamp"] is None
    ):
        return None
    return min(fit_data_a["end_timestamp"], fit_data_b["end_timestamp"]) - max(
        fit_data_a["start_timestamp"], fit_data_b["start_timestamp"]
    )


def _build_overlap_metrics(fit_data_a, fit_data_b):
    common_timestamps = sorted(
        fit_data_a["timestamps"].intersection(fit_data_b["timestamps"])
    )
    return {
        "overlap_duration": _format_overlap_duration(common_timestamps),
        "overlap_datapoints": len(common_timestamps),
    }


def _format_overlap_duration(common_timestamps):
    if not common_timestamps:
        total_seconds = 0
    else:
        total_seconds = int(common_timestamps[-1] - common_timestamps[0])

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
