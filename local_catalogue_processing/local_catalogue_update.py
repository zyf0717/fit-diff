import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from garmin_fit_sdk import Decoder, Stream


# Load .env file from root directory
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    # Support quoted values in .env (e.g. paths with spaces).
                    os.environ[key] = value.strip().strip("'").strip('"')


load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TAG_KEYWORDS = ["eq02", "pacer", "h10"]


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        logger.warning("Error reading config.yaml: %s", exc)
        return {}
    return data or {}




def _normalize_tag_map(raw_map):
    if isinstance(raw_map, dict):
        return raw_map
    if isinstance(raw_map, list):
        merged = {}
        for item in raw_map:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return {}


CONFIG = load_config()
FILENAME_TAG_KEYWORDS_MAP = _normalize_tag_map(
    CONFIG.get("filename_tag_keywords_map") or {}
)


def _get_level_tag_configs(config):
    levels = {}
    for key, value in config.items():
        key_name = str(key).strip().lower()
        map_match = re.fullmatch(r"l(\d+)_tag_keywords_map", key_name)
        if map_match:
            level_num = int(map_match.group(1))
            level_map = _normalize_tag_map(value)
            levels.setdefault(level_num, {})["map"] = level_map
    if not levels:
        return [{"level": 1, "map": {}}]
    configs = []
    for level_num in sorted(levels.keys()):
        entry = levels[level_num]
        configs.append(
            {
                "level": level_num,
                "map": entry.get("map", {}),
            }
        )
    return configs


TAG_LEVELS = _get_level_tag_configs(CONFIG)


def list_local_files(directory):
    """
    List all files with the given extensions from a local directory
    """
    extension_patterns = [
        re.compile(r"\.fit$", re.IGNORECASE),
        re.compile(r"\.(csv|xlsx)$", re.IGNORECASE),
    ]
    search_patterns = [
        re.compile(r"eq02", re.IGNORECASE),
        re.compile(r"eqlifemonitor", re.IGNORECASE),
        re.compile(r"pacer", re.IGNORECASE),
    ]
    matched_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if extension_patterns[0].search(file):
                matched_files.append(os.path.join(root, file))
                continue
            if extension_patterns[1].search(file) and any(
                p.search(file) for p in search_patterns
            ):
                matched_files.append(os.path.join(root, file))
    return matched_files


def build_file_dataframe(files, base_dir):
    """
    Build a dataframe with filename, relative_path, and tags.
    """
    date_pattern = re.compile(r"\d{8}")
    columns = [
        "date",
        "filename",
        "relative_path",
        "tags",
    ]
    df = (
        pd.DataFrame(
            [_build_file_row(path, base_dir, date_pattern) for path in files],
            columns=columns,
        )
        .sort_values("relative_path")
        .reset_index(drop=True)
    )

    return df


def _extract_iso_date(filename, date_pattern):
    match = date_pattern.search(filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


# def _fit_epoch_to_gmt8_iso(epoch_seconds):
#     gmt8 = timezone(timedelta(hours=8))
#     return datetime.fromtimestamp(epoch_seconds, tz=gmt8).isoformat(timespec="seconds")
#
#
# def _extract_fit_time_bounds(file_path):
#     logger.info(
#         "Reading FIT file for time bounds: %s", os.path.basename(str(file_path))
#     )
#     try:
#         stream = Stream.from_file(str(file_path))
#         decoder = Decoder(stream)
#         messages, _ = decoder.read(
#             apply_scale_and_offset=True,
#             convert_datetimes_to_dates=False,
#             convert_types_to_strings=True,
#             enable_crc_check=True,
#             expand_sub_fields=True,
#             expand_components=True,
#             merge_heart_rates=True,
#             mesg_listener=None,
#         )
#     except Exception as exc:
#         logger.warning(
#             "Error reading FIT file %s: %s", os.path.basename(str(file_path)), exc
#         )
#         return None, None
#
#     timestamps = [
#         msg.get("timestamp")
#         for msg in messages.get("record_mesgs", [])
#         if msg.get("timestamp") is not None
#     ]
#     if timestamps:
#         fit_start = min(timestamps)
#         fit_end = max(timestamps)
#     else:
#         session_mesgs = messages.get("session_mesgs", [])
#         starts = [
#             msg.get("start_time")
#             for msg in session_mesgs
#             if msg.get("start_time") is not None
#         ]
#         ends = [
#             msg.get("timestamp")
#             for msg in session_mesgs
#             if msg.get("timestamp") is not None
#         ]
#         fit_start = min(starts) if starts else None
#         fit_end = max(ends) if ends else None
#
#     if fit_start is None and fit_end is None:
#         return None, None
#
#     fit_epoch_offset = 631065600
#     start_iso = (
#         _fit_epoch_to_gmt8_iso(fit_start + fit_epoch_offset)
#         if fit_start is not None
#         else None
#     )
#     end_iso = (
#         _fit_epoch_to_gmt8_iso(fit_end + fit_epoch_offset)
#         if fit_end is not None
#         else None
#     )
#     logger.info(
#         "Parsed FIT time bounds for %s: start_time=%s end_time=%s",
#         os.path.basename(str(file_path)),
#         start_iso,
#         end_iso,
#     )
#     return start_iso, end_iso
#
#
# def _datetime_to_gmt8_iso(dt_value):
#     gmt8 = timezone(timedelta(hours=8))
#     if dt_value.tzinfo is None:
#         dt_value = dt_value.replace(tzinfo=gmt8)
#     return dt_value.astimezone(gmt8).isoformat(timespec="seconds")
#
#
# def _log_time_bounds(file_path, start_iso, end_iso):
#     logger.info(
#         "Parsed tabular time bounds for %s: start_time=%s end_time=%s",
#         os.path.basename(str(file_path)),
#         start_iso,
#         end_iso,
#     )


def _match_map_tags(values, tag_map):
    matched = []
    seen = set()
    for key, value in tag_map.items():
        key_str = str(key).lower()
        if any(key_str in candidate for candidate in values):
            mapped = str(value).lower()
            if mapped not in seen:
                matched.append(mapped)
                seen.add(mapped)
    return matched


def build_level_tags(file_path, base_dir):
    if base_dir:
        rel_path = os.path.relpath(file_path, base_dir)
    else:
        rel_path = str(file_path)
    parts = list(Path(rel_path).parts)
    if parts:
        parts = parts[:-1]
    merged = []
    seen = set()
    for level in TAG_LEVELS:
        level_map = level.get("map", {})
        level_index = level["level"] - 1
        if level_index < 0 or level_index >= len(parts):
            continue
        segment = parts[level_index].lower()
        matched = _match_map_tags([segment], level_map)
        for tag in matched:
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
    return merged


def _merge_tag_lists(*tag_lists):
    merged = []
    seen = set()
    for tags in tag_lists:
        for tag in tags:
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
    return merged


# def _parse_datetime_series(values, formats):
#     for fmt in formats:
#         parsed = pd.to_datetime(values, format=fmt, errors="coerce")
#         if not parsed.isna().all():
#             return parsed.dropna()
#     return pd.Series(dtype="datetime64[ns]")
#
#
# def _extract_tabular_time_bounds(file_path):
#     logger.info(
#         "Reading tabular file for time bounds: %s",
#         os.path.basename(str(file_path)),
#     )
#     try:
#         is_csv = str(file_path).lower().endswith(".csv")
#         is_xlsx = str(file_path).lower().endswith(".xlsx")
#         if is_csv:
#             df = pd.read_csv(file_path, dtype=str)
#         elif is_xlsx:
#             df = pd.read_excel(file_path, dtype=str, header=2)
#         else:
#             logger.warning(
#                 "Unsupported tabular file type %s", os.path.basename(str(file_path))
#             )
#             return None, None
#     except Exception as exc:
#         logger.warning(
#             "Error reading tabular file %s: %s",
#             os.path.basename(str(file_path)),
#             exc,
#         )
#         return None, None
#
#     if df.empty:
#         return None, None
#
#     if "Time (HH:mm:ss.000)" in df.columns and (
#         "Date (d/M/yyyy)" in df.columns or "Date (M/d/yyyy)" in df.columns
#     ):
#         date_col = (
#             "Date (d/M/yyyy)" if "Date (d/M/yyyy)" in df.columns else "Date (M/d/yyyy)"
#         )
#         date_format = "%d/%m/%Y" if date_col == "Date (d/M/yyyy)" else "%m/%d/%Y"
#         combined = (
#             df[date_col].astype(str).str.strip()
#             + " "
#             + df["Time (HH:mm:ss.000)"].astype(str).str.strip()
#         )
#         timestamps = _parse_datetime_series(
#             combined,
#             [f"{date_format} %H:%M:%S.%f", f"{date_format} %H:%M:%S"],
#         )
#         if timestamps.empty:
#             return None, None
#         start_iso = _datetime_to_gmt8_iso(timestamps.min())
#         end_iso = _datetime_to_gmt8_iso(timestamps.max())
#         _log_time_bounds(file_path, start_iso, end_iso)
#         return start_iso, end_iso
#
#     if {"Date", "Start time", "Duration"}.issubset(df.columns):
#         meta = df[["Date", "Start time", "Duration"]].dropna().head(1)
#         if meta.empty:
#             return None, None
#         date_str = str(meta.iloc[0]["Date"]).strip()
#         start_str = str(meta.iloc[0]["Start time"]).strip()
#         duration_str = str(meta.iloc[0]["Duration"]).strip()
#         start_dt = pd.to_datetime(
#             f"{date_str} {start_str}", format="%d-%m-%Y %H:%M:%S", errors="coerce"
#         )
#         if pd.isna(start_dt):
#             return None, None
#         duration = pd.to_timedelta(duration_str, errors="coerce")
#         if pd.isna(duration):
#             start_iso = _datetime_to_gmt8_iso(start_dt)
#             _log_time_bounds(file_path, start_iso, None)
#             return start_iso, None
#         end_dt = start_dt + duration
#         start_iso = _datetime_to_gmt8_iso(start_dt)
#         end_iso = _datetime_to_gmt8_iso(end_dt)
#         _log_time_bounds(file_path, start_iso, end_iso)
#         return start_iso, end_iso
#
#     logger.warning(
#         "No supported timestamp columns in tabular file %s",
#         os.path.basename(str(file_path)),
#     )
#     return None, None


def _build_file_row(path, base_dir, date_pattern):
    filename = os.path.basename(path)
    level_tags = build_level_tags(path, base_dir)
    filename_value = os.path.splitext(os.path.basename(str(path)))[0].lower()
    filename_tags = _match_map_tags(
        [filename_value], FILENAME_TAG_KEYWORDS_MAP
    )
    tags = _merge_tag_lists(level_tags, filename_tags)
    if "h10" not in tags and "eq02" not in tags and "pacer" not in tags:
        tags.append("pacer")
    row = {
        "filename": filename,
        "relative_path": os.path.relpath(path, base_dir),
        "date": _extract_iso_date(filename, date_pattern),
        "tags": "|".join(tags) if tags else None,
    }
    return row


if __name__ == "__main__":
    start_time = time.perf_counter()
    directory = os.getenv("LOCAL_FOLDER_PATH")
    files = list_local_files(directory)
    df = build_file_dataframe(files, directory)
    df.to_csv("local_files_catalogue.csv", index=False)
    elapsed = time.perf_counter() - start_time
    logger.info("local_catalogue_update completed in %.2f seconds", elapsed)
