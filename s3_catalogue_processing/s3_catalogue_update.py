import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import aioboto3
import pandas as pd
import yaml
from botocore.exceptions import ClientError, NoCredentialsError


def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.strip().strip("'").strip('"')


load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CATALOGUE_FILE = SCRIPT_DIR / "fit_files_catalogue.csv"
SYNC_SCRIPT = SCRIPT_DIR / "sync_catalogue_to_s3.sh"
S3_PREFIX = os.getenv("FIT_FILES_S3_PREFIX", "fit_files/")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_PROFILE = os.getenv("AWS_PROFILE")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.data_processing import process_file

EMPTY_TIME_BOUNDS = {
    "start_epoch": None,
    "end_epoch": None,
    "start_datetime": None,
    "end_datetime": None,
    "duration_seconds": None,
}

CATALOGUE_COLUMNS = [
    "etag",
    "paired_etag",
    "pairing_group",
    "pair_index",
    "date",
    "filename",
    "device_type",
    "tags",
    "s3_key",
    "size_mb",
    "last_modified",
    "start_datetime",
    "end_datetime",
    "duration_seconds",
    "paired_overlap_seconds",
    "paired_overlap_pct",
]


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
PATH_TAG_KEYWORD_MAP = {}
for config_key, config_value in CONFIG.items():
    if re.fullmatch(r"l\d+_tag_keywords_map", str(config_key).strip().lower()):
        PATH_TAG_KEYWORD_MAP.update(_normalize_tag_map(config_value))


def _extract_iso_date(filename, date_pattern):
    match = date_pattern.search(filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def _tokenize_path_part(part):
    values = [part.lower()]
    values.extend(token for token in re.split(r"[^a-zA-Z0-9]+", part.lower()) if token)
    return values


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


def _merge_tag_lists(*tag_lists):
    merged = []
    seen = set()
    for tags in tag_lists:
        for tag in tags:
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
    return merged


def _relative_path_from_key(key, prefix):
    if prefix and key.startswith(prefix):
        return key[len(prefix) :]
    return key


def _build_path_tags(relative_path):
    parts = list(Path(relative_path).parts[:-1])
    candidates = []
    for part in parts:
        candidates.extend(_tokenize_path_part(part))
    return _match_map_tags(candidates, PATH_TAG_KEYWORD_MAP)


def _build_file_row(file_info, prefix, date_pattern):
    key = file_info["key"]
    filename = Path(key).name
    relative_path = _relative_path_from_key(key, prefix)
    pairing_group = str(Path(relative_path).parent)
    filename_stem = Path(filename).stem.lower()

    path_tags = _build_path_tags(relative_path)
    filename_tags = _match_map_tags([filename_stem], FILENAME_TAG_KEYWORDS_MAP)
    tags = _merge_tag_lists(path_tags, filename_tags)

    device_type = "ref" if "h10" in tags or "eq02" in tags else "test"
    if "h10" not in tags and "eq02" not in tags and "pacer" not in tags:
        tags.append("pacer")

    last_modified = file_info.get("last_modified")
    if hasattr(last_modified, "isoformat"):
        last_modified = last_modified.isoformat(timespec="seconds")

    return {
        "etag": file_info.get("etag"),
        "date": _extract_iso_date(filename, date_pattern),
        "filename": filename,
        "device_type": device_type,
        "tags": "|".join(tags) if tags else None,
        "pairing_group": pairing_group,
        "s3_key": key,
        "size_mb": file_info.get("size_mb"),
        "last_modified": last_modified,
    }


def _epoch_to_iso(timestamp_value):
    if timestamp_value is None:
        return None
    return datetime.fromtimestamp(
        int(timestamp_value), tz=timezone.utc
    ).isoformat(timespec="seconds")


def _extract_time_bounds(s3_key, bucket_name, aws_profile=None, file_processor=None):
    processor = file_processor or process_file
    s3_url = f"s3://{bucket_name}/{s3_key}"
    try:
        record_df = processor(s3_url, aws_profile=aws_profile)["records"]
    except Exception as exc:
        logger.warning("Failed to read FIT timestamps for %s: %s", s3_url, exc)
        return EMPTY_TIME_BOUNDS.copy()

    if "timestamp" not in record_df.columns:
        return EMPTY_TIME_BOUNDS.copy()

    timestamps = record_df["timestamp"].dropna()
    if timestamps.empty:
        return EMPTY_TIME_BOUNDS.copy()

    timestamps = timestamps.astype("int64")
    start_epoch = int(timestamps.min())
    end_epoch = int(timestamps.max())
    duration_seconds = max(end_epoch - start_epoch, 0)
    return {
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "start_datetime": _epoch_to_iso(start_epoch),
        "end_datetime": _epoch_to_iso(end_epoch),
        "duration_seconds": duration_seconds,
    }


def _enrich_with_time_bounds(df, bucket_name, aws_profile=None, file_processor=None):
    df = df.copy()
    metadata_by_key = {}
    for s3_key in df["s3_key"].tolist():
        metadata_by_key[s3_key] = _extract_time_bounds(
            s3_key,
            bucket_name=bucket_name,
            aws_profile=aws_profile,
            file_processor=file_processor,
        )

    for column in [
        "start_epoch",
        "end_epoch",
        "start_datetime",
        "end_datetime",
        "duration_seconds",
    ]:
        df[column] = df["s3_key"].map(lambda key: metadata_by_key[key][column])
    return df


def _calculate_overlap_seconds(start_a, end_a, start_b, end_b):
    if None in {start_a, end_a, start_b, end_b}:
        return None
    overlap_seconds = min(end_a, end_b) - max(start_a, start_b)
    return max(overlap_seconds, 0)


def _calculate_overlap_pct(overlap_seconds, duration_a, duration_b):
    if overlap_seconds is None or duration_a is None or duration_b is None:
        return None
    denominator = max(duration_a, duration_b)
    if denominator == 0:
        return 1.0 if overlap_seconds == 0 else None
    return overlap_seconds / denominator


def _build_overlap_candidates(group_df):
    candidates = {}
    for test_row in group_df[group_df["device_type"] == "test"].itertuples(index=False):
        for ref_row in group_df[group_df["device_type"] == "ref"].itertuples(index=False):
            overlap_seconds = _calculate_overlap_seconds(
                test_row.start_epoch,
                test_row.end_epoch,
                ref_row.start_epoch,
                ref_row.end_epoch,
            )
            if overlap_seconds is None or overlap_seconds <= 0:
                continue
            overlap_pct = _calculate_overlap_pct(
                overlap_seconds,
                test_row.duration_seconds,
                ref_row.duration_seconds,
            )
            candidates[(test_row.etag, ref_row.etag)] = {
                "test_etag": test_row.etag,
                "ref_etag": ref_row.etag,
                "overlap_seconds": overlap_seconds,
                "overlap_pct": overlap_pct,
                "overlap_start_epoch": max(test_row.start_epoch, ref_row.start_epoch),
                "overlap_end_epoch": min(test_row.end_epoch, ref_row.end_epoch),
            }
    return candidates


def _build_overlap_components(group_df, candidate_map):
    adjacency = {etag: set() for etag in group_df["etag"].tolist()}
    for test_etag, ref_etag in candidate_map.keys():
        adjacency[test_etag].add(ref_etag)
        adjacency[ref_etag].add(test_etag)

    components = []
    visited = set()
    for etag in adjacency:
        if etag in visited or not adjacency[etag]:
            continue
        stack = [etag]
        component = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            stack.extend(adjacency[node] - visited)
        components.append(component)
    return components


def _select_best_pairs(component_tests, component_refs, candidate_map):
    tests = list(component_tests)
    refs = list(component_refs)

    @lru_cache(maxsize=None)
    def solve(test_index, used_ref_mask):
        best = (0, 0.0, 0, tuple())
        if test_index >= len(tests):
            return best

        best = solve(test_index + 1, used_ref_mask)
        current_test_etag = tests[test_index]
        for ref_index, ref_etag in enumerate(refs):
            if used_ref_mask & (1 << ref_index):
                continue
            candidate = candidate_map.get((current_test_etag, ref_etag))
            if candidate is None:
                continue

            tail = solve(test_index + 1, used_ref_mask | (1 << ref_index))
            overlap_pct = candidate["overlap_pct"] or 0.0
            proposal = (
                tail[0] + 1,
                tail[1] + overlap_pct,
                tail[2] + candidate["overlap_seconds"],
                ((current_test_etag, ref_etag),) + tail[3],
            )
            if proposal[:3] > best[:3]:
                best = proposal
        return best

    return list(solve(0, 0)[3])


def _pair_group_rows(df):
    paired_etags = {etag: None for etag in df["etag"].tolist()}
    pair_indexes = {etag: None for etag in df["etag"].tolist()}
    paired_overlap_seconds = {etag: None for etag in df["etag"].tolist()}
    paired_overlap_pct = {etag: None for etag in df["etag"].tolist()}

    for _, group_df in df.groupby("pairing_group", dropna=False, sort=False):
        candidate_map = _build_overlap_candidates(group_df)
        components = _build_overlap_components(group_df, candidate_map)
        next_pair_index = 1

        for component in components:
            component_df = group_df[group_df["etag"].isin(component)]
            component_tests = sorted(
                component_df[component_df["device_type"] == "test"]["etag"].tolist()
            )
            component_refs = sorted(
                component_df[component_df["device_type"] == "ref"]["etag"].tolist()
            )
            if not component_tests or not component_refs:
                continue

            is_simple_pair = (
                len(component) == 2
                and len(component_tests) == 1
                and len(component_refs) == 1
            )
            matched_pairs = (
                [(component_tests[0], component_refs[0])]
                if is_simple_pair
                else _select_best_pairs(component_tests, component_refs, candidate_map)
            )

            for test_etag, ref_etag in sorted(
                matched_pairs,
                key=lambda pair: (
                    candidate_map[pair]["overlap_start_epoch"],
                    -candidate_map[pair]["overlap_seconds"],
                    pair[0],
                    pair[1],
                ),
            ):
                candidate = candidate_map[(test_etag, ref_etag)]
                paired_etags[test_etag] = ref_etag
                paired_etags[ref_etag] = test_etag
                pair_indexes[test_etag] = next_pair_index
                pair_indexes[ref_etag] = next_pair_index
                paired_overlap_seconds[test_etag] = candidate["overlap_seconds"]
                paired_overlap_seconds[ref_etag] = candidate["overlap_seconds"]
                paired_overlap_pct[test_etag] = candidate["overlap_pct"]
                paired_overlap_pct[ref_etag] = candidate["overlap_pct"]
                next_pair_index += 1

    df = df.copy()
    df["paired_etag"] = df["etag"].map(paired_etags)
    df["pair_index"] = df["etag"].map(pair_indexes)
    df["paired_overlap_seconds"] = df["etag"].map(paired_overlap_seconds)
    df["paired_overlap_pct"] = df["etag"].map(paired_overlap_pct)
    return df


def build_file_dataframe(
    files, prefix, bucket_name=None, aws_profile=None, file_processor=None
):
    date_pattern = re.compile(r"\d{8}")
    resolved_bucket = bucket_name or S3_BUCKET
    resolved_profile = aws_profile or AWS_PROFILE
    df = pd.DataFrame(
        [_build_file_row(file_info, prefix, date_pattern) for file_info in files]
    )
    df = _enrich_with_time_bounds(
        df,
        bucket_name=resolved_bucket,
        aws_profile=resolved_profile,
        file_processor=file_processor,
    )
    df = _pair_group_rows(df)
    df = df.loc[:, CATALOGUE_COLUMNS].sort_values(
        [
            "pairing_group",
            "start_datetime",
            "pair_index",
            "device_type",
            "filename",
            "etag",
        ]
    )
    df = df.reset_index(drop=True)
    return df


async def list_s3_fit_files(bucket_name, prefix, profile_name=None):
    files = []
    session = (
        aioboto3.Session(profile_name=profile_name)
        if profile_name
        else aioboto3.Session()
    )

    try:
        async with session.client("s3") as s3:
            paginator = s3.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            async for page in page_iterator:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.lower().endswith(".fit"):
                        continue
                    files.append(
                        {
                            "key": key,
                            "etag": obj.get("ETag", "").strip('"'),
                            "size_mb": round(obj["Size"] / (1024 * 1024), 2),
                            "last_modified": obj.get("LastModified"),
                        }
                    )
    except (NoCredentialsError, ClientError) as exc:
        logger.error("S3 error: %s", exc)
        return []
    except Exception as exc:
        logger.error("Unexpected error while listing S3 files: %s", exc)
        return []

    return sorted(files, key=lambda item: item["key"])


def sync_catalogue_to_s3():
    if not SYNC_SCRIPT.exists():
        logger.warning("Sync script not found: %s", SYNC_SCRIPT)
        return
    try:
        subprocess.run(
            [str(SYNC_SCRIPT)],
            check=True,
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
        )
        logger.info("Successfully synced catalogue to S3")
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to sync catalogue to S3: %s", exc.stderr or exc)


async def main():
    start_time = time.perf_counter()

    if not S3_BUCKET:
        logger.error("Missing S3_BUCKET. Check .env file.")
        raise SystemExit(1)

    logger.info("Updating catalogue from S3 bucket %s with prefix %s", S3_BUCKET, S3_PREFIX)
    fit_files = await list_s3_fit_files(S3_BUCKET, S3_PREFIX, AWS_PROFILE)
    if not fit_files:
        logger.info("No .fit files found in s3://%s/%s", S3_BUCKET, S3_PREFIX)
        return

    df = build_file_dataframe(fit_files, S3_PREFIX)
    df.to_csv(CATALOGUE_FILE, index=False)
    logger.info("Wrote %d catalogue rows to %s", len(df), CATALOGUE_FILE)

    sync_catalogue_to_s3()

    elapsed = time.perf_counter() - start_time
    logger.info("s3_catalogue_update completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    asyncio.run(main())
