"""Local SQLite cache for cloud pair benchmark summaries."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CLOUD_PAIR_SUMMARY_CACHE_SCHEMA_VERSION = "cloud_pair_summary_v1"
DEFAULT_CACHE_DB_PATH = (
    Path(__file__).resolve().parents[2] / ".cache" / "fit_diff_cloud_cache.sqlite3"
)


def get_cloud_cache_db_path() -> Path:
    """Return the configured local SQLite cache path."""
    configured_path = os.getenv("FIT_DIFF_CLOUD_CACHE_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_CACHE_DB_PATH


def _connect_cache_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the local cache database, creating parent directories as needed."""
    resolved_path = db_path or get_cloud_cache_db_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def init_cloud_cache(db_path: Path | None = None) -> None:
    """Ensure the cloud benchmark cache schema exists."""
    with _connect_cache_db(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cloud_pair_summary_cache (
                cache_key TEXT PRIMARY KEY,
                test_etag TEXT NOT NULL,
                ref_etag TEXT NOT NULL,
                metric TEXT NOT NULL,
                auto_shift_method TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cloud_pair_summary_lookup
            ON cloud_pair_summary_cache (
                test_etag,
                ref_etag,
                metric,
                auto_shift_method,
                schema_version
            )
            """
        )


def build_cloud_pair_summary_cache_key(
    test_etag: str,
    ref_etag: str,
    metric: str,
    auto_shift_method: str,
    schema_version: str = CLOUD_PAIR_SUMMARY_CACHE_SCHEMA_VERSION,
) -> str:
    """Build a stable cache key for one benchmarked test/ref pair."""
    raw_key = "|".join(
        [test_etag, ref_etag, metric, auto_shift_method, schema_version]
    ).encode("utf-8")
    return hashlib.sha256(raw_key).hexdigest()


def _normalize_json_value(value: Any):
    """Convert pandas/numpy scalars to JSON-safe Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def get_cached_cloud_pair_summary(
    test_etag: str,
    ref_etag: str,
    metric: str,
    auto_shift_method: str,
    schema_version: str = CLOUD_PAIR_SUMMARY_CACHE_SCHEMA_VERSION,
    db_path: Path | None = None,
):
    """Return a cached cloud pair summary row when present."""
    init_cloud_cache(db_path)
    cache_key = build_cloud_pair_summary_cache_key(
        test_etag=test_etag,
        ref_etag=ref_etag,
        metric=metric,
        auto_shift_method=auto_shift_method,
        schema_version=schema_version,
    )
    with _connect_cache_db(db_path) as connection:
        row = connection.execute(
            """
            SELECT result_json
            FROM cloud_pair_summary_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["result_json"])


def put_cached_cloud_pair_summary(
    test_etag: str,
    ref_etag: str,
    metric: str,
    auto_shift_method: str,
    result_row: dict[str, Any],
    schema_version: str = CLOUD_PAIR_SUMMARY_CACHE_SCHEMA_VERSION,
    db_path: Path | None = None,
) -> None:
    """Persist one cloud pair summary row in the local SQLite cache."""
    init_cloud_cache(db_path)
    cache_key = build_cloud_pair_summary_cache_key(
        test_etag=test_etag,
        ref_etag=ref_etag,
        metric=metric,
        auto_shift_method=auto_shift_method,
        schema_version=schema_version,
    )
    now = datetime.now(UTC).isoformat()
    normalized_row = {
        key: _normalize_json_value(value) for key, value in result_row.items()
    }
    with _connect_cache_db(db_path) as connection:
        connection.execute(
            """
            INSERT INTO cloud_pair_summary_cache (
                cache_key,
                test_etag,
                ref_etag,
                metric,
                auto_shift_method,
                schema_version,
                status,
                created_at,
                updated_at,
                result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                status = excluded.status,
                updated_at = excluded.updated_at,
                result_json = excluded.result_json
            """,
            (
                cache_key,
                test_etag,
                ref_etag,
                metric,
                auto_shift_method,
                schema_version,
                str(normalized_row.get("Status", "UNKNOWN")),
                now,
                now,
                json.dumps(normalized_row, sort_keys=True),
            ),
        )


def clear_cloud_cache(db_path: Path | None = None) -> None:
    """Delete all cached cloud pair summaries."""
    init_cloud_cache(db_path)
    with _connect_cache_db(db_path) as connection:
        connection.execute("DELETE FROM cloud_pair_summary_cache")
