from pathlib import Path

from src.utils.cloud_cache import (
    clear_cloud_cache,
    get_cached_cloud_pair_common_metrics,
    get_cached_cloud_pair_summary,
    init_cloud_cache,
    put_cached_cloud_pair_common_metrics,
    put_cached_cloud_pair_summary,
)


def test_cloud_pair_summary_cache_round_trip(tmp_path: Path):
    db_path = tmp_path / "cloud_cache.sqlite3"
    init_cloud_cache(db_path)

    result_row = {
        "pair_id": "p1",
        "Group": "yifei",
        "Date": "2025-08-01",
        "Test File": "test.fit",
        "Ref File": "ref.fit",
        "Metric": "heart_rate",
        "Auto-shift": "Minimize MAE",
        "Status": "OK",
        "Applied Shift (s)": 1.0,
        "Mean Bias": 0.5,
        "MAE": 1.0,
    }
    put_cached_cloud_pair_summary(
        test_etag="test-etag",
        ref_etag="ref-etag",
        metric="heart_rate",
        auto_shift_method="Minimize MAE",
        result_row=result_row,
        common_metrics=["heart_rate", "cadence"],
        db_path=db_path,
    )

    cached_row = get_cached_cloud_pair_summary(
        test_etag="test-etag",
        ref_etag="ref-etag",
        metric="heart_rate",
        auto_shift_method="Minimize MAE",
        db_path=db_path,
    )

    assert cached_row == result_row
    assert get_cached_cloud_pair_common_metrics(
        test_etag="test-etag",
        ref_etag="ref-etag",
        db_path=db_path,
    ) == ["cadence", "heart_rate"]


def test_cloud_pair_summary_cache_misses_when_etag_changes(tmp_path: Path):
    db_path = tmp_path / "cloud_cache.sqlite3"
    init_cloud_cache(db_path)
    put_cached_cloud_pair_summary(
        test_etag="test-etag-v1",
        ref_etag="ref-etag",
        metric="heart_rate",
        auto_shift_method="Minimize MAE",
        result_row={"Status": "OK", "Mean Bias": 0.5},
        db_path=db_path,
    )

    cached_row = get_cached_cloud_pair_summary(
        test_etag="test-etag-v2",
        ref_etag="ref-etag",
        metric="heart_rate",
        auto_shift_method="Minimize MAE",
        db_path=db_path,
    )

    assert cached_row is None


def test_clear_cloud_cache_removes_cached_rows(tmp_path: Path):
    db_path = tmp_path / "cloud_cache.sqlite3"
    init_cloud_cache(db_path)
    put_cached_cloud_pair_summary(
        test_etag="test-etag",
        ref_etag="ref-etag",
        metric="heart_rate",
        auto_shift_method="Minimize MAE",
        result_row={"Status": "OK", "Mean Bias": 0.5},
        db_path=db_path,
    )

    clear_cloud_cache(db_path)

    cached_row = get_cached_cloud_pair_summary(
        test_etag="test-etag",
        ref_etag="ref-etag",
        metric="heart_rate",
        auto_shift_method="Minimize MAE",
        db_path=db_path,
    )

    assert cached_row is None


def test_cloud_pair_common_metrics_can_be_cached_without_summary(tmp_path: Path):
    db_path = tmp_path / "cloud_cache.sqlite3"
    init_cloud_cache(db_path)

    put_cached_cloud_pair_common_metrics(
        test_etag="test-etag",
        ref_etag="ref-etag",
        common_metrics=["heart_rate", "cadence"],
        db_path=db_path,
    )

    cached_metrics = get_cached_cloud_pair_common_metrics(
        test_etag="test-etag",
        ref_etag="ref-etag",
        db_path=db_path,
    )

    assert cached_metrics == ["cadence", "heart_rate"]
