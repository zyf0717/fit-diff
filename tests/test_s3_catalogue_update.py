from datetime import datetime, timezone

import pandas as pd

from s3_catalogue_processing.s3_catalogue_update import build_file_dataframe
def _make_processor(timestamp_map):
    def fake_processor(s3_url, aws_profile=None):
        timestamps = timestamp_map[s3_url]
        return {
            "records": pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "filename": [s3_url.split("/")[-1]] * len(timestamps),
                }
            )
        }

    return fake_processor


def test_build_file_dataframe_includes_time_bounds_and_simple_pair():
    files = [
        {
            "key": "fit_files/yifei/20250910_watchOS_h10.fit",
            "etag": "ref-etag",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 3, 0, tzinfo=timezone.utc),
        },
        {
            "key": "fit_files/yifei/20250910_0jqzV7D4_polar_pacer.fit",
            "etag": "test-etag",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 3, 5, tzinfo=timezone.utc),
        },
    ]
    processor = _make_processor(
        {
            "s3://bucket/fit_files/yifei/20250910_watchOS_h10.fit": [100, 130, 160],
            "s3://bucket/fit_files/yifei/20250910_0jqzV7D4_polar_pacer.fit": [
                120,
                150,
                180,
            ],
        }
    )

    df = build_file_dataframe(
        files,
        prefix="fit_files/",
        bucket_name="bucket",
        file_processor=processor,
    )

    assert list(df["etag"]) == ["ref-etag", "test-etag"]
    assert list(df["paired_etag"]) == ["test-etag", "ref-etag"]
    assert list(df["pair_index"]) == [1, 1]
    assert list(df["start_datetime"]) == [
        "1970-01-01T00:01:40+00:00",
        "1970-01-01T00:02:00+00:00",
    ]
    assert list(df["end_datetime"]) == [
        "1970-01-01T00:02:40+00:00",
        "1970-01-01T00:03:00+00:00",
    ]
    assert list(df["duration_seconds"]) == [60, 60]
    assert list(df["paired_overlap_seconds"]) == [40, 40]
    assert list(df["paired_overlap_pct"]) == [40 / 60, 40 / 60]


def test_build_file_dataframe_does_not_pair_non_overlapping_files():
    files = [
        {
            "key": "fit_files/yifei/20250910_watchOS_h10.fit",
            "etag": "ref-etag",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 3, 0, tzinfo=timezone.utc),
        },
        {
            "key": "fit_files/yifei/20250910_0jqzV7D4_polar_pacer.fit",
            "etag": "test-etag",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 3, 5, tzinfo=timezone.utc),
        },
    ]
    processor = _make_processor(
        {
            "s3://bucket/fit_files/yifei/20250910_watchOS_h10.fit": [100, 110, 120],
            "s3://bucket/fit_files/yifei/20250910_0jqzV7D4_polar_pacer.fit": [
                200,
                210,
                220,
            ],
        }
    )

    df = build_file_dataframe(
        files,
        prefix="fit_files/",
        bucket_name="bucket",
        file_processor=processor,
    )

    assert df["paired_etag"].isna().all()
    assert df["pair_index"].isna().all()
    assert df["paired_overlap_seconds"].isna().all()
    assert df["paired_overlap_pct"].isna().all()


def test_build_file_dataframe_uses_highest_overlap_pct_for_ambiguous_component():
    files = [
        {
            "key": "fit_files/yifei/20250910_ref_a_h10.fit",
            "etag": "ref-a",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 3, 0, tzinfo=timezone.utc),
        },
        {
            "key": "fit_files/yifei/20250910_ref_b_h10.fit",
            "etag": "ref-b",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 5, 0, tzinfo=timezone.utc),
        },
        {
            "key": "fit_files/yifei/20250910_test_a_pacer.fit",
            "etag": "test-a",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 3, 10, tzinfo=timezone.utc),
        },
        {
            "key": "fit_files/yifei/20250910_test_b_pacer.fit",
            "etag": "test-b",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 5, 10, tzinfo=timezone.utc),
        },
    ]
    processor = _make_processor(
        {
            "s3://bucket/fit_files/yifei/20250910_ref_a_h10.fit": [20, 70, 120],
            "s3://bucket/fit_files/yifei/20250910_ref_b_h10.fit": [90, 125, 160],
            "s3://bucket/fit_files/yifei/20250910_test_a_pacer.fit": [0, 50, 100],
            "s3://bucket/fit_files/yifei/20250910_test_b_pacer.fit": [40, 90, 140],
        }
    )

    df = build_file_dataframe(
        files,
        prefix="fit_files/",
        bucket_name="bucket",
        file_processor=processor,
    )

    by_etag = df.set_index("etag")
    assert by_etag.loc["test-a", "paired_etag"] == "ref-a"
    assert by_etag.loc["ref-a", "paired_etag"] == "test-a"
    assert by_etag.loc["test-b", "paired_etag"] == "ref-b"
    assert by_etag.loc["ref-b", "paired_etag"] == "test-b"
    assert by_etag.loc["test-a", "paired_overlap_seconds"] == 80
    assert by_etag.loc["test-b", "paired_overlap_seconds"] == 50
    assert by_etag.loc["test-a", "paired_overlap_pct"] == 0.8
    assert by_etag.loc["test-b", "paired_overlap_pct"] == 0.5


def test_build_file_dataframe_leaves_bounds_empty_when_fit_read_fails():
    files = [
        {
            "key": "fit_files/yifei/20250910_watchOS_h10.fit",
            "etag": "ref-etag",
            "size_mb": 0.1,
            "last_modified": datetime(2025, 9, 10, 3, 0, tzinfo=timezone.utc),
        }
    ]

    def failing_processor(s3_url, aws_profile=None):
        raise RuntimeError("read failed")

    df = build_file_dataframe(
        files,
        prefix="fit_files/",
        bucket_name="bucket",
        file_processor=failing_processor,
    )

    row = df.iloc[0]
    assert pd.isna(row["start_datetime"])
    assert pd.isna(row["end_datetime"])
    assert pd.isna(row["duration_seconds"])
    assert pd.isna(row["paired_etag"])
    assert pd.isna(row["pair_index"])
