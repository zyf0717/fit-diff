import pandas as pd

from local_catalogue_processing.fit_overlap import (
    build_folder_fit_overlap_metrics,
    compare_fit_files_by_timestamp,
)
from local_catalogue_processing.local_catalogue_update import build_file_dataframe


def test_compare_fit_files_by_timestamp_counts_common_timestamps(monkeypatch):
    timestamp_map = {
        "file_a.fit": {100, 101, 102, 105},
        "file_b.fit": {101, 102, 103, 105},
    }

    monkeypatch.setattr(
        "local_catalogue_processing.fit_overlap.read_fit_record_timestamps",
        lambda file_path: timestamp_map[file_path],
    )

    metrics = compare_fit_files_by_timestamp("file_a.fit", "file_b.fit")

    assert metrics == {
        "overlap_duration": "00:00:04",
        "overlap_datapoints": 3,
    }


def test_build_folder_fit_overlap_metrics_pairs_by_timestamp_bounds(monkeypatch):
    folder_a_fit_1 = "/tmp/session_a/20339867911_ACTIVITY_H10.fit"
    folder_a_fit_2 = "/tmp/session_a/20250910_PdZY711l_polar_pacer.fit"
    folder_a_fit_3 = "/tmp/session_a/20268236580_ACTIVITY_H10.fit"
    folder_a_fit_4 = "/tmp/session_a/20250903_0LY2WmAM_polar_pacer.fit"
    folder_a_fit_5 = "/tmp/session_a/backup_h10.fit"
    folder_b_fit_1 = "/tmp/session_b/watch.fit"
    folder_b_fit_2 = "/tmp/session_b/chest.fit"

    timestamp_map = {
        folder_a_fit_1: {10, 11, 12},
        folder_a_fit_2: {11, 12, 13},
        folder_a_fit_3: {30, 31},
        folder_a_fit_4: {30, 31},
        folder_a_fit_5: {30, 31},
        folder_b_fit_1: {20, 21, 22},
        folder_b_fit_2: {21, 22, 23},
    }

    monkeypatch.setattr(
        "local_catalogue_processing.fit_overlap.read_fit_record_timestamps",
        lambda file_path: timestamp_map[file_path],
    )

    overlap_by_path = build_folder_fit_overlap_metrics(
        [
            folder_a_fit_1,
            folder_a_fit_2,
            folder_a_fit_3,
            folder_a_fit_4,
            folder_a_fit_5,
            "/tmp/session_a/eq02_export.csv",
            folder_b_fit_1,
            folder_b_fit_2,
        ]
    )

    assert overlap_by_path == {
        folder_a_fit_1: {
            "paired_file_path": folder_a_fit_2,
            "overlap_duration": "00:00:01",
            "overlap_datapoints": 2,
        },
        folder_a_fit_2: {
            "paired_file_path": folder_a_fit_1,
            "overlap_duration": "00:00:01",
            "overlap_datapoints": 2,
        },
        folder_a_fit_3: {
            "paired_file_path": folder_a_fit_4,
            "overlap_duration": "00:00:01",
            "overlap_datapoints": 2,
        },
        folder_a_fit_4: {
            "paired_file_path": folder_a_fit_3,
            "overlap_duration": "00:00:01",
            "overlap_datapoints": 2,
        },
        folder_b_fit_1: {
            "paired_file_path": folder_b_fit_2,
            "overlap_duration": "00:00:01",
            "overlap_datapoints": 2,
        },
        folder_b_fit_2: {
            "paired_file_path": folder_b_fit_1,
            "overlap_duration": "00:00:01",
            "overlap_datapoints": 2,
        },
    }


def test_build_file_dataframe_includes_overlap_columns(monkeypatch, tmp_path):
    session_dir = tmp_path / "participant" / "session_1"
    session_dir.mkdir(parents=True)

    fit_a = str(session_dir / "watch_20240101.fit")
    fit_b = str(session_dir / "h10_20240101.fit")
    csv_file = str(session_dir / "eq02_export.csv")

    monkeypatch.setattr(
        "local_catalogue_processing.local_catalogue_update.build_folder_fit_overlap_metrics",
        lambda files: {
            fit_a: {
                "paired_file_path": fit_b,
                "overlap_duration": "00:10:00",
                "overlap_datapoints": 601,
            },
            fit_b: {
                "paired_file_path": fit_a,
                "overlap_duration": "00:10:00",
                "overlap_datapoints": 601,
            },
        },
    )

    df = build_file_dataframe([fit_a, fit_b, csv_file], str(tmp_path))

    assert "paired_file_path" in df.columns
    assert "overlap_duration" in df.columns
    assert "overlap_datapoints" in df.columns
    assert str(df["overlap_datapoints"].dtype) == "Int64"

    fit_rows = df[df["filename"].isin(["watch_20240101.fit", "h10_20240101.fit"])]
    paired_paths = dict(zip(fit_rows["filename"], fit_rows["paired_file_path"]))
    assert paired_paths == {
        "watch_20240101.fit": "participant/session_1/h10_20240101.fit",
        "h10_20240101.fit": "participant/session_1/watch_20240101.fit",
    }
    assert fit_rows["overlap_duration"].tolist() == ["00:10:00", "00:10:00"]
    assert fit_rows["overlap_datapoints"].tolist() == [601, 601]

    csv_row = df[df["filename"] == "eq02_export.csv"].iloc[0]
    assert csv_row["paired_file_path"] is None
    assert csv_row["overlap_duration"] is None
    assert pd.isna(csv_row["overlap_datapoints"])
