import pytest

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

    fit_a = str(session_dir / "polar_pacer_20240101.fit")
    fit_b = str(session_dir / "h10_20240101.fit")

    monkeypatch.setattr(
        "local_catalogue_processing.local_catalogue_update.build_folder_fit_overlap_metrics",
        lambda files, role_by_path: {
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

    df = build_file_dataframe([fit_a, fit_b], str(tmp_path))

    assert "paired_file_path" in df.columns
    assert "overlap_duration" in df.columns
    assert "overlap_datapoints" in df.columns
    assert "pairing_error" in df.columns
    assert str(df["overlap_datapoints"].dtype) == "Int64"

    fit_rows = df[
        df["filename"].isin(["polar_pacer_20240101.fit", "h10_20240101.fit"])
    ]
    paired_paths = dict(zip(fit_rows["filename"], fit_rows["paired_file_path"]))
    assert paired_paths == {
        "polar_pacer_20240101.fit": "participant/session_1/h10_20240101.fit",
        "h10_20240101.fit": "participant/session_1/polar_pacer_20240101.fit",
    }
    assert fit_rows["overlap_duration"].tolist() == ["00:10:00", "00:10:00"]
    assert fit_rows["overlap_datapoints"].tolist() == [601, 601]
    assert fit_rows["pairing_error"].isna().all()


def test_build_file_dataframe_pairs_by_folder_and_date_and_extracts_activity_tags(
    monkeypatch, tmp_path
):
    session_dir = tmp_path / "nxg129"
    session_dir.mkdir()
    files = [
        str(session_dir / "20260617_5ZglABAW_polar_pacer.fit"),
        str(session_dir / "nxg129_eqlifemonitor_fam_20260617.csv"),
        str(session_dir / "20260624_PxdEvGDQ_polar_pacer.fit"),
        str(session_dir / "nxg129_eqlifemonitor_run_20260624.csv"),
        str(session_dir / "20260629_0LYbDk2K_polar_pacer.fit"),
        str(session_dir / "nxg129_eqlifemonitor_rm_20260629.csv"),
    ]
    monkeypatch.setattr(
        "local_catalogue_processing.local_catalogue_update.build_folder_fit_overlap_metrics",
        lambda files, role_by_path: {},
    )

    df = build_file_dataframe(files, str(tmp_path))
    rows = df.set_index("filename")

    assert rows.loc[
        "20260617_5ZglABAW_polar_pacer.fit", "paired_file_path"
    ] == "nxg129/nxg129_eqlifemonitor_fam_20260617.csv"
    assert rows.loc[
        "nxg129_eqlifemonitor_fam_20260617.csv", "paired_file_path"
    ] == "nxg129/20260617_5ZglABAW_polar_pacer.fit"
    assert "fam" in rows.loc[
        "nxg129_eqlifemonitor_fam_20260617.csv", "tags"
    ].split("|")
    assert "run" in rows.loc[
        "nxg129_eqlifemonitor_run_20260624.csv", "tags"
    ].split("|")
    assert "rm" in rows.loc[
        "nxg129_eqlifemonitor_rm_20260629.csv", "tags"
    ].split("|")
    assert df["overlap_duration"].isna().all()
    assert df["overlap_datapoints"].isna().all()
    assert df["pairing_error"].isna().all()


@pytest.mark.parametrize(
    "filenames",
    [
        ["polar_pacer_20240101.fit"],
        [
            "polar_pacer_20240101.fit",
            "h10_20240101.fit",
            "eq02_20240101.csv",
        ],
        ["polar_pacer_a_20240101.fit", "polar_pacer_b_20240101.fit"],
        ["h10_20240101.fit", "eq02_20240101.csv"],
        ["pacer_export.csv", "eq02_export.csv"],
    ],
)
def test_build_file_dataframe_includes_invalid_groups_with_pairing_errors(
    monkeypatch, tmp_path, filenames
):
    files = [str(tmp_path / filename) for filename in filenames]
    monkeypatch.setattr(
        "local_catalogue_processing.local_catalogue_update.build_folder_fit_overlap_metrics",
        lambda files, role_by_path: {},
    )

    df = build_file_dataframe(files, str(tmp_path))

    assert len(df) == len(files)
    assert df["paired_file_path"].isna().all()
    assert df["pairing_error"].notna().all()
    assert df["pairing_error"].str.contains(
        "expected exactly 2 files with exactly 1 pacer", regex=False
    ).all()


def test_overlap_pair_with_same_date_leftover_is_annotated(monkeypatch, tmp_path):
    pacer = str(tmp_path / "polar_pacer_20240101.fit")
    h10 = str(tmp_path / "h10_20240101.fit")
    eq02 = str(tmp_path / "eq02_20240101.csv")
    monkeypatch.setattr(
        "local_catalogue_processing.local_catalogue_update.build_folder_fit_overlap_metrics",
        lambda files, role_by_path: {
            pacer: {"paired_file_path": h10},
            h10: {"paired_file_path": pacer},
        },
    )

    df = build_file_dataframe([pacer, h10, eq02], str(tmp_path)).set_index("filename")

    assert df.loc["polar_pacer_20240101.fit", "pairing_error"] is None
    assert df.loc["h10_20240101.fit", "pairing_error"] is None
    assert "files=1" in df.loc["eq02_20240101.csv", "pairing_error"]


def test_unknown_fit_keeps_pacer_fallback_and_cannot_pair_with_pacer(
    monkeypatch, tmp_path
):
    files = [
        str(tmp_path / "polar_pacer_20240101.fit"),
        str(tmp_path / "unknown_watch_20240101.fit"),
    ]
    monkeypatch.setattr(
        "local_catalogue_processing.local_catalogue_update.build_folder_fit_overlap_metrics",
        lambda files, role_by_path: {},
    )

    df = build_file_dataframe(files, str(tmp_path))

    assert df["pairing_error"].str.contains("pacers=2", regex=False).all()
    assert df["pairing_error"].str.contains(
        "unknown_watch_20240101.fit [tags=pacer]", regex=False
    ).all()


def test_date_pairing_never_crosses_folders(monkeypatch, tmp_path):
    files = [
        str(tmp_path / "a" / "polar_pacer_20240101.fit"),
        str(tmp_path / "b" / "h10_20240101.fit"),
    ]
    monkeypatch.setattr(
        "local_catalogue_processing.local_catalogue_update.build_folder_fit_overlap_metrics",
        lambda files, role_by_path: {},
    )

    df = build_file_dataframe(files, str(tmp_path))

    assert df["paired_file_path"].isna().all()
    assert df["pairing_error"].notna().all()
    assert str(tmp_path / "a") in "\n".join(df["pairing_error"])
    assert str(tmp_path / "b") in "\n".join(df["pairing_error"])


def test_filename_rm_tag_requires_token_boundary(monkeypatch, tmp_path):
    files = [
        str(tmp_path / "polar_pacer_20240101.fit"),
        str(tmp_path / "eq02_armband_20240101.csv"),
    ]
    monkeypatch.setattr(
        "local_catalogue_processing.local_catalogue_update.build_folder_fit_overlap_metrics",
        lambda files, role_by_path: {},
    )

    df = build_file_dataframe(files, str(tmp_path))
    eq02_tags = df.loc[df["filename"].str.contains("armband"), "tags"].iloc[0]
    assert "rm" not in eq02_tags.split("|")
