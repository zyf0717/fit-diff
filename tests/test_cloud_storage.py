import pandas as pd

from src.reactives_cloud_storage import (
    align_pair_data,
    build_cloud_pair_manifest,
    compute_pair_summary,
)


def test_build_cloud_pair_manifest_returns_one_row_per_test_ref_pair():
    manifest_df = pd.DataFrame(
        [
            {
                "etag": "test-etag",
                "paired_etag": "ref-etag",
                "pairing_group": "yifei",
                "pair_index": 1,
                "date": "2025-08-04",
                "filename": "test.fit",
                "device_type": "test",
                "s3_key": "fit_files/yifei/test.fit",
            },
            {
                "etag": "ref-etag",
                "paired_etag": "test-etag",
                "pairing_group": "yifei",
                "pair_index": 1,
                "date": "2025-08-04",
                "filename": "ref.fit",
                "device_type": "ref",
                "s3_key": "fit_files/yifei/ref.fit",
            },
        ]
    )

    pair_df = build_cloud_pair_manifest(manifest_df)

    assert len(pair_df) == 1
    row = pair_df.iloc[0]
    assert row["test_etag"] == "test-etag"
    assert row["ref_etag"] == "ref-etag"
    assert row["test_s3_key"] == "fit_files/yifei/test.fit"
    assert row["ref_s3_key"] == "fit_files/yifei/ref.fit"
    assert "test.fit <> ref.fit" in row["pair_label"]


def test_align_pair_data_uses_existing_local_alignment_path():
    test_df = pd.DataFrame(
        {
            "timestamp": [100, 101, 102],
            "filename": ["test.fit", "test.fit", "test.fit"],
            "heart_rate": [150, 152, 154],
        }
    )
    ref_df = pd.DataFrame(
        {
            "timestamp": [101, 102, 103],
            "filename": ["ref.fit", "ref.fit", "ref.fit"],
            "heart_rate": [149, 151, 153],
        }
    )

    aligned_df = align_pair_data(test_df, ref_df, "heart_rate")

    assert aligned_df["timestamp"].tolist() == [101, 102]
    assert aligned_df["heart_rate_test"].tolist() == [152, 154]
    assert aligned_df["heart_rate_ref"].tolist() == [149, 151]


def test_compute_pair_summary_returns_expected_metrics():
    aligned_df = pd.DataFrame(
        {
            "heart_rate_test": [10.0, 14.0, 18.0],
            "heart_rate_ref": [8.0, 13.0, 20.0],
        }
    )

    summary = compute_pair_summary(aligned_df, "heart_rate")

    assert summary["Samples"] == 3
    assert summary["Mean Bias"] == 0.333
    assert summary["MAE"] == 1.667
    assert summary["MSE"] == 3.0
    assert summary["MAPE (%)"] == 14.231
    assert summary["CCC"] == 0.915
    assert summary["Pearson Corr"] == 0.995
    assert summary["LoA Lower"] == -3.747
    assert summary["LoA Upper"] == 4.413
