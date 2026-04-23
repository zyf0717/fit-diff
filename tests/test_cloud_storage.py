import pandas as pd
import plotly.graph_objects as go

from src.reactives_cloud_storage import (
    align_pair_data,
    build_cloud_pair_manifest,
    build_cloud_pair_results,
    build_cloud_pair_selection_table,
    compute_pair_summary,
    create_cloud_metric_range_plot,
    filter_cloud_pair_manifest,
    get_cloud_manifest_date_bounds,
    get_cloud_manifest_groups,
    get_selected_cloud_pair_ids,
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
                "paired_overlap_pct": 0.875,
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
    assert row["paired_overlap_pct"] == 0.875
    assert "test.fit <> ref.fit" in row["pair_label"]


def test_get_cloud_manifest_groups_returns_sorted_unique_groups():
    pair_df = pd.DataFrame(
        [
            {"pairing_group": "zoe"},
            {"pairing_group": "amy"},
            {"pairing_group": "zoe"},
            {"pairing_group": None},
        ]
    )

    groups = get_cloud_manifest_groups(pair_df)

    assert groups == ["amy", "zoe"]


def test_filter_cloud_pair_manifest_filters_by_group_and_date_range():
    pair_df = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "pair_label": "yifei 2025-08-01",
                "pairing_group": "yifei",
                "pair_index": 1,
                "date": "2025-08-01",
                "test_etag": "t1",
                "test_filename": "t1.fit",
                "test_s3_key": "fit_files/yifei/t1.fit",
                "ref_etag": "r1",
                "ref_filename": "r1.fit",
                "ref_s3_key": "fit_files/yifei/r1.fit",
            },
            {
                "pair_id": "p2",
                "pair_label": "yifei 2025-08-03",
                "pairing_group": "yifei",
                "pair_index": 2,
                "date": "2025-08-03",
                "test_etag": "t2",
                "test_filename": "t2.fit",
                "test_s3_key": "fit_files/yifei/t2.fit",
                "ref_etag": "r2",
                "ref_filename": "r2.fit",
                "ref_s3_key": "fit_files/yifei/r2.fit",
            },
            {
                "pair_id": "p3",
                "pair_label": "amy 2025-08-02",
                "pairing_group": "amy",
                "pair_index": 1,
                "date": "2025-08-02",
                "test_etag": "t3",
                "test_filename": "t3.fit",
                "test_s3_key": "fit_files/amy/t3.fit",
                "ref_etag": "r3",
                "ref_filename": "r3.fit",
                "ref_s3_key": "fit_files/amy/r3.fit",
            },
        ]
    )

    filtered_df = filter_cloud_pair_manifest(
        pair_df,
        selected_groups=["yifei"],
        start_date="2025-08-02",
        end_date="2025-08-03",
    )

    assert filtered_df["pair_id"].tolist() == ["p2"]


def test_filter_cloud_pair_manifest_returns_empty_when_groups_explicitly_cleared():
    pair_df = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "pair_label": "yifei 2025-08-01",
                "pairing_group": "yifei",
                "pair_index": 1,
                "date": "2025-08-01",
                "test_etag": "t1",
                "test_filename": "t1.fit",
                "test_s3_key": "fit_files/yifei/t1.fit",
                "ref_etag": "r1",
                "ref_filename": "r1.fit",
                "ref_s3_key": "fit_files/yifei/r1.fit",
            }
        ]
    )

    filtered_df = filter_cloud_pair_manifest(pair_df, selected_groups=[])

    assert filtered_df.empty


def test_get_cloud_manifest_date_bounds_returns_filtered_min_max_dates():
    pair_df = pd.DataFrame(
        [
            {"date": "2025-08-04"},
            {"date": "2025-08-01"},
            {"date": "2025-08-03"},
        ]
    )

    start_date, end_date = get_cloud_manifest_date_bounds(pair_df)

    assert start_date.isoformat() == "2025-08-01"
    assert end_date.isoformat() == "2025-08-04"


def test_build_cloud_pair_selection_table_returns_sortable_display_columns():
    pair_df = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "pairing_group": "yifei",
                "date": "2025-08-01",
                "paired_overlap_pct": 0.723456,
                "test_filename": "test.fit",
                "ref_filename": "ref.fit",
            }
        ]
    )

    selection_df = build_cloud_pair_selection_table(pair_df)

    assert selection_df.columns.tolist() == [
        "Group",
        "Date",
        "Test File",
        "Ref File",
        "Overlap (%)",
    ]
    assert selection_df.iloc[0].to_dict() == {
        "Group": "yifei",
        "Date": "2025-08-01",
        "Test File": "test.fit",
        "Ref File": "ref.fit",
        "Overlap (%)": 72.35,
    }


def test_get_selected_cloud_pair_ids_maps_selected_rows_to_pair_ids():
    pair_df = pd.DataFrame(
        [
            {"pair_id": "p1"},
            {"pair_id": "p2"},
            {"pair_id": "p3"},
        ]
    )

    selected_pair_ids = get_selected_cloud_pair_ids(pair_df, [2, "0", 9, "bad"])

    assert selected_pair_ids == ["p3", "p1"]


def test_build_cloud_pair_results_returns_summary_rows():
    pair_data = {
        "p1": {
            "meta": {
                "pairing_group": "yifei",
                "date": "2025-08-01",
                "test_filename": "test.fit",
                "ref_filename": "ref.fit",
            },
            "test_df": pd.DataFrame(
                {
                    "timestamp": [100, 101, 102],
                    "filename": ["test.fit", "test.fit", "test.fit"],
                    "heart_rate": [150, 152, 154],
                }
            ),
            "ref_df": pd.DataFrame(
                {
                    "timestamp": [100, 101, 102],
                    "filename": ["ref.fit", "ref.fit", "ref.fit"],
                    "heart_rate": [149, 151, 153],
                }
            ),
            "common_metrics": ["heart_rate"],
        }
    }

    result_df = build_cloud_pair_results(pair_data, "heart_rate", "Minimize MAE")

    assert result_df.iloc[0]["Status"] == "OK"
    assert result_df.iloc[0]["Metric"] == "heart_rate"
    assert result_df.iloc[0]["Mean Bias"] == 1.0
    assert result_df.iloc[0]["MAE"] == 1.0


def test_create_cloud_metric_range_plot_returns_plotly_figure():
    results_df = pd.DataFrame(
        [
            {
                "Group": "yifei",
                "Date": "2025-08-01",
                "Test File": "test.fit",
                "Ref File": "ref.fit",
                "Status": "OK",
                "Mean Bias": 1.0,
                "MAE": 1.0,
                "MSE": 1.0,
                "MAPE (%)": 0.5,
                "CCC": 0.98,
                "Pearson Corr": 0.99,
                "LoA Lower": -2.0,
                "LoA Upper": 4.0,
            }
        ]
    )

    figure = create_cloud_metric_range_plot(results_df, "Mean Bias")

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 3
    assert figure.data[1].type == "box"
    assert figure.data[1].fillcolor == "rgba(33, 37, 41, 0.14)"
    assert figure.data[1].whiskerwidth == 0


def test_create_cloud_metric_range_plot_empty_state_uses_theme_and_height():
    figure = create_cloud_metric_range_plot(
        pd.DataFrame(),
        "Mean Bias",
        theme_settings={
            "mode": "dark",
            "font_color": "#f8f9fa",
        },
    )

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 0
    assert figure.layout.height == 150
    assert figure.layout.font.color == "#f8f9fa"
    assert figure.layout.xaxis.visible is False
    assert figure.layout.yaxis.visible is False


def test_create_cloud_metric_range_plot_applies_dark_theme_colors():
    results_df = pd.DataFrame(
        [
            {
                "Group": "yifei",
                "Date": "2025-08-01",
                "Test File": "test.fit",
                "Ref File": "ref.fit",
                "Status": "OK",
                "Mean Bias": 1.0,
            }
        ]
    )

    figure = create_cloud_metric_range_plot(
        results_df,
        "Mean Bias",
        theme_settings={
            "mode": "dark",
            "font_color": "#f8f9fa",
            "muted_color": "rgba(248, 249, 250, 0.35)",
            "box_fill_color": "rgba(248, 249, 250, 0.22)",
            "box_line_color": "rgba(248, 249, 250, 0.42)",
        },
    )

    assert figure.layout.font.color == "#f8f9fa"
    assert figure.data[0].line.color == "rgba(248, 249, 250, 0.35)"
    assert figure.data[1].fillcolor == "rgba(248, 249, 250, 0.22)"


def test_create_cloud_metric_range_plot_highlights_selected_pair():
    results_df = pd.DataFrame(
        [
            {
                "pair_id": "pair-1",
                "Group": "yifei",
                "Date": "2025-08-01",
                "Test File": "test-a.fit",
                "Ref File": "ref-a.fit",
                "Status": "OK",
                "Mean Bias": 1.0,
            },
            {
                "pair_id": "pair-2",
                "Group": "yifei",
                "Date": "2025-08-02",
                "Test File": "test-b.fit",
                "Ref File": "ref-b.fit",
                "Status": "OK",
                "Mean Bias": 2.0,
            },
        ]
    )

    figure = create_cloud_metric_range_plot(
        results_df,
        "Mean Bias",
        selected_pair_id="pair-2",
    )

    marker_trace = figure.data[2]
    assert list(marker_trace.marker.color) == ["#2c7fb8", "#084298"]
    assert list(marker_trace.marker.size) == [10, 12]


def test_create_cloud_metric_range_plot_adds_benchmark_line_when_configured():
    results_df = pd.DataFrame(
        [
            {
                "Group": "yifei",
                "Date": "2025-08-01",
                "Test File": "test.fit",
                "Ref File": "ref.fit",
                "Status": "OK",
                "Mean Bias": 1.0,
            }
        ]
    )

    figure = create_cloud_metric_range_plot(
        results_df,
        "Mean Bias",
        benchmark_indicator=10.0,
    )

    assert len(figure.data) == 4
    benchmark_trace = figure.data[3]
    assert list(benchmark_trace.x) == [10.0, 10.0]
    assert benchmark_trace.line.color == "#d62728"
    assert benchmark_trace.line.dash == "dot"


def test_create_cloud_metric_range_plot_pads_axis_when_benchmark_hits_edge():
    results_df = pd.DataFrame(
        [
            {
                "Group": "yifei",
                "Date": "2025-08-01",
                "Test File": "test.fit",
                "Ref File": "ref.fit",
                "Status": "OK",
                "Mean Bias": 1.0,
            }
        ]
    )

    figure = create_cloud_metric_range_plot(
        results_df,
        "Mean Bias",
        benchmark_indicator=10.0,
    )

    axis_trace = figure.data[0]
    assert axis_trace.x[1] > 10.0


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

    aligned_df, applied_shift = align_pair_data(test_df, ref_df, "heart_rate")

    assert applied_shift == 0
    assert aligned_df["timestamp"].tolist() == [101, 102]
    assert aligned_df["heart_rate_test"].tolist() == [152, 154]
    assert aligned_df["heart_rate_ref"].tolist() == [149, 151]


def test_align_pair_data_applies_auto_shift_before_alignment():
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
            "heart_rate": [150, 152, 154],
        }
    )

    aligned_df, applied_shift = align_pair_data(
        test_df, ref_df, "heart_rate", auto_shift_method="Minimize MAE"
    )

    assert applied_shift == 1.0
    assert aligned_df["timestamp"].tolist() == [101, 102, 103]
    assert aligned_df["heart_rate_test"].tolist() == [150, 152, 154]
    assert aligned_df["heart_rate_ref"].tolist() == [150, 152, 154]


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
