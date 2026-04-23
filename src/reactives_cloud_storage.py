"""Cloud Storage reactive functions for manifest-driven S3 pair analysis."""

import os

import pandas as pd
import plotly.graph_objects as go
from shiny import Inputs, reactive, render, ui
from shiny.types import SilentException
from shinywidgets import output_widget, render_widget

from src.utils import (
    get_cached_cloud_pair_summary,
    get_cloud_cache_db_path,
    init_cloud_cache,
    put_cached_cloud_pair_summary,
    read_catalogue,
)
from src.utils.cloud_analysis import (
    align_pair_data,
    build_cloud_pair_results,
    compute_pair_summary,
    load_cached_common_metrics,
    load_cloud_pair_data,
)
from src.utils.cloud_manifest import (
    PAIR_MANIFEST_COLUMNS,
    PAIR_SELECTION_TABLE_COLUMNS,
    build_cloud_pair_manifest,
    build_cloud_pair_selection_table,
    coerce_input_date,
    filter_cloud_pair_manifest,
    get_cloud_manifest_date_bounds,
    get_cloud_manifest_groups,
    get_selected_cloud_pair_ids,
)
from src.utils.cloud_plots import RANGE_PLOT_SPECS, create_cloud_metric_range_plot


def create_cloud_storage_reactives(
    inputs: Inputs, session=None, local_pair_override=None
):
    """Create Cloud Storage reactive functions."""

    bucket_name = os.getenv("S3_BUCKET")
    aws_profile = os.getenv("AWS_PROFILE")
    manifest_key = os.getenv("CATALOGUE_CSV_KEY")
    cloud_cache_db_path = get_cloud_cache_db_path()
    init_cloud_cache(cloud_cache_db_path)
    cloud_analysis_request = reactive.Value(None)
    selected_cloud_plot_pair_id = reactive.Value(None)
    date_filter_state = {"group_signature": None}

    def _safe_input(input_name, default=None):
        try:
            input_obj = getattr(inputs, input_name, None)
            if input_obj is None:
                return default
            return input_obj()
        except SilentException:
            return default

    @reactive.Calc
    def _cloud_manifest():
        manifest_df = read_catalogue()
        if not isinstance(manifest_df, pd.DataFrame):
            return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)
        return build_cloud_pair_manifest(manifest_df)

    @render.ui
    def cloudGroupSelector():
        pair_df = _cloud_manifest()
        if pair_df.empty:
            return None

        choices = get_cloud_manifest_groups(pair_df)
        return ui.input_selectize(
            "selected_cloud_groups",
            "Filter groups:",
            choices=choices,
            selected=choices,
            multiple=True,
        )

    def _selected_cloud_groups():
        selected_groups = _safe_input("selected_cloud_groups")
        if selected_groups is None:
            return get_cloud_manifest_groups(_cloud_manifest())
        return selected_groups

    def _group_signature(selected_groups) -> tuple[str, ...]:
        return tuple(sorted(str(group) for group in selected_groups if group))

    @render.ui
    def cloudDateRangeSelector():
        selected_groups = _selected_cloud_groups()
        group_signature = _group_signature(selected_groups)
        group_filtered_df = filter_cloud_pair_manifest(
            _cloud_manifest(),
            selected_groups=selected_groups,
        )
        start_date, end_date = get_cloud_manifest_date_bounds(group_filtered_df)
        if start_date is None or end_date is None:
            return None

        selected_range = _safe_input("selected_cloud_date_range")
        selected_start = start_date
        selected_end = end_date
        if date_filter_state["group_signature"] != group_signature:
            date_filter_state["group_signature"] = group_signature
        elif isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
            selected_start = coerce_input_date(selected_range[0]) or start_date
            selected_end = coerce_input_date(selected_range[1]) or end_date
            selected_start = min(max(selected_start, start_date), end_date)
            selected_end = min(max(selected_end, start_date), end_date)
            if selected_start > selected_end:
                selected_start = start_date
                selected_end = end_date

        return ui.input_date_range(
            "selected_cloud_date_range",
            "Filter date range:",
            start=selected_start,
            end=selected_end,
            min=start_date,
            max=end_date,
        )

    @reactive.Calc
    def _filtered_cloud_manifest():
        selected_groups = _selected_cloud_groups()
        group_signature = _group_signature(selected_groups)
        selected_range = _safe_input("selected_cloud_date_range")
        start_date = None
        end_date = None
        if (
            date_filter_state["group_signature"] == group_signature
            and isinstance(selected_range, (list, tuple))
            and len(selected_range) == 2
        ):
            start_date = selected_range[0]
            end_date = selected_range[1]
        return filter_cloud_pair_manifest(
            _cloud_manifest(),
            selected_groups=selected_groups,
            start_date=start_date,
            end_date=end_date,
        )

    @render.data_frame
    def cloudPairSelectionTable():
        pair_df = _filtered_cloud_manifest()
        if pair_df.empty:
            return pd.DataFrame(columns=PAIR_SELECTION_TABLE_COLUMNS)

        return render.DataGrid(
            build_cloud_pair_selection_table(pair_df),
            selection_mode="rows",
        )

    @reactive.Calc
    def _selected_cloud_pair_ids():
        return get_selected_cloud_pair_ids(
            _filtered_cloud_manifest(),
            _safe_input("cloudPairSelectionTable_selected_rows", []),
        )

    @reactive.Effect
    @reactive.event(inputs.cloudRefreshAnalysis)
    def _capture_cloud_analysis_request():
        cloud_analysis_request.set(
            {
                "pair_df": _filtered_cloud_manifest().copy(),
                "selected_pair_ids": _selected_cloud_pair_ids(),
                "metric": _safe_input("cloud_comparison_metric", "heart_rate"),
                "auto_shift_method": _safe_input(
                    "cloud_auto_shift_method", "Minimize MAE"
                ),
            }
        )

    @reactive.Calc
    def _cloud_analysis_request():
        return cloud_analysis_request.get()

    def _build_local_pair_override(pair_id: str, request: dict):
        pair_data = load_cloud_pair_data(
            request["pair_df"],
            [pair_id],
            bucket_name,
            aws_profile,
        )
        entry = pair_data.get(pair_id)
        if not entry or entry.get("error"):
            return None

        return {
            "pair_id": pair_id,
            "metric": request["metric"],
            "auto_shift_method": request["auto_shift_method"],
            "test_filename": entry["meta"]["test_filename"],
            "ref_filename": entry["meta"]["ref_filename"],
            "test_df": entry["test_df"].copy(),
            "ref_df": entry["ref_df"].copy(),
        }

    @reactive.Calc
    def _selected_cloud_pair_data():
        request = _cloud_analysis_request()
        if request is None:
            return {}
        return load_cloud_pair_data(
            request["pair_df"],
            request["selected_pair_ids"],
            bucket_name,
            aws_profile,
        )

    @reactive.Calc
    def _cloud_common_metrics():
        return load_cached_common_metrics(
            _filtered_cloud_manifest(),
            _selected_cloud_pair_ids(),
            cloud_cache_db_path,
        )

    @render.ui
    def cloudMetricSelector():
        choices = _cloud_common_metrics()
        if not choices:
            choices = ["heart_rate"]

        selected = "heart_rate" if "heart_rate" in choices else choices[0]
        return ui.input_select(
            "cloud_comparison_metric",
            "Select comparison metric:",
            choices=choices,
            selected=selected,
        )

    @render.ui
    def cloudAutoShiftSelector():
        return ui.input_select(
            "cloud_auto_shift_method",
            "Auto-shift by:",
            choices=[
                "None (manual)",
                "Minimize MAE",
                "Minimize MSE",
                "Maximize Concordance Correlation",
                "Maximize Pearson Correlation",
            ],
            selected="Minimize MAE",
        )

    @reactive.Calc
    def _cloud_pair_results():
        request = _cloud_analysis_request()
        if request is None:
            return pd.DataFrame()
        pair_df = request["pair_df"]
        selected_pair_ids = request["selected_pair_ids"]
        metric = request["metric"]
        auto_shift_method = request["auto_shift_method"]

        if pair_df.empty or not selected_pair_ids or not metric:
            return pd.DataFrame()

        selected_pairs_df = pair_df[pair_df["pair_id"].isin(selected_pair_ids)].copy()
        if selected_pairs_df.empty:
            return pd.DataFrame()

        cached_rows_by_pair_id = {}
        missing_pair_ids = []
        for row in selected_pairs_df.itertuples(index=False):
            cached_row = get_cached_cloud_pair_summary(
                test_etag=row.test_etag,
                ref_etag=row.ref_etag,
                metric=metric,
                auto_shift_method=auto_shift_method,
                db_path=cloud_cache_db_path,
            )
            if cached_row is None:
                missing_pair_ids.append(row.pair_id)
            else:
                cached_row = {
                    **cached_row,
                    "pair_id": row.pair_id,
                    "Group": row.group,
                    "Date": row.date,
                    "Test File": row.test_filename,
                    "Ref File": row.ref_filename,
                }
                cached_rows_by_pair_id[row.pair_id] = cached_row

        computed_rows_by_pair_id = {}
        if missing_pair_ids:
            missing_pair_data = load_cloud_pair_data(
                pair_df,
                missing_pair_ids,
                bucket_name,
                aws_profile,
            )
            for pair_id, entry in missing_pair_data.items():
                result_row = build_cloud_pair_result_row(
                    pair_id,
                    entry,
                    metric,
                    auto_shift_method,
                )
                computed_rows_by_pair_id[pair_id] = result_row
                if result_row.get("Status") == "OK":
                    meta = entry["meta"]
                    put_cached_cloud_pair_summary(
                        test_etag=meta["test_etag"],
                        ref_etag=meta["ref_etag"],
                        metric=metric,
                        auto_shift_method=auto_shift_method,
                        result_row=result_row,
                        common_metrics=entry.get("common_metrics"),
                        db_path=cloud_cache_db_path,
                    )

        ordered_rows = []
        for row in selected_pairs_df.itertuples(index=False):
            if row.pair_id in cached_rows_by_pair_id:
                ordered_rows.append(cached_rows_by_pair_id[row.pair_id])
            elif row.pair_id in computed_rows_by_pair_id:
                ordered_rows.append(computed_rows_by_pair_id[row.pair_id])

        return pd.DataFrame(ordered_rows)

    @reactive.Effect
    def _clear_selected_cloud_pair_if_missing():
        results_df = _cloud_pair_results()
        current_pair_id = selected_cloud_plot_pair_id.get()
        if current_pair_id is None:
            return
        if (
            results_df.empty
            or current_pair_id not in results_df.get("pair_id", []).tolist()
        ):
            selected_cloud_plot_pair_id.set(None)

    @render.data_frame
    def cloudPairSummaryTable():
        result_df = _cloud_pair_results()
        if result_df.empty:
            return pd.DataFrame()
        return render.DataGrid(
            result_df.drop(columns=["pair_id"], errors="ignore"),
            selection_mode="none",
        )

    @render.ui
    def cloudMetricRangePlotGrid():
        cards = []
        for spec in RANGE_PLOT_SPECS:
            cards.append(
                ui.card(
                    ui.card_header(spec["card_title"]),
                    output_widget(
                        spec["output_id"],
                        height="150px",
                        fill=True,
                    ),
                )
            )
        return ui.layout_columns(*cards, col_widths=[6] * len(cards))

    def _make_cloud_metric_plot_renderer(
        output_id: str,
        metric_name: str,
        benchmark_indicator: float | None = None,
    ):
        def _plot():
            results_df = _cloud_pair_results()
            request = _cloud_analysis_request()
            figure = create_cloud_metric_range_plot(
                results_df,
                metric_name,
                benchmark_indicator=benchmark_indicator,
                theme_settings=_safe_input("plotly_theme"),
                selected_pair_id=selected_cloud_plot_pair_id.get(),
            )

            figure_widget = go.FigureWidget(figure)
            if "Status" not in results_df.columns:
                plot_df = pd.DataFrame()
            else:
                plot_df = results_df[results_df["Status"] == "OK"].reset_index(
                    drop=True
                )
            marker_trace = next(
                (
                    trace
                    for trace in figure_widget.data
                    if getattr(trace, "type", None) == "scatter"
                    and getattr(trace, "mode", None) == "markers"
                ),
                None,
            )

            if marker_trace is not None and not plot_df.empty and request is not None:

                def _handle_click(_trace, points, _selector):
                    if not points.point_inds:
                        return

                    point_index = points.point_inds[0]
                    if point_index >= len(plot_df):
                        return

                    pair_id = plot_df.iloc[point_index].get("pair_id")
                    if not pair_id:
                        return

                    selected_cloud_plot_pair_id.set(pair_id)
                    if local_pair_override is not None:
                        pair_override = _build_local_pair_override(pair_id, request)
                        if pair_override is not None:
                            local_pair_override.set(pair_override)
                    if session is not None:
                        ui.update_navs(
                            "mainNavset",
                            selected="Benchmarking",
                            session=session,
                        )

                marker_trace.on_click(_handle_click)

            return figure_widget

        _plot.__name__ = output_id
        return render_widget(_plot)

    cloud_metric_plot_renderers = {
        spec["output_id"]: _make_cloud_metric_plot_renderer(
            spec["output_id"],
            spec["metric_name"],
            benchmark_indicator=spec["benchmark_indicator"],
        )
        for spec in RANGE_PLOT_SPECS
    }

    return {
        "_cloud_manifest": _cloud_manifest,
        "_filtered_cloud_manifest": _filtered_cloud_manifest,
        "_cloud_analysis_request": _cloud_analysis_request,
        "_selected_cloud_pair_ids": _selected_cloud_pair_ids,
        "_selected_cloud_pair_data": _selected_cloud_pair_data,
        "_cloud_common_metrics": _cloud_common_metrics,
        "_cloud_pair_results": _cloud_pair_results,
        "cloudGroupSelector": cloudGroupSelector,
        "cloudDateRangeSelector": cloudDateRangeSelector,
        "cloudPairSelectionTable": cloudPairSelectionTable,
        "cloudMetricSelector": cloudMetricSelector,
        "cloudAutoShiftSelector": cloudAutoShiftSelector,
        "cloudMetricRangePlotGrid": cloudMetricRangePlotGrid,
        **cloud_metric_plot_renderers,
        "cloudPairSummaryTable": cloudPairSummaryTable,
    }
