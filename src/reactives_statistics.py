"""Statistics and table reactive functions for the FIT file comparison app."""

import logging

import pandas as pd
from shiny import Inputs, reactive, render, ui

from src.utils import (
    calculate_basic_stats,
    generate_llm_summary_stream,
    get_bias_agreement_stats,
    get_correlation_stats,
    get_error_magnitude_stats,
    get_file_information,
    get_raw_data_sample,
)

logger = logging.getLogger(__name__)


def create_statistics_reactives(
    inputs: Inputs, file_reactives: dict, data_reactives: dict
):
    """Create statistics and table reactive functions."""

    def _get_stats():
        aligned_data = data_reactives["_get_validated_aligned_data"]()
        if aligned_data is None:
            return pd.DataFrame()
        return calculate_basic_stats(
            aligned_data, data_reactives["_get_comparison_metric"]()
        )

    def _get_bias_stats():
        aligned_data = data_reactives["_get_validated_aligned_data"]()
        if aligned_data is None:
            return pd.DataFrame()
        return get_bias_agreement_stats(
            aligned_data, data_reactives["_get_comparison_metric"]()
        )

    def _get_error_stats():
        aligned_data = data_reactives["_get_validated_aligned_data"]()
        if aligned_data is None:
            return pd.DataFrame()
        return get_error_magnitude_stats(
            aligned_data, data_reactives["_get_comparison_metric"]()
        )

    def _get_correlation_stats():
        aligned_data = data_reactives["_get_validated_aligned_data"]()
        if aligned_data is None:
            return pd.DataFrame()
        return get_correlation_stats(
            aligned_data, data_reactives["_get_comparison_metric"]()
        )

    @render.data_frame
    def basicStatsTable():
        return data_reactives["_safe_execute"](
            _get_stats, "basicStatsTable", pd.DataFrame()
        )

    @render.data_frame
    def biasAgreementTable():
        return data_reactives["_safe_execute"](
            _get_bias_stats, "biasAgreementTable", pd.DataFrame()
        )

    @render.data_frame
    def errorMagnitudeTable():
        return data_reactives["_safe_execute"](
            _get_error_stats, "errorMagnitudeTable", pd.DataFrame()
        )

    @render.data_frame
    def correlationTable():
        return data_reactives["_safe_execute"](
            _get_correlation_stats, "correlationTable", pd.DataFrame()
        )

    md = ui.MarkdownStream("streamOutput")

    @reactive.effect
    @reactive.event(inputs.llm_summary_regen)
    async def llm_summary_effect():
        await md.stream(
            generate_llm_summary_stream(
                metric=inputs.comparison_metric(),
                bias_stats=_get_bias_stats(),
                error_stats=_get_error_stats(),
                correlation_stats=_get_correlation_stats(),
            )
        )

    @render.data_frame
    def fileInfoTable():
        # Use raw data for file information, not prepared/filtered data
        fit_data = file_reactives["_all_fit_data"]()
        if not fit_data or (
            isinstance(fit_data, tuple) and (fit_data[0].empty or fit_data[1].empty)
        ):
            return pd.DataFrame()
        test_data, ref_data = fit_data
        result = get_file_information(test_data, ref_data)
        if result is not None:
            return render.DataGrid(result, selection_mode="rows")
        return pd.DataFrame()

    @render.data_frame
    def rawDataTable():
        # Use raw data instead of prepared/filtered data
        fit_data = file_reactives["_all_fit_data"]()
        if not fit_data or (
            isinstance(fit_data, tuple) and (fit_data[0].empty or fit_data[1].empty)
        ):
            return pd.DataFrame()

        test_data, ref_data = fit_data

        sample_size = (
            inputs.raw_data_sample_size()
            if hasattr(inputs, "raw_data_sample_size")
            else 100
        )

        # Get selected rows from file information table
        selected_filenames = None
        try:
            # Check if fileInfoTable has selections
            file_info_selection = inputs.fileInfoTable_selected_rows()
            if file_info_selection:
                # Get the file information data to extract filenames
                file_info_df = get_file_information(test_data, ref_data)
                if file_info_df is not None and not file_info_df.empty:
                    # Extract filenames from selected rows (0-indexed)
                    selected_rows = [int(i) for i in file_info_selection]
                    selected_filenames = file_info_df.iloc[selected_rows][
                        "filename"
                    ].tolist()
        except (AttributeError, KeyError, IndexError):
            # If no selection or error accessing selection, show all files
            selected_filenames = None

        return get_raw_data_sample(test_data, ref_data, sample_size, selected_filenames)

    return {
        "_get_stats": _get_stats,
        "_get_bias_stats": _get_bias_stats,
        "_get_error_stats": _get_error_stats,
        "_get_correlation_stats": _get_correlation_stats,
        "basicStatsTable": basicStatsTable,
        "biasAgreementTable": biasAgreementTable,
        "errorMagnitudeTable": errorMagnitudeTable,
        "correlationTable": correlationTable,
        "llm_summary_effect": llm_summary_effect,
        "fileInfoTable": fileInfoTable,
        "rawDataTable": rawDataTable,
    }
