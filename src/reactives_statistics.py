"""Statistics and table reactive functions for the FIT file comparison app."""

import logging

import pandas as pd
from shiny import Inputs, reactive, render, ui

from src.utils import (
    calculate_basic_stats,
    generate_llm_summary_stream,
    get_accuracy_stats,
    get_agreement_stats,
    get_bias_stats,
    get_file_information,
    get_raw_data_sample,
)

logger = logging.getLogger(__name__)


def create_statistics_reactives(
    inputs: Inputs, file_reactives: dict, data_reactives: dict
):
    """Create statistics and table reactive functions."""

    def _get_stats():
        aligned_data = data_reactives["_get_data_by_selected_range"]()
        if aligned_data is None:
            return pd.DataFrame()
        return calculate_basic_stats(
            aligned_data, data_reactives["_get_comparison_metric"]()
        )

    def _get_bias_stats():
        aligned_data = data_reactives["_get_data_by_selected_range"]()
        if aligned_data is None:
            return pd.DataFrame()
        return get_bias_stats(aligned_data, data_reactives["_get_comparison_metric"]())

    def _get_accuracy_stats():
        aligned_data = data_reactives["_get_data_by_selected_range"]()
        if aligned_data is None:
            return pd.DataFrame()
        return get_accuracy_stats(
            aligned_data, data_reactives["_get_comparison_metric"]()
        )

    def _get_agreement_stats():
        aligned_data = data_reactives["_get_data_by_selected_range"]()
        if aligned_data is None:
            return pd.DataFrame()
        return get_agreement_stats(
            aligned_data, data_reactives["_get_comparison_metric"]()
        )

    @render.data_frame
    def basicStatsTable():
        return data_reactives["_safe_execute"](
            _get_stats, "basicStatsTable", pd.DataFrame()
        )

    @render.data_frame
    def biasTable():
        return data_reactives["_safe_execute"](
            _get_bias_stats, "biasTable", pd.DataFrame()
        )

    @render.data_frame
    def accuracyTable():
        return data_reactives["_safe_execute"](
            _get_accuracy_stats, "accuracyTable", pd.DataFrame()
        )

    @render.data_frame
    def agreementTable():
        return data_reactives["_safe_execute"](
            _get_agreement_stats, "agreementTable", pd.DataFrame()
        )

    md = ui.MarkdownStream("streamOutput")

    @reactive.effect
    @reactive.event(inputs.llm_summary_regen)
    async def llm_summary_effect():
        try:
            metric = data_reactives["_get_comparison_metric"]()
            await md.stream(
                generate_llm_summary_stream(
                    metric=metric,
                    bias_stats=_get_bias_stats(),
                    accuracy_stats=_get_accuracy_stats(),
                    agreement_stats=_get_agreement_stats(),
                )
            )
        except Exception as e:
            logger.error("Error generating LLM summary: %s", e, exc_info=True)

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
        "_get_accuracy_stats": _get_accuracy_stats,
        "_get_agreement_stats": _get_agreement_stats,
        "basicStatsTable": basicStatsTable,
        "biasTable": biasTable,
        "accuracyTable": accuracyTable,
        "agreementTable": agreementTable,
        "llm_summary_effect": llm_summary_effect,
        "fileInfoTable": fileInfoTable,
        "rawDataTable": rawDataTable,
    }
