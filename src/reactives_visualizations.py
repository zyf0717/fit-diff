"""Visualization reactive functions for the FIT file comparison app."""

import logging

from shiny import Inputs
from shinywidgets import render_widget

from src.utils import (
    create_bland_altman_plot,
    create_error_histogram,
    create_metric_plot,
    create_rolling_error_plot,
)

logger = logging.getLogger(__name__)


def create_visualization_reactives(inputs: Inputs, data_reactives: dict):
    """Create visualization reactive functions."""

    @render_widget
    def metricPlot():
        def _create_plot():
            aligned_data = data_reactives["_get_aligned_data_with_outlier_removal"]()
            if aligned_data is None:
                return None
            return create_metric_plot(
                aligned_data, data_reactives["_get_comparison_metric"]()
            )

        return data_reactives["_safe_execute"](_create_plot, "metricPlot")

    @render_widget
    def errorHistogramPlot():
        def _create_histogram():
            aligned_data = data_reactives["_get_aligned_data_with_outlier_removal"]()
            if aligned_data is None:
                return None
            return create_error_histogram(
                aligned_data, data_reactives["_get_comparison_metric"]()
            )

        return data_reactives["_safe_execute"](_create_histogram, "errorHistogramPlot")

    @render_widget
    def blandAltmanPlot():
        def _create_bland_altman():
            aligned_data = data_reactives["_get_aligned_data_with_outlier_removal"]()
            if aligned_data is None:
                return None
            return create_bland_altman_plot(
                aligned_data, data_reactives["_get_comparison_metric"]()
            )

        return data_reactives["_safe_execute"](_create_bland_altman, "blandAltmanPlot")

    @render_widget
    def rollingErrorPlot():
        def _create_rolling_error():
            aligned_data = data_reactives["_get_aligned_data_with_outlier_removal"]()
            if aligned_data is None:
                return None
            return create_rolling_error_plot(
                aligned_data, data_reactives["_get_comparison_metric"]()
            )

        return data_reactives["_safe_execute"](
            _create_rolling_error, "rollingErrorPlot"
        )

    return {
        "metricPlot": metricPlot,
        "errorHistogramPlot": errorHistogramPlot,
        "blandAltmanPlot": blandAltmanPlot,
        "rollingErrorPlot": rollingErrorPlot,
    }
