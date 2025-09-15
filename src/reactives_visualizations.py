"""Visualization reactive functions for the FIT file comparison app."""

import logging

# Plotly FigureWidget is needed for interactive callbacks (click/selection)
import plotly.graph_objects as go
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
            fig = create_metric_plot(
                aligned_data, data_reactives["_get_comparison_metric"]()
            )
            if fig is None:
                return None

            # Convert to interactive FigureWidget so callbacks fire in Shiny
            fw = go.FigureWidget(fig)

            def _on_relayout_callback(change):
                """
                This function is triggered by the observe() method
                when the layout changes, such as from a zoom event.
                """
                relayout_data = change.new

                # Add a check to ensure relayout_data is not None before proceeding
                if relayout_data is None:
                    logger.info("Relayout data is None, skipping callback.")
                    return

                logger.info(
                    "Relayout event received with keys: %s", list(relayout_data.keys())
                )

                # Access the current figure layout to get actual ranges
                # This is useful when the relayout_data doesn't contain explicit ranges
                try:
                    current_xaxis = fw.layout.xaxis
                    current_yaxis = fw.layout.yaxis

                    if hasattr(current_xaxis, "range") and current_xaxis.range:
                        logger.info(
                            "Current X-axis range from figure: %s", current_xaxis.range
                        )

                    if hasattr(current_yaxis, "range") and current_yaxis.range:
                        logger.info(
                            "Current Y-axis range from figure: %s", current_yaxis.range
                        )

                except Exception as e:
                    logger.debug("Could not access current figure ranges: %s", e)

            # Use the observe() method to capture changes to the internal
            # _js2py_layoutDelta property, which holds relayout data from the browser.
            fw.observe(_on_relayout_callback, names="_js2py_layoutDelta")

            return fw

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
