import logging

import shinyswatch
from shiny import Inputs, Outputs, Session, reactive

from src.reactives_data_processing import create_data_processing_reactives
from src.reactives_file_handling import create_file_handling_reactives
from src.reactives_statistics import create_statistics_reactives
from src.reactives_ui import create_ui_reactives
from src.reactives_visualizations import create_visualization_reactives

logger = logging.getLogger(__name__)


def server(input: Inputs, output: Outputs, session: Session):
    # Create reactive groups
    file_reactives = create_file_handling_reactives(input, session)
    data_reactives = create_data_processing_reactives(input, file_reactives)
    ui_reactives = create_ui_reactives(input, file_reactives, data_reactives)
    visualization_reactives = create_visualization_reactives(input, data_reactives)
    statistics_reactives = create_statistics_reactives(
        input, file_reactives, data_reactives
    )

    # Enable dynamic theme switching
    shinyswatch.theme_picker_server()

    @reactive.Effect
    @reactive.event(input.logout)
    async def _():
        await session.send_custom_message("logout", {})

    # Register UI reactives
    output.mainContent = ui_reactives["mainContent"]
    output.testFileSelector = ui_reactives["testFileSelector"]
    output.refFileSelector = ui_reactives["refFileSelector"]
    output.comparisonMetricSelector = ui_reactives["comparisonMetricSelector"]
    output.outlierRemovalSelector = ui_reactives["outlierRemovalSelector"]

    # Register visualization reactives
    output.metricPlot = visualization_reactives["metricPlot"]
    output.errorHistogramPlot = visualization_reactives["errorHistogramPlot"]
    output.blandAltmanPlot = visualization_reactives["blandAltmanPlot"]
    output.rollingErrorPlot = visualization_reactives["rollingErrorPlot"]

    # Register statistics reactives
    output.basicStatsTable = statistics_reactives["basicStatsTable"]
    output.biasAgreementTable = statistics_reactives["biasAgreementTable"]
    output.errorMagnitudeTable = statistics_reactives["errorMagnitudeTable"]
    output.correlationTable = statistics_reactives["correlationTable"]
    output.rawDataTable = statistics_reactives["rawDataTable"]
