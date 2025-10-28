import logging

import shinyswatch
from shiny import Inputs, Outputs, Session, reactive

from src.reactives_data_processing import create_data_processing_reactives
from src.reactives_file_handling import create_file_handling_reactives
from src.reactives_statistics import create_statistics_reactives
from src.reactives_ui import create_ui_reactives
from src.reactives_visualizations import create_visualization_reactives

logger = logging.getLogger(__name__)


def server(inputs: Inputs, output: Outputs, session: Session):
    # Create shared/cross-module reactive values
    metric_plot_x_range = reactive.Value(None)
    metric_plot_y_range = reactive.Value(None)

    # Create reactive groups
    file_reactives = create_file_handling_reactives(inputs, session)
    data_reactives = create_data_processing_reactives(
        inputs, file_reactives, metric_plot_x_range, metric_plot_y_range
    )
    ui_reactives = create_ui_reactives(inputs, file_reactives, data_reactives)
    visualization_reactives = create_visualization_reactives(
        inputs, data_reactives, metric_plot_x_range, metric_plot_y_range
    )
    statistics_reactives = create_statistics_reactives(
        inputs, file_reactives, data_reactives
    )

    # Enable dynamic theme switching
    shinyswatch.theme_picker_server()

    @reactive.Effect
    @reactive.event(inputs.logout)
    async def _():
        await session.send_custom_message("logout", {})
