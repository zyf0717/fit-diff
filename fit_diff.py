import logging

from shiny import App

from src.app_server import server
from src.app_ui import app_ui

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Output to console/terminal
)

if __name__ == "__main__":
    app = App(app_ui, server)
    app.run(port=8001)
