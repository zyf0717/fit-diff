from shiny import App

from src.app_server import server
from src.app_ui import app_ui

if __name__ == "__main__":
    app = App(app_ui, server)
    app.run(port=8001)
