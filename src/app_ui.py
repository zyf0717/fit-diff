import shinyswatch
from shiny import ui

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.tags.script(
                """
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://fit-diff.yifei.sg/cdn-cgi/access/logout";
                });
                """
            ),
            ui.input_file(
                "testFileUpload",
                "Upload test FIT files",
                multiple=True,
                accept=[".fit"],
            ),
            ui.input_file(
                "refFileUpload",
                "Upload reference FIT files",
                multiple=True,
                accept=[".fit"],
            ),
            shinyswatch.theme_picker_ui(),
            ui.hr(),
            ui.input_action_button("logout", "Logout"),
        ),
        ui.output_ui("main_content"),
    ),
    theme=shinyswatch.theme.flatly,
)
