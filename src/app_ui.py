import shinyswatch
from shiny import ui

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.tags.script(
                """
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://fit-diff.paperclipds.dev/cdn-cgi/access/logout";
                });
                """
            ),
            ui.input_file(
                "testFileUpload",
                "Upload test file(s)",
                multiple=True,
                accept=[".fit", ".csv", ".FIT", ".CSV"],
            ),
            ui.input_file(
                "refFileUpload",
                "Upload reference file(s)",
                multiple=True,
                accept=[".fit", ".csv", ".FIT", ".CSV"],
            ),
            shinyswatch.theme_picker_ui(),
            ui.hr(),
            ui.input_action_button("logout", "Logout"),
        ),
        ui.output_ui("mainContent"),
    ),
    theme=shinyswatch.theme.flatly,
)
