import shinyswatch
from shiny import ui

app_ui = ui.page_fluid(
    ui.tags.script(
        """
        Shiny.addCustomMessageHandler("toggle_disabled", function(msg) {
        var el = document.getElementById(msg.id);
        if (!el) return;

        // Toggle the disabled attribute
        if (msg.disabled) {
            el.setAttribute("disabled", "disabled");
        } else {
            el.removeAttribute("disabled");
        }

        // Grey out the label (if it exists)
        var label = document.querySelector("label[for='" + msg.id + "']");
        if (label) {
            label.style.opacity = msg.disabled ? "0.5" : "1.0";
        }

        // Grey out the input itself for extra clarity
        el.style.opacity = msg.disabled ? "0.6" : "1.0";
        el.style.backgroundColor = msg.disabled ? "#f0f0f0" : "";
        });
        """
    ),
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
