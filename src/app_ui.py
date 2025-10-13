import shinyswatch
from shiny import ui

app_ui = ui.page_fluid(
    ui.tags.style(
        """
        /* Make the sidebar independently scrollable */
        .sidebar {
            max-height: 100vh;
            overflow-y: auto;
            position: sticky;
            top: 0;
            background: inherit;
        }
        """
    ),
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
    ui.navset_bar(
        ui.nav_panel(
            "Benchmarking",
            ui.page_sidebar(
                ui.sidebar(
                    ui.tags.script(
                        """
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://fit-diff.paperclips.dev/cdn-cgi/access/logout";
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
        ),
        ui.nav_panel(
            "Batch",
            ui.page_sidebar(
                ui.sidebar(
                    ui.output_ui("batchTagOptions"),
                ),
                ui.output_ui("batchContent"),
            ),
        ),
        title="",
    ),
    theme=shinyswatch.theme.flatly,
)
