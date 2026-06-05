import shinyswatch
from shiny import ui

app_ui = ui.page_fluid(
    ui.tags.style("""
        /* Make the sidebar independently scrollable */
        .sidebar {
            max-height: 100vh;
            overflow-y: auto;
            position: sticky;
            top: 0;
            background: inherit;
        }

        .cloud-pair-actions {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
            min-height: 100%;
        }

        .cloud-pair-table-wrap {
            display: flex;
            justify-content: center;
            width: 100%;
        }

        .cloud-range-metric-body {
            align-items: stretch;
            row-gap: 0.75rem;
        }

        .cloud-range-metric-plot {
            min-height: 170px;
        }

        .cloud-range-metric-stats {
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.45rem;
            height: 100%;
            min-height: 170px;
            padding-left: 0.75rem;
            border-left: 1px solid var(--bs-border-color);
        }

        .cloud-range-metric-stat {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.75rem;
            font-size: 0.9rem;
            line-height: 1.2;
        }

        .cloud-range-metric-stat-label,
        .cloud-range-metric-empty {
            color: var(--bs-secondary-color);
        }

        .cloud-range-metric-stat-value {
            font-variant-numeric: tabular-nums;
            text-align: right;
            font-weight: 600;
        }

        @media (max-width: 991.98px) {
            .cloud-range-metric-stats {
                min-height: auto;
                padding-left: 0;
                padding-top: 0.75rem;
                border-left: none;
                border-top: 1px solid var(--bs-border-color);
            }
        }
        """),
    ui.tags.script("""
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
        """),
    ui.tags.script("""
        (function() {
        function parseColor(color) {
            if (!color) return null;
            var value = color.trim();
            if (value.startsWith("rgb")) {
                var matches = value.match(/[\\d.]+/g);
                if (!matches || matches.length < 3) return null;
                return matches.slice(0, 3).map(function(channel) {
                    return Number(channel);
                });
            }
            if (value.startsWith("#")) {
                var hex = value.slice(1);
                if (hex.length === 3) {
                    hex = hex.split("").map(function(char) {
                        return char + char;
                    }).join("");
                }
                if (hex.length !== 6) return null;
                return [0, 2, 4].map(function(index) {
                    return parseInt(hex.slice(index, index + 2), 16);
                });
            }
            return null;
        }

        function withAlpha(color, alpha, fallback) {
            var rgb = parseColor(color);
            if (!rgb) return fallback;
            return "rgba(" + rgb[0] + ", " + rgb[1] + ", " + rgb[2] + ", " + alpha + ")";
        }

        function getThemeValue(cssVarName, fallback) {
            var rootStyles = getComputedStyle(document.documentElement);
            var bodyStyles = getComputedStyle(document.body);
            return rootStyles.getPropertyValue(cssVarName).trim()
                || bodyStyles.getPropertyValue(cssVarName).trim()
                || fallback;
        }

        function getLuminance(rgb) {
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        }

        function collectPlotlyTheme() {
            var bodyBg = getThemeValue("--bs-body-bg", getComputedStyle(document.body).backgroundColor || "#ffffff");
            var bodyColor = getThemeValue("--bs-body-color", getComputedStyle(document.body).color || "#212529");
            var borderColor = getThemeValue("--bs-border-color", "#ced4da");
            var secondaryColor = getThemeValue("--bs-secondary-color", bodyColor);
            var bodyRgb = parseColor(bodyBg) || [255, 255, 255];
            var darkMode = getLuminance(bodyRgb) < 140;

            return {
                mode: darkMode ? "dark" : "light",
                paper_bgcolor: "rgba(0, 0, 0, 0)",
                plot_bgcolor: "rgba(0, 0, 0, 0)",
                font_color: bodyColor,
                axis_color: withAlpha(secondaryColor, darkMode ? 0.8 : 0.7, bodyColor),
                grid_color: withAlpha(borderColor, darkMode ? 0.35 : 0.3, "rgba(127, 127, 127, 0.3)"),
                muted_color: withAlpha(bodyColor, darkMode ? 0.45 : 0.25, "rgba(127, 127, 127, 0.25)"),
                annotation_bgcolor: withAlpha(bodyBg, darkMode ? 0.92 : 0.9, bodyBg),
                zero_line_color: withAlpha(bodyColor, darkMode ? 0.6 : 0.4, "rgba(0, 0, 0, 0.4)"),
                box_fill_color: withAlpha(bodyColor, darkMode ? 0.22 : 0.14, "rgba(127, 127, 127, 0.14)"),
                box_line_color: withAlpha(bodyColor, darkMode ? 0.42 : 0.28, "rgba(127, 127, 127, 0.28)")
            };
        }

        var debounceHandle = null;
        var lastThemePayload = null;
        function pushPlotlyTheme() {
            if (!window.Shiny || !window.Shiny.setInputValue) return;
            var nextThemePayload = collectPlotlyTheme();
            var serializedPayload = JSON.stringify(nextThemePayload);
            if (serializedPayload === lastThemePayload) return;
            lastThemePayload = serializedPayload;
            window.Shiny.setInputValue("plotly_theme", nextThemePayload, {priority: "event"});
        }

        function queuePlotlyThemePush() {
            if (debounceHandle !== null) {
                window.clearTimeout(debounceHandle);
            }
            debounceHandle = window.setTimeout(pushPlotlyTheme, 50);
        }

        document.addEventListener("DOMContentLoaded", function() {
            queuePlotlyThemePush();

            var observer = new MutationObserver(queuePlotlyThemePush);
            observer.observe(document.documentElement, {
                attributes: true,
                attributeFilter: ["class", "style", "data-bs-theme"]
            });
            document.addEventListener("change", queuePlotlyThemePush, true);
        });
        })();
        """),
    ui.navset_bar(
        ui.nav_panel(
            "Benchmarking",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.tags.script("""
                Shiny.addCustomMessageHandler("logout", function(_) {
                    window.location.href = "https://fit-diff.paperclips.dev/cdn-cgi/access/logout";
                });
                """),
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
                ui.output_ui("benchmarkingContent"),
            ),
        ),
        ui.nav_panel(
            "Batch Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.output_ui("cloudGroupSelector"),
                    ui.output_ui("cloudDateRangeSelector"),
                    ui.output_ui("cloudAutoShiftSelector"),
                    ui.output_ui("cloudMetricSelector"),
                ),
                ui.accordion(
                    ui.accordion_panel(
                        "Select File Pairs",
                        ui.layout_columns(
                            ui.div(
                                ui.input_action_button(
                                    "cloudSelectAllRows",
                                    "Select All Rows",
                                ),
                                ui.input_action_button(
                                    "cloudRefreshAnalysis",
                                    "Refresh Sections Below",
                                ),
                                class_="cloud-pair-actions",
                            ),
                            ui.div(
                                ui.output_data_frame("cloudPairSelectionTable"),
                                class_="cloud-pair-table-wrap",
                            ),
                            col_widths=[2, 10],
                        ),
                        value="cloud-pair-selection",
                    ),
                    id="cloudPairSelectionAccordion",
                    open="cloud-pair-selection",
                    multiple=False,
                    width="100%",
                ),
                ui.hr(),
                ui.output_ui("cloudMetricRangePlotGrid"),
                ui.hr(),
                ui.card(
                    ui.card_header("LLM Generated Summary"),
                    ui.layout_columns(
                        ui.input_action_button("cloudLlmSummaryRegen", "Ask BotBot!"),
                        ui.div(
                            ui.output_ui("cloudLlmLoadingText"),
                            ui.output_markdown_stream(
                                "cloudStreamOutput", auto_scroll=False
                            ),
                        ),
                        col_widths=[3, 9],
                    ),
                ),
                ui.hr(),
                ui.card(
                    ui.card_header("Per-Pair Summary"),
                    ui.output_data_frame("cloudPairSummaryTable"),
                ),
            ),
        ),
        id="mainNavset",
        title="FIT-DIFF",
    ),
    theme=shinyswatch.theme.flatly,
)
