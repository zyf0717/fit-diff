# FIT File Analysis Tool

A comprehensive Shiny Python web application primarily for comparing FIT files from fitness devices. This tool provides statistical analysis, interactive visualizations, and LLM-powered insights for fitness data.

Note: This project is in active development. Features and interfaces may change frequently.

## Core Features

- **Interactive Web Interface**: Upload and compare multiple FIT files, with partial CSV support
- **Statistical Analysis**: Bias, accuracy, agreement etc.
- **Interactive Visualizations**: Dynamic plots with zoom, pan, and selection capabilities

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```bash
uv python install
uv sync --locked
```

uv installs the Python version pinned in `.python-version` and creates a local
`.venv` with the dependencies in `uv.lock`. `pyproject.toml` is the dependency
manifest; commit it and the lockfile together when changing dependencies. No
environment activation is needed when using `uv run`.

`uv sync --locked` includes test and notebook tools from the `dev` group. For a
server-only environment, use `uv sync --locked --no-dev`. In an editor, select
the `.venv` interpreter for Python files and notebooks.

Pandas stays on version 2 because CSV timestamp conversion currently assumes
nanosecond datetime storage. Keep this constraint until that conversion and the
catalogue's missing-value handling support pandas 3.

### Optional: LLM Integration Setup

For AI-powered analysis features, create a `.env` file in the project root:

```env
AI_KNOW_API_KEY=your_ai_know_api_key
AI_KNOW_API_URL=https://ai-know.nus.edu.sg/backend/completions/oai
API_KEY_ID=your_api_key_id
API_KEY_SECRET=your_api_key_secret
LLM_API_URL=your_llm_api_url
```

`AI_KNOW_API_URL` + `AI_KNOW_API_KEY` are now the default LLM path. The legacy `LLM_API_URL` + Cloudflare access headers remain supported as fallback.

## Usage

### Web Application

Run the main Shiny application:

```bash
uv run --locked fit_diff.py
```

Then open your browser to [http://localhost:8001](http://localhost:8001).

#### Running with systemd (Linux)

The user service runs the application in the foreground, restarts it on failure,
and sends logs to the journal. It expects the repository at `~/repos/fit-diff`
and uses its `.venv` directly; adjust the paths in `systemd/fit-diff.service` if
your installation differs. Run the installation commands above before starting
the service. The application loads the repository's `.env` file as usual.

Install the service and enable lingering so it starts at boot and keeps running
without a login session:

```bash
mkdir -p ~/.config/systemd/user
install -m 644 systemd/fit-diff.service ~/.config/systemd/user/fit-diff.service
systemctl --user daemon-reload
loginctl enable-linger "$USER"
```

When migrating from cron, back up `crontab -l`, remove only the FIT Diff
`@reboot .../fit-diff/cron.sh` entry with `crontab -e`, and stop the existing
`python fit_diff.py` process before starting the service to free port 8001.

```bash
systemctl --user enable --now fit-diff.service
systemctl --user status fit-diff.service
journalctl --user -u fit-diff.service -f
```

To deploy an update, stop the service before syncing its environment:

```bash
systemctl --user stop fit-diff.service
uv sync --locked --no-dev
systemctl --user start fit-diff.service
```

If you change the unit file, reinstall it and run `systemctl --user daemon-reload`
before restarting. To stop the service and disable startup at boot, use
`systemctl --user disable --now fit-diff.service`.

#### Using the Interface

1. **Upload FIT Files**: Use the file upload widgets to select reference and test FIT files
2. **Select Metrics**: Choose which metrics to compare (heart rate, cadence, speed etc.)
3. **Configure Analysis**: Set start and end times, shifting options etc.
4. **View Results**: Examine interactive plots, statistical summaries, and AI-generated insights

### Benchmarking FIT Parsers

Compare performance between Garmin FIT SDK and fitparse:

```bash
uv run --locked benchmark_fit_parsers.py
```

This tool helps determine which parser works best for your specific FIT files and use cases.

### Running Tests

Execute the full test suite:

```bash
uv run --locked pytest tests/
```

Run specific test modules:

```bash
uv run --locked pytest tests/test_data_processing.py
uv run --locked pytest tests/test_statistics.py
uv run --locked pytest tests/test_visualizations.py
```

## Project Structure

```text
fit-diff/
├── src/                         # Main application source code
│   ├── app_server.py            # Shiny server logic and reactive coordination
│   ├── app_ui.py                # User interface definition and layout
│   ├── reactives_*.py           # Reactive modules for different features
│   └── utils/                   # Utility functions and helpers
│       ├── data_processing.py   # FIT file parsing and data transformation
│       ├── statistics.py        # Statistical analysis and metrics
│       ├── visualizations.py    # Plot generation and styling
│       └── llm_integration.py   # AI-powered analysis (optional)
├── tests/                       # Test suite
│   ├── fixtures/                # Sample FIT files for testing
│   └── test_*.py                # Unit tests for each module
├── sandbox/                     # Development experiments and notebooks
├── fit_diff.py                  # Main application entry point
├── benchmark_fit_parsers.py     # Parser performance comparison tool
├── pyproject.toml               # Runtime and development dependencies
├── uv.lock                      # Locked dependency versions
└── .python-version              # Python version used by uv
```

### Adding New Features

1. **Data Processing**: Add new parsers or metrics in `src/utils/data_processing.py`
2. **Statistics**: Implement new statistical methods in `src/utils/statistics.py`
3. **Visualizations**: Create new plot types in `src/utils/visualizations.py`
4. **UI Components**: Add reactive components in the appropriate `src/reactives_*.py` module, and update `src/reactives_ui.py` accordingly
5. **Server Orchestration**: Update `src/app_server.py` to integrate all new additions into the reactive flow of the app
