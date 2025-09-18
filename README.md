# FIT File Analysis Tool

A comprehensive Shiny Python web application for analyzing, comparing, and benchmarking FIT files from fitness devices. This tool provides advanced statistical analysis, interactive visualizations, and LLM-powered insights for fitness data.

## Core Features

- **Interactive Web Interface**: Upload and compare multiple FIT files
- **Statistical Analysis**: Bias, accuracy, agreement etc.
- **Interactive Visualizations**: Dynamic plots with zoom, pan, and selection capabilities

## Installation

### Prerequisites

- Python 3.12 or higher
- Conda (recommended) or pip

### Using Conda (Recommended)

1. Create the environment:

```bash
conda env create -f environment.yml
```

1. Activate the environment:

```bash
conda activate fit-diff
```

### Optional: LLM Integration Setup

For AI-powered analysis features, create a `.env` file in the project root:

```env
API_KEY_ID=your_api_key_id
API_KEY_SECRET=your_api_key_secret
LLM_API_URL=your_llm_api_url
```

### Manual Installation

Alternatively, install dependencies with pip:

```bash
pip install shiny==1.4.0 shinyswatch==0.9.0 fitparse garmin-fit-sdk==21.171.0
pip install numpy pandas plotly pytest python-dotenv python-duckdb scipy
pip install aiohttp requests faicons==0.2.2 shinywidgets==0.6.2
```

## Usage

### Web Application

Run the main Shiny application:

```bash
python fit_diff.py
```

Then open your browser to [http://localhost:8001](http://localhost:8001).

#### Using the Interface

1. **Upload FIT Files**: Use the file upload widgets to select reference and test FIT files
2. **Select Metrics**: Choose which metrics to compare (heart rate, cadence, speed etc.)
3. **Configure Analysis**: Set start and end times, shifting options etc.
4. **View Results**: Examine interactive plots, statistical summaries, and AI-generated insights

### Benchmarking FIT Parsers

Compare performance between Garmin FIT SDK and fitparse:

```bash
python benchmark_fit_parsers.py
```

This tool helps determine which parser works best for your specific FIT files and use cases.

### Running Tests

Execute the full test suite:

```bash
pytest tests/
```

Run specific test modules:

```bash
pytest tests/test_data_processing.py
pytest tests/test_statistics.py
pytest tests/test_visualizations.py
```

## Project Structure

```text
fit-diff/
├── src/                          # Main application source code
│   ├── app_server.py            # Shiny server logic and reactive coordination
│   ├── app_ui.py                # User interface definition and layout
│   ├── reactives_*.py           # Reactive modules for different features
│   └── utils/                   # Utility functions and helpers
│       ├── data_processing.py   # FIT file parsing and data transformation
│       ├── statistics.py        # Statistical analysis and metrics
│       ├── visualizations.py    # Plot generation and styling
│       └── llm_integration.py   # AI-powered analysis (optional)
├── tests/                       # Test suite
│   ├── fixtures/               # Sample FIT files for testing
│   └── test_*.py               # Unit tests for each module
├── sandbox/                    # Development experiments and notebooks
├── fit_diff.py                 # Main application entry point
├── benchmark_fit_parsers.py    # Parser performance comparison tool
└── environment.yml             # Conda environment specification
```

### Adding New Features

1. **Data Processing**: Add new parsers or metrics in `src/utils/data_processing.py`
1. **Statistics**: Implement new statistical methods in `src/utils/statistics.py`
1. **Visualizations**: Create new plot types in `src/utils/visualizations.py`
1. **UI Components**: Add reactive components in the appropriate `src/reactives_*.py` module
