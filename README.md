# FIT File Diff Analyzer

A lean Shiny Python application for comparing uploaded FIT files.

## Features

- Upload multiple FIT files through web interface
- Extract and compare key metrics between files
- Display differences in an interactive table
- Support for common FIT file message types (record, session, lap, etc.)

## Installation

### Using Conda

1. Create the environment:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate fit-diff
```

### Manual Installation

```bash
pip install shiny pandas fitparse numpy
```

## Usage

Run the application:

```bash
python app.py
```

Then open your browser to the displayed URL (typically <http://localhost:8000>).

## Project Structure

```
fit-diff/
├── app.py                 # Main Shiny application
├── src/
│   ├── fit_processor.py   # FIT file processing logic
│   └── diff_analyzer.py   # Comparison and diff analysis
├── tests/
│   ├── test_fit_processor.py
│   ├── test_diff_analyzer.py
│   └── fixtures/          # Test FIT files
├── environment.yml        # Conda environment definition
└── README.md
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Supported FIT Message Types

- record (GPS/sensor data points)
- session (workout summaries)
- lap (lap/interval data)
- device_info (device information)
- file_id (file metadata)

## Contributing

This is a lean implementation. To extend functionality:

1. Add new message types in `FitProcessor.supported_messages`
2. Extend comparison logic in `DiffAnalyzer`
3. Add new UI components in `app.py`
