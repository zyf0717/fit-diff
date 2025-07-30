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

## Usage

Run the application:

```bash
python app.py
```

Then open your browser to the displayed URL (typically <http://localhost:8000>).

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Contributing

This is a lean implementation. To extend functionality:

1. Extend comparison logic in `DiffAnalyzer`
2. Add new UI components in `app.py`
