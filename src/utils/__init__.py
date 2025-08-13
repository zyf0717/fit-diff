"""
Utils package - refactored into separate modules for better organization.

This __init__.py maintains backward compatibility by importing all functions
from the separate modules.
"""

# Data processing functions
from .data_processing import (
    get_file_information,
    get_raw_data_sample,
    prepare_data_for_analysis,
    process_fit,
    remove_outliers,
)

# LLM integration functions
from .llm_integration import generate_llm_summary_stream

# Statistical analysis functions
from .statistics import (
    calculate_basic_stats,
    get_bias_agreement_stats,
    get_correlation_stats,
    get_error_magnitude_stats,
)

# Visualization functions
from .visualizations import (
    create_bland_altman_plot,
    create_error_histogram,
    create_metric_plot,
    create_rolling_error_plot,
)

__all__ = [
    # Data processing
    "process_fit",
    "prepare_data_for_analysis",
    "remove_outliers",
    "get_file_information",
    "get_raw_data_sample",
    # Statistics
    "calculate_basic_stats",
    "get_bias_agreement_stats",
    "get_error_magnitude_stats",
    "get_correlation_stats",
    # Visualizations
    "create_metric_plot",
    "create_error_histogram",
    "create_bland_altman_plot",
    "create_rolling_error_plot",
    # LLM integration
    "generate_llm_summary_stream",
]
