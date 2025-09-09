"""
Tests for LLM integration functions.
"""

import pandas as pd
import pytest

from src.utils import generate_llm_summary_stream


class TestGenerateLLMSummaryStream:
    """Test cases for generate_llm_summary_stream function."""

    def create_test_stats(self):
        """Create test statistics data."""
        bias_stats = pd.DataFrame(
            {"Metric": ["Mean Bias", "Cohen's d"], "Value": [2.5, 0.3]}
        )

        error_stats = pd.DataFrame({"Metric": ["MAE", "RMSE"], "Value": [3.2, 4.1]})

        correlation_stats = pd.DataFrame(
            {"Metric": ["Correlation Coefficient"], "Value": [0.85]}
        )

        return bias_stats, error_stats, correlation_stats

    @pytest.mark.asyncio
    async def test_generate_llm_summary_stream_empty_stats(self):
        """Test LLM summary stream with empty statistics."""
        # Test with None stats
        result_chunks = []
        async for chunk in generate_llm_summary_stream(
            "heart_rate", None, pd.DataFrame(), pd.DataFrame()
        ):
            result_chunks.append(chunk)

        # Should return specific error message without calling API
        assert len(result_chunks) == 1
        assert "Insufficient statistics" in result_chunks[0]

    @pytest.mark.asyncio
    async def test_generate_llm_summary_stream_empty_dataframes(self):
        """Test LLM summary stream with empty DataFrames."""
        # Test with empty DataFrames
        result_chunks = []
        async for chunk in generate_llm_summary_stream(
            "heart_rate", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        ):
            result_chunks.append(chunk)

        # Should return specific error message without calling API
        assert len(result_chunks) == 1
        assert "Insufficient statistics" in result_chunks[0]

    @pytest.mark.asyncio
    async def test_generate_llm_summary_stream_mixed_empty_stats(self):
        """Test LLM summary stream with mixed empty and non-empty stats."""
        bias_stats, error_stats, correlation_stats = self.create_test_stats()

        # Test with one empty DataFrame
        result_chunks = []
        async for chunk in generate_llm_summary_stream(
            "heart_rate", bias_stats, pd.DataFrame(), correlation_stats
        ):
            result_chunks.append(chunk)

        # Should return specific error message without calling API
        assert len(result_chunks) == 1
        assert "Insufficient statistics" in result_chunks[0]
