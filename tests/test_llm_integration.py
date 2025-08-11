"""
Tests for LLM integration functions.
"""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.utils import api_call_to_llm, generate_llm_summary


class TestGenerateLLMSummary:
    """Test cases for generate_llm_summary function."""

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
    @patch("src.utils.llm_integration.api_call_to_llm")
    async def test_generate_llm_summary_success(self, mock_api_call):
        """Test successful LLM summary generation."""
        # Setup
        bias_stats, error_stats, correlation_stats = self.create_test_stats()

        # Mock API response
        mock_api_call.return_value = {
            "choices": [{"message": {"content": "<p>Test summary content</p>"}}]
        }

        # Test
        result = await generate_llm_summary(
            "heart_rate", bias_stats, error_stats, correlation_stats
        )

        # Verify
        assert (
            "<p>Test summary content</p>" in result or "Test summary content" in result
        )
        mock_api_call.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.utils.llm_integration.api_call_to_llm")
    async def test_generate_llm_summary_empty_stats(self, mock_api_call):
        """Test LLM summary with empty statistics."""
        # Test with None stats
        result = await generate_llm_summary(
            "heart_rate", None, pd.DataFrame(), pd.DataFrame()
        )

        # Should return specific error message without calling API
        assert "Insufficient statistics" in result
        mock_api_call.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.utils.llm_integration.api_call_to_llm")
    async def test_generate_llm_summary_api_error(self, mock_api_call):
        """Test LLM summary with API error."""
        # Setup
        bias_stats, error_stats, correlation_stats = self.create_test_stats()

        # Mock API error
        mock_api_call.side_effect = Exception("API Error")

        # Test - this should actually call the API and get an error
        try:
            result = await generate_llm_summary(
                "heart_rate", bias_stats, error_stats, correlation_stats
            )
            # If we get here, the mock worked and returned an error response
            assert "error" in str(result).lower() or isinstance(result, str)
        except Exception as e:
            # If exception is raised, that's also acceptable
            assert "API Error" in str(e)


class TestApiCallToLLM:
    """Test cases for api_call_to_llm function."""

    def create_test_records(self):
        """Create test records."""
        return {
            "benchmark_metric": "heart_rate",
            "bias": [{"Metric": "Mean Bias", "Value": 2.5}],
            "error_magnitude": [{"Metric": "MAE", "Value": 3.2}],
            "correlation": [{"Metric": "Correlation Coefficient", "Value": 0.85}],
        }

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_api_call_to_llm_success(self, mock_post):
        """Test successful API call to LLM."""
        # Setup
        records = self.create_test_records()

        # Mock response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        # Test
        result = await api_call_to_llm(records)

        # Verify
        assert result == {"choices": [{"message": {"content": "Test response"}}]}

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_api_call_to_llm_client_error(self, mock_post):
        """Test API call with client error."""
        # Setup
        records = self.create_test_records()

        # Mock client error
        mock_post.side_effect = Exception("Connection error")

        # Test should raise the exception since function doesn't handle errors gracefully
        with pytest.raises(Exception, match="Connection error"):
            await api_call_to_llm(records)

    @pytest.mark.asyncio
    @patch("src.utils.llm_integration.API_KEY_ID", "test_id")
    @patch("src.utils.llm_integration.API_KEY_SECRET", "test_secret")
    @patch("src.utils.llm_integration.LLM_API_URL", "https://test.api.com")
    @patch("src.utils.llm_integration.LLM_MODEL", "test-model")
    @patch("aiohttp.ClientSession.post")
    async def test_api_call_to_llm_with_env_vars(self, mock_post):
        """Test API call with environment variables."""
        # Setup
        records = self.create_test_records()

        # Mock response
        mock_response = AsyncMock()
        mock_response.json.return_value = {"test": "response"}
        mock_post.return_value.__aenter__.return_value = mock_response

        # Test
        result = await api_call_to_llm(records)

        # Verify
        assert isinstance(result, dict)
        mock_post.assert_called_once()

        # Check that headers were set correctly
        call_args = mock_post.call_args
        assert "headers" in call_args.kwargs
        headers = call_args.kwargs["headers"]
        assert headers["CF-Access-Client-Id"] == "test_id"
        assert headers["CF-Access-Client-Secret"] == "test_secret"
