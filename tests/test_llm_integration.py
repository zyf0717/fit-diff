"""
Tests for LLM integration functions.
"""

from unittest.mock import ANY

import pandas as pd
import pytest

from src.utils import llm_integration
from src.utils import generate_llm_summary_stream


class TestGenerateLLMSummaryStream:
    """Test cases for generate_llm_summary_stream function."""

    def test_normalize_ai_know_api_url_appends_chat_completions(self):
        assert (
            llm_integration._normalize_ai_know_api_url(
                "https://ai-know.nus.edu.sg/backend/completions/oai"
            )
            == "https://ai-know.nus.edu.sg/backend/completions/oai/chat/completions"
        )

    def test_normalize_ai_know_api_url_preserves_full_chat_path(self):
        assert (
            llm_integration._normalize_ai_know_api_url(
                "https://ai-know.nus.edu.sg/backend/v2/completions/oai/chat/completions"
            )
            == "https://ai-know.nus.edu.sg/backend/v2/completions/oai/chat/completions"
        )

    def create_test_stats(self):
        """Create test statistics data."""
        bias_stats = pd.DataFrame(
            {"Metric": ["Mean Bias", "Cohen's d"], "Value": [2.5, 0.3]}
        )

        accuracy_stats = pd.DataFrame({"Metric": ["MAE", "RMSE"], "Value": [3.2, 4.1]})

        agreement_stats = pd.DataFrame(
            {"Metric": ["Correlation Coefficient"], "Value": [0.85]}
        )

        return bias_stats, accuracy_stats, agreement_stats

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
        bias_stats, accuracy_stats, agreement_stats = self.create_test_stats()

        # Test with one empty DataFrame
        result_chunks = []
        async for chunk in generate_llm_summary_stream(
            "heart_rate", bias_stats, pd.DataFrame(), agreement_stats
        ):
            result_chunks.append(chunk)

        # Should return specific error message without calling API
        assert len(result_chunks) == 1
        assert "Insufficient statistics" in result_chunks[0]

    def test_resolve_llm_request_config_prefers_ai_know(self, monkeypatch):
        """AI_KNOW should be the default endpoint when configured."""
        monkeypatch.setenv(
            "AI_KNOW_API_URL",
            "https://ai-know.nus.edu.sg/backend/completions/oai",
        )
        monkeypatch.setenv("AI_KNOW_API_KEY", "ai-know-token")
        monkeypatch.setenv("LLM_API_URL", "https://legacy.example.com")
        monkeypatch.setenv("API_KEY_ID", "legacy-id")
        monkeypatch.setenv("API_KEY_SECRET", "legacy-secret")

        llm_api_url, headers, model = llm_integration._resolve_llm_request_config()

        assert (
            llm_api_url
            == "https://ai-know.nus.edu.sg/backend/completions/oai/chat/completions"
        )
        assert headers == {
            "Ocp-Apim-Subscription-Key": "ai-know-token",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        assert model == "gpt-5.4"

    def test_resolve_llm_request_config_uses_ai_know_model_override(
        self, monkeypatch
    ):
        """AI_KNOW_MODEL should override the default OpenAI-compatible model."""
        monkeypatch.setenv(
            "AI_KNOW_API_URL",
            "https://ai-know.nus.edu.sg/backend/completions/oai",
        )
        monkeypatch.setenv("AI_KNOW_API_KEY", "ai-know-token")
        monkeypatch.setenv("AI_KNOW_MODEL", "gpt-custom")

        llm_api_url, headers, model = llm_integration._resolve_llm_request_config()

        assert (
            llm_api_url
            == "https://ai-know.nus.edu.sg/backend/completions/oai/chat/completions"
        )
        assert headers == {
            "Ocp-Apim-Subscription-Key": "ai-know-token",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        assert model == "gpt-custom"

    def test_resolve_llm_request_config_falls_back_to_legacy(self, monkeypatch):
        """Legacy endpoint should still work when AI_KNOW is not configured."""
        monkeypatch.delenv("AI_KNOW_API_URL", raising=False)
        monkeypatch.delenv("AI_KNOW_API_KEY", raising=False)
        monkeypatch.delenv("AI_KNOW_MODEL", raising=False)
        monkeypatch.setenv("LLM_API_URL", "https://legacy.example.com")
        monkeypatch.setenv("API_KEY_ID", "legacy-id")
        monkeypatch.setenv("API_KEY_SECRET", "legacy-secret")

        llm_api_url, headers, model = llm_integration._resolve_llm_request_config()

        assert llm_api_url == "https://legacy.example.com"
        assert headers == {
            "CF-Access-Client-Id": "legacy-id",
            "CF-Access-Client-Secret": "legacy-secret",
            "Content-Type": "application/json",
        }
        assert model is None

    @pytest.mark.asyncio
    async def test_generate_llm_summary_stream_uses_ai_know_by_default(
        self, monkeypatch
    ):
        """Streaming requests should use AI_KNOW auth and URL when available."""
        bias_stats, accuracy_stats, agreement_stats = self.create_test_stats()
        requests = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"Hello "}}]}'
                yield 'data: {"choices":[{"delta":{"content":"world"}}]}'
                yield "data: [DONE]"

        class FakeStreamContext:
            async def __aenter__(self):
                return FakeResponse()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, headers=None, json=None):
                requests.append(
                    {
                        "method": method,
                        "url": url,
                        "headers": headers,
                        "json": json,
                    }
                )
                return FakeStreamContext()

        monkeypatch.setenv(
            "AI_KNOW_API_URL",
            "https://ai-know.nus.edu.sg/backend/completions/oai",
        )
        monkeypatch.setenv("AI_KNOW_API_KEY", "ai-know-token")
        monkeypatch.setenv("LLM_API_URL", "https://legacy.example.com")
        monkeypatch.setenv("API_KEY_ID", "legacy-id")
        monkeypatch.setenv("API_KEY_SECRET", "legacy-secret")
        monkeypatch.setattr(llm_integration.httpx, "AsyncClient", FakeAsyncClient)

        result_chunks = []
        async for chunk in generate_llm_summary_stream(
            "heart_rate", bias_stats, accuracy_stats, agreement_stats
        ):
            result_chunks.append(chunk)

        assert "".join(result_chunks) == "Hello world"
        assert requests == [
            {
                "method": "POST",
                "url": "https://ai-know.nus.edu.sg/backend/completions/oai/chat/completions",
                "headers": {
                    "Ocp-Apim-Subscription-Key": "ai-know-token",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                "json": {
                    "messages": ANY,
                    "model": "gpt-5.4",
                    "stream": True,
                },
            }
        ]
