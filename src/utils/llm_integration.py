"""
LLM integration utilities for generating summaries.
"""

import json
import os
from typing import AsyncGenerator

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)


def _normalize_ai_know_api_url(ai_know_api_url: str) -> str:
    normalized = ai_know_api_url.strip().rstrip("/")
    if normalized.endswith("/chat/completions") or normalized.endswith(
        "/v1/chat/completions"
    ):
        return normalized
    return f"{normalized}/chat/completions"


def _resolve_llm_request_config() -> tuple[str, dict[str, str], str | None]:
    ai_know_api_url = os.getenv("AI_KNOW_API_URL", "").strip()
    ai_know_api_key = os.getenv("AI_KNOW_API_KEY", "").strip()
    if ai_know_api_url and ai_know_api_key:
        ai_know_model = os.getenv("AI_KNOW_MODEL", "gpt-5.4").strip()
        return (
            _normalize_ai_know_api_url(ai_know_api_url),
            {
                "Ocp-Apim-Subscription-Key": ai_know_api_key,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            ai_know_model or "gpt-5.4",
        )

    llm_api_url = os.getenv("LLM_API_URL", "").strip()
    headers = {"Content-Type": "application/json"}

    api_key_id = os.getenv("API_KEY_ID", "").strip()
    if api_key_id:
        headers["CF-Access-Client-Id"] = api_key_id

    api_key_secret = os.getenv("API_KEY_SECRET", "").strip()
    if api_key_secret:
        headers["CF-Access-Client-Secret"] = api_key_secret

    return llm_api_url, headers, None


def _stats_payload(
    metric: str,
    bias: pd.DataFrame,
    accuracy: pd.DataFrame,
    agreement: pd.DataFrame,
) -> dict:
    return {
        "benchmark_metric": metric,
        "bias": bias.to_dict(orient="records"),
        "accuracy": accuracy.to_dict(orient="records"),
        "agreement": agreement.to_dict(orient="records"),
    }


async def generate_llm_summary_stream(
    metric: str,
    bias_stats: pd.DataFrame,
    accuracy_stats: pd.DataFrame,
    agreement_stats: pd.DataFrame,
) -> AsyncGenerator[str, None]:
    """
    Stream a model-written summary for the provided statistics.
    Yields text chunks as they arrive (SSE 'delta.content').
    """
    # Guard: require all three stats
    if (
        bias_stats is None
        or bias_stats.empty
        or accuracy_stats is None
        or accuracy_stats.empty
        or agreement_stats is None
        or agreement_stats.empty
    ):
        yield "Insufficient statistics."
        return

    records = _stats_payload(metric, bias_stats, accuracy_stats, agreement_stats)

    # Build OpenAI-style chat request
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise data analyst. Reason logically and explain to non-technical readers in plain language.\n"
                "OUTPUT RULES:\n"
                "- Return ONLY a valid Markdown snippet.\n"
                "- Around 300 words, no preamble, no code fences, no quotes.\n"
                "- Preserve all numbers EXACTLY as given (no rounding, no unit changes, no re-computation).\n"
                "- Use bullet points for key statistics and numeric ranges.\n"
                "- Focus on stats that materially influence the verdict; caveat clearly if anything is an inference.\n"
                "- Explain what the metrics mean in simple terms.\n"
                "- Always end with a **Verdict:** …"
            ),
        },
        {
            "role": "user",
            "content": (
                "Interpret the following JSON stats for wearable-device benchmarking of "
                f"{records.get('benchmark_metric', '')}. Consider these sources where present: "
                "bias, accuracy, agreement, significance tests, etc.\n\n"
                "Produce a concise summary as instructed above.\n\n"
                f"Payload (JSON):\n{json.dumps(records, ensure_ascii=False)}"
            ),
        },
    ]

    payload = {
        "messages": messages,
        # "temperature": 0.2,
        "stream": True,
    }
    llm_api_url, headers, model = _resolve_llm_request_config()
    if not llm_api_url:
        yield "LLM endpoint is not configured."
        return
    if model:
        payload["model"] = model

    timeout = httpx.Timeout(connect=10, read=None, write=10, pool=10)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", llm_api_url, headers=headers, json=payload
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    chunk = delta.get("content")
                    if chunk:
                        yield chunk
                except Exception:
                    # Ignore any non-JSON keepalives or partial lines
                    continue
