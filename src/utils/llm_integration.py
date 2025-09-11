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
API_KEY_ID = os.getenv("API_KEY_ID", "")
API_KEY_SECRET = os.getenv("API_KEY_SECRET", "")
LLM_API_URL = os.getenv("LLM_API_URL", "")


def _stats_payload(
    metric: str,
    validity: pd.DataFrame,
    precision: pd.DataFrame,
    reliability: pd.DataFrame,
) -> dict:
    return {
        "benchmark_metric": metric,
        "validity": validity.to_dict(orient="records"),
        "precision": precision.to_dict(orient="records"),
        "reliability": reliability.to_dict(orient="records"),
    }


async def generate_llm_summary_stream(
    metric: str,
    validity_stats: pd.DataFrame,
    precision_stats: pd.DataFrame,
    reliability_stats: pd.DataFrame,
) -> AsyncGenerator[str, None]:
    """
    Stream a model-written summary for the provided statistics.
    Yields text chunks as they arrive (SSE 'delta.content').
    """
    # Guard: require all three stats
    if (
        validity_stats is None
        or validity_stats.empty
        or precision_stats is None
        or precision_stats.empty
        or reliability_stats is None
        or reliability_stats.empty
    ):
        yield "Insufficient statistics: all of validity, precision, and reliability stats must be present and non-empty."
        return

    records = _stats_payload(metric, validity_stats, precision_stats, reliability_stats)

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
                "validity, precision, reliability, significance tests, etc.\n\n"
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
    headers = {
        "CF-Access-Client-Id": API_KEY_ID,
        "CF-Access-Client-Secret": API_KEY_SECRET,
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(connect=10, read=None, write=10, pool=10)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", LLM_API_URL, headers=headers, json=payload
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
