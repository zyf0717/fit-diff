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
    metric: str, bias: pd.DataFrame, err: pd.DataFrame, corr: pd.DataFrame
) -> dict:
    return {
        "benchmark_metric": metric,
        "bias": bias.to_dict(orient="records"),
        "error_magnitude": err.to_dict(orient="records"),
        "correlation": corr.to_dict(orient="records"),
    }


async def generate_llm_summary_stream(
    metric: str,
    bias_stats: pd.DataFrame,
    error_stats: pd.DataFrame,
    correlation_stats: pd.DataFrame,
) -> AsyncGenerator[str, None]:
    """
    Stream a model-written summary for the provided statistics.
    Yields text chunks as they arrive (SSE 'delta.content').
    """
    # Guard: require all three stats
    if (
        bias_stats is None
        or bias_stats.empty
        or error_stats is None
        or error_stats.empty
        or correlation_stats is None
        or correlation_stats.empty
    ):
        yield "Insufficient statistics: all of bias, error, and correlation stats must be present and non-empty."
        return

    records = _stats_payload(metric, bias_stats, error_stats, correlation_stats)

    # Build OpenAI-style chat request
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise data analyst. Reason logically and explain to non-technical readers in plain language. "
                "OUTPUT RULES:\n"
                "- Return ONLY a valid Markdown snippet.\n"
                "- Around 200 words, no preamble, no code fences, no quotes.\n"
                "- Preserve all numbers EXACTLY as given (no rounding, no unit changes, no re-computation).\n"
                "- Use bullet points for key statistics and numeric ranges.\n"
                "- Focus on stats that materially influence the verdict; caveat clearly if anything is an inference.\n"
                "- Always end with a **Verdict:** …"
            ),
        },
        {
            "role": "user",
            "content": (
                "Interpret the following JSON benchmark stats for wearable-device benchmarking of "
                f"{records.get('benchmark_metric', '')}. Consider these sources where present: "
                "bias, error magnitude (MAE, RMSE), correlation, significance tests.\n\n"
                "Produce a concise dashboard-style summary as instructed above.\n\n"
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
                    last_obj = obj
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    chunk = delta.get("content")
                    if chunk:
                        yield chunk
                except Exception:
                    # Ignore any non-JSON keepalives or partial lines
                    continue


# async def llm_stream_generator(
#     endpoint: str,
#     text: str,
# ) -> AsyncGenerator[str, None]:
#     # Normalize input
#     msg = (text or "Hello! What model are you?").strip()
#     if not msg:
#         return  # nothing to stream

#     payload = {
#         "messages": [{"role": "user", "content": msg}],
#         "stream": True,
#     }
#     headers = {
#         "CF-Access-Client-Id": API_KEY_ID,
#         "CF-Access-Client-Secret": API_KEY_SECRET,
#         "Content-Type": "application/json",
#     }
#     timeout = httpx.Timeout(connect=5, read=None, write=5, pool=10)

#     async with httpx.AsyncClient(timeout=timeout) as client:
#         async with client.stream("POST", endpoint, headers=headers, json=payload) as r:
#             r.raise_for_status()
#             async for line in r.aiter_lines():
#                 if not line or not line.startswith("data: "):
#                     continue
#                 data = line[6:].strip()
#                 if data == "[DONE]":
#                     break
#                 try:
#                     obj = json.loads(data)
#                     delta = obj.get("choices", [{}])[0].get("delta", {})
#                     chunk = delta.get("content")
#                     if chunk:
#                         yield chunk
#                 except Exception:
#                     # If a line isn't JSON, just ignore and continue streaming
#                     continue
