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


# async def generate_llm_summary(
#     metric: str,
#     bias_stats: pd.DataFrame,
#     error_stats: pd.DataFrame,
#     correlation_stats: pd.DataFrame,
# ) -> str:
#     """
#     Generate a summary for the LLM based on the provided statistics.
#     """
#     # Only proceed if all three stats are present and non-empty
#     if (
#         bias_stats is not None
#         and not bias_stats.empty
#         and error_stats is not None
#         and not error_stats.empty
#         and correlation_stats is not None
#         and not correlation_stats.empty
#     ):
#         records = {
#             "benchmark_metric": metric,
#             "bias": bias_stats.to_dict(orient="records"),
#             "error_magnitude": error_stats.to_dict(orient="records"),
#             "correlation": correlation_stats.to_dict(orient="records"),
#         }
#         response = await api_call_to_llm(records)
#         return response.get("choices", [{}])[0].get("message", {}).get("content", "")
#     else:
#         return "Insufficient statistics: all of bias, error, and correlation stats must be present and non-empty."


# async def api_call_to_llm(records: dict) -> dict:
#     """
#     Make an async API call to the LLM with the provided records.
#     """
#     headers = {
#         "CF-Access-Client-Id": API_KEY_ID,
#         "CF-Access-Client-Secret": API_KEY_SECRET,
#         "Content-Type": "application/json",
#     }
#     prompt = f"""
#     Reason logically. Interpret the following JSON benchmark stats for non-technical readers, and explain in layman terms.
#     Context: benchmarking of wearable devices for {records.get("benchmark_metric", "")}.
#     Use these categories as potential sources: bias, error magnitude (MAE, RMSE), correlation, significance tests.

#     - Always end with a verdict.
#     - Focus primarily on the key statistics that materially influence your verdict.
#     - Preserve all numbers exactly as given; do not change units or rounding.
#     - No speculative causes; caveat clearly if something is an inference.
#     - Dashboard tone; ≤ 200 words; no preamble, no code fences, no quotes.
#     - Output a valid HTML snippet only (no <html>, <head>, or <body>).
#     - Use only <p>, <ul>, <li>, <strong>, <em>.

#     JSON:
#     {json.dumps(records, indent=2)}
#     """
#     data = {
#         "messages": [{"role": "user", "content": prompt}],
#     }

#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 LLM_API_URL, headers=headers, json=data, timeout=30
#             ) as response:
#                 response.raise_for_status()
#                 return await response.json()
#     except aiohttp.ClientError as e:
#         return {"error": str(e)}


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
                "- Dashboard tone; ≤ 200 words; no preamble, no code fences, no quotes.\n"
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
