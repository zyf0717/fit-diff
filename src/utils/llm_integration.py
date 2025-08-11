"""
LLM integration utilities for generating summaries.
"""

import json
import os

import aiohttp
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)
API_KEY_ID = os.getenv("API_KEY_ID", "")
API_KEY_SECRET = os.getenv("API_KEY_SECRET", "")
LLM_API_URL = os.getenv("LLM_API_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")


async def generate_llm_summary(
    metric: str,
    bias_stats: pd.DataFrame,
    error_stats: pd.DataFrame,
    correlation_stats: pd.DataFrame,
) -> str:
    """
    Generate a summary for the LLM based on the provided statistics.
    """
    # Only proceed if all three stats are present and non-empty
    if (
        bias_stats is not None
        and not bias_stats.empty
        and error_stats is not None
        and not error_stats.empty
        and correlation_stats is not None
        and not correlation_stats.empty
    ):
        records = {
            "benchmark_metric": metric,
            "bias": bias_stats.to_dict(orient="records"),
            "error_magnitude": error_stats.to_dict(orient="records"),
            "correlation": correlation_stats.to_dict(orient="records"),
        }
        response = await api_call_to_llm(records)
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        return "Insufficient statistics: all of bias, error, and correlation stats must be present and non-empty."


async def api_call_to_llm(records: dict) -> dict:
    """
    Make an async API call to the LLM with the provided records.
    """
    headers = {
        "CF-Access-Client-Id": API_KEY_ID,
        "CF-Access-Client-Secret": API_KEY_SECRET,
        "Content-Type": "application/json",
    }
    prompt = f"""
    Reason logically. Interpret the following JSON benchmark stats for non-technical readers, and explain in layman terms.
    Context: benchmarking of wearable devices for {records.get("benchmark_metric", "")}.
    Use these categories as potential sources: bias, error magnitude (MAE, RMSE), correlation, significance tests.

    - Always end with a verdict.
    - Focus primarily on the key statistics that materially influence your verdict.
    - Preserve all numbers exactly as given; do not change units or rounding.
    - No speculative causes; caveat clearly if something is an inference.
    - Dashboard tone; ≤ 200 words; no preamble, no code fences, no quotes.
    - Output a valid HTML snippet only (no <html>, <head>, or <body>).
    - Use only <p>, <ul>, <li>, <strong>, <em>.

    JSON:
    {json.dumps(records, indent=2)}
    """
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                LLM_API_URL, headers=headers, json=data, timeout=30
            ) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        return {"error": str(e)}
