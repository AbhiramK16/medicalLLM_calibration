"""Shared model-call interface.

Provider SDK details, retries, timeouts, and token/cost capture belong here.
Experiment pipelines should call this layer instead of importing provider SDKs
directly.
"""

from __future__ import annotations

import json
import os
import urllib.request


def query(
    backbone: str,
    prompt: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> str:
    """Return the raw completion text for one prompt via an OpenAI-compatible endpoint.

    Credentials default to the MODEL_API_KEY / MODEL_API_BASE environment
    variables. Retries, timeouts, token accounting, and per-case cost capture
    are TODO.
    """
    key = api_key or os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set; add it to .env")
    base = (
        api_base or os.environ.get("MODEL_API_BASE") or "https://api.openai.com/v1"
    ).rstrip("/")
    payload = {
        "model": backbone,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]
