# src/xrefrag/generate/common/llm.py
"""
XRefRAG Generator — LLM utilities

This module centralizes:
- OpenAI/AzureOpenAI client creation
- model call wrappers (JSON and text)
- robust JSON parsing / extraction (first JSON object)
- lightweight retry with jitter (for transient failures)
- consistent error shaping for reporting

Design notes:
- Keep this module dependency-light (stdlib + openai).
- Do not encode task-specific prompts here; prompts live in nodes/ scripts.
- For Azure, `model` MUST be the deployment name (e.g., "gpt-5.2-MBZUAI").
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    OpenAI = None
    AzureOpenAI = None


# ---------------------------------------------------------------------
# Errors / results
# ---------------------------------------------------------------------
@dataclass
class LLMCallResult:
    ok: bool
    content: str
    raw_json: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 1
    model: str | None = None


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------
def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def build_client(provider: str | None = None) -> tuple[Any, str]:
    """
    Returns (client, effective_model_name).

    - If provider=azure (or XREFRAG_LLM_PROVIDER=azure):
        uses AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
        and returns deployment name from AZURE_OPENAI_DEPLOYMENT
    - Else (openai):
        uses OPENAI_API_KEY and returns model from OPENAI_MODEL (optional; caller may override)
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai` (>=1.0.0)")

    provider = (provider or _env("XREFRAG_LLM_PROVIDER", "openai") or "openai").lower()

    if provider == "azure":
        if AzureOpenAI is None:
            raise RuntimeError("AzureOpenAI not available. Upgrade `openai` package (>=1.0.0).")

        endpoint = _env("AZURE_OPENAI_ENDPOINT")
        api_key = _env("AZURE_OPENAI_API_KEY")
        api_version = _env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        # Use AZURE_OPENAI_DEPLOYMENT_GPT52 for generate module
        deployment = _env("AZURE_OPENAI_DEPLOYMENT_GPT52") or _env("AZURE_OPENAI_DEPLOYMENT")

        missing = [
            k
            for k, v in [
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_API_VERSION", api_version),
                ("AZURE_OPENAI_DEPLOYMENT_GPT52", deployment),
            ]
            if not v
        ]
        if missing:
            raise RuntimeError(f"Missing Azure env vars: {', '.join(missing)}")

        # Normalize endpoint: remove trailing slashes and prevent "/openai" duplication
        # (the AzureOpenAI SDK will add the correct path internally)
        if endpoint:
            endpoint = endpoint.strip().rstrip("/")
            if endpoint.endswith("/openai"):
                endpoint = endpoint[:-7]

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        return client, deployment  # deployment is the 'model' arg for Azure

    # default: OpenAI
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    client = OpenAI(api_key=api_key)
    model = _env("OPENAI_MODEL", "gpt-4o-mini")
    return client, model


def is_azure_client(client: Any) -> bool:
    # safest: check class name (avoids import identity issues)
    return client.__class__.__name__ == "AzureOpenAI"


# ---------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.I)
    s = re.sub(r"\s*```$", "", s.strip())
    return s.strip()


def extract_first_json_object(s: str) -> str | None:
    if not s:
        return None
    s2 = strip_code_fences(s)
    m = _JSON_BLOCK_RE.search(s2)
    return m.group(0).strip() if m else None


def parse_json_lenient(s: str) -> dict[str, Any] | None:
    if not s:
        return None
    s2 = strip_code_fences(s)
    try:
        obj = json.loads(s2)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    obj_txt = extract_first_json_object(s2)
    if not obj_txt:
        return None
    try:
        obj = json.loads(obj_txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------
# OpenAI/Azure wrappers
# ---------------------------------------------------------------------
def call_chat_completion(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    seed: int | None = None,
) -> str:
    """
    One-shot chat completion. Returns assistant content (string).

    Notes:
    - For AzureOpenAI: `model` must be the *deployment* name.
    - Azure uses `max_completion_tokens`. Standard OpenAI uses `max_tokens`.
    """
    extra: dict[str, Any] = {}
    if seed is not None:
        extra["seed"] = int(seed)

    kwargs: dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **extra,
    )

    if is_azure_client(client):
        kwargs["max_completion_tokens"] = int(max_tokens)
    else:
        kwargs["max_tokens"] = int(max_tokens)

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def call_json(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    seed: int | None = None,
    retries: int = 2,
    min_backoff_s: float = 0.4,
    max_backoff_s: float = 1.2,
) -> LLMCallResult:
    attempts = 0
    last_err: str | None = None
    rng = random.Random(seed or 13)

    for i in range(max(1, retries + 1)):
        attempts += 1
        try:
            content = call_chat_completion(
                client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            obj = parse_json_lenient(content)
            if obj is None:
                last_err = "invalid_json"
                raise ValueError("Model output is not valid JSON object.")
            return LLMCallResult(
                ok=True, content=content, raw_json=obj, attempts=attempts, model=model
            )
        except Exception as e:
            last_err = str(e)
            if i < retries:
                sleep_s = min_backoff_s + (max_backoff_s - min_backoff_s) * rng.random()
                time.sleep(sleep_s)

    return LLMCallResult(
        ok=False, content="", raw_json=None, error=last_err, attempts=attempts, model=model
    )


def call_text(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    seed: int | None = None,
    retries: int = 1,
    min_backoff_s: float = 0.3,
    max_backoff_s: float = 0.9,
) -> LLMCallResult:
    attempts = 0
    last_err: str | None = None
    rng = random.Random(seed or 13)

    for i in range(max(1, retries + 1)):
        attempts += 1
        try:
            content = call_chat_completion(
                client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            if not content:
                last_err = "empty_response"
                raise ValueError("Model returned empty content.")
            return LLMCallResult(
                ok=True, content=content, raw_json=None, attempts=attempts, model=model
            )
        except Exception as e:
            last_err = str(e)
            if i < retries:
                sleep_s = min_backoff_s + (max_backoff_s - min_backoff_s) * rng.random()
                time.sleep(sleep_s)

    return LLMCallResult(
        ok=False, content="", raw_json=None, error=last_err, attempts=attempts, model=model
    )


# ---------------------------------------------------------------------
# Minimal logging helper
# ---------------------------------------------------------------------
def log_llm_error(prefix: str, result: LLMCallResult) -> None:
    if result.ok:
        return
    msg = f"[{prefix}] LLM call failed after {result.attempts} attempt(s). error={result.error}\n"
    try:
        sys.stderr.write(msg)
    except Exception:
        pass
