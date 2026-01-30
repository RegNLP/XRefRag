#!/usr/bin/env python3
"""
llm_test.py — XRefRAG Azure-only LLM connectivity test

Reads settings from .env (located next to this file) and runs:
  1) Azure OpenAI HTTP check (deployments route)
  2) AzureOpenAI SDK check

This is designed for your Azure-only setup (GPT-5.2 for generate + judge).

Requirements:
  pip install python-dotenv httpx openai

Usage:
  python llm_test.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None


# ----------------------------
# .env loading (forced)
# ----------------------------
def load_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if load_dotenv is None:
        print("ERROR: python-dotenv not installed. Run: pip install python-dotenv")
        sys.exit(1)
    if not env_path.exists():
        print(f"ERROR: .env not found next to llm_test.py: {env_path}")
        sys.exit(1)
    load_dotenv(dotenv_path=env_path, override=True)


def require_env(name: str) -> str:
    v = os.getenv(name)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    if v == "":
        raise RuntimeError(f"Empty env var: {name}")
    return v


def mask(value: str, keep_last: int = 4) -> str:
    if len(value) <= keep_last:
        return "*" * len(value)
    return "*" * (len(value) - keep_last) + value[-keep_last:]


def debug_secret(name: str, value: str) -> None:
    tail = value[-12:] if len(value) > 12 else value
    print(f"{name}: len={len(value)} masked={mask(value)} tail_repr={tail!r}")


def normalize_azure_openai_endpoint(endpoint: str) -> str:
    """
    Ensures endpoint is just: https://<resource>.openai.azure.com
    (no trailing slash, no /openai suffix)
    """
    e = endpoint.strip().rstrip("/")
    if e.endswith("/openai"):
        e = e[:-7]
    return e


def http_post_json(url: str, headers: dict[str, str], payload: dict) -> tuple[int, str]:
    import httpx  # type: ignore

    r = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    return r.status_code, (r.text or "")


# ----------------------------
# Tests
# ----------------------------
def test_azure_http(endpoint: str, api_key: str, api_version: str, deployment: str) -> bool:
    print("\n=== Test 1: Azure OpenAI HTTP (deployments route) ===")
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    payload = {
        "messages": [{"role": "user", "content": 'Return ONLY JSON: {"ok": true}'}],
        "temperature": 0,
        # GPT-5.2 on Azure requires max_completion_tokens (NOT max_tokens)
        "max_completion_tokens": 30,
    }
    headers = {"Content-Type": "application/json", "api-key": api_key}

    status, body = http_post_json(url, headers=headers, payload=payload)
    print(f"URL: {url}")
    print(f"Status: {status}")

    if status != 200:
        print("Body (first 400 chars):", body[:400])
        print("Result: FAIL")
        return False

    print("Result: PASS (HTTP 200)")
    return True


def test_azure_sdk(endpoint: str, api_key: str, api_version: str, deployment: str) -> bool:
    print("\n=== Test 2: AzureOpenAI SDK ===")
    try:
        from openai import AzureOpenAI  # type: ignore

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": 'Return ONLY JSON: {"ok": true}'}],
            temperature=0.0,
            max_completion_tokens=30,
        )

        out = (resp.choices[0].message.content or "").strip()
        print("Output:", out)

        # best-effort validation (do not over-restrict)
        try:
            obj = json.loads(out)
            if isinstance(obj, dict) and obj.get("ok") is True:
                print("Result: PASS (parsed ok:true)")
                return True
        except Exception:
            pass

        print("Result: PASS (SDK call succeeded)")
        return True

    except Exception as e:
        print("Result: FAIL")
        print(f"Error: {type(e).__name__}: {e}")
        return False


def main() -> int:
    load_env()

    provider = os.getenv("XREFRAG_LLM_PROVIDER", "azure")
    if provider.lower() != "azure":
        print(f"ERROR: This llm_test.py is Azure-only, but XREFRAG_LLM_PROVIDER={provider!r}")
        return 1

    endpoint = normalize_azure_openai_endpoint(require_env("AZURE_OPENAI_ENDPOINT"))
    api_key = require_env("AZURE_OPENAI_API_KEY")
    api_version = require_env("AZURE_OPENAI_API_VERSION")
    deployment = require_env("AZURE_OPENAI_DEPLOYMENT_GPT52")

    print("=== XRefRAG Azure LLM Test ===")
    print(f"XREFRAG_ENV: {os.getenv('XREFRAG_ENV', '')}")
    print(f"XREFRAG_LOG_LEVEL: {os.getenv('XREFRAG_LOG_LEVEL', '')}")
    print(f"Provider: {provider}")
    print(f"Endpoint: {endpoint}")
    print(f"API version: {api_version}")
    print(f"Deployment: {deployment}")
    debug_secret("AZURE_OPENAI_API_KEY", api_key)

    ok_http = test_azure_http(endpoint, api_key, api_version, deployment)
    ok_sdk = test_azure_sdk(endpoint, api_key, api_version, deployment)

    print("\n=== Summary ===")
    print(f"Azure HTTP: {'PASS' if ok_http else 'FAIL'}")
    print(f"Azure SDK : {'PASS' if ok_sdk else 'FAIL'}")

    return 0 if (ok_http and ok_sdk) else 1


if __name__ == "__main__":
    raise SystemExit(main())
