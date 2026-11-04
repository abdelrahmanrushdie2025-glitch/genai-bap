"""Thin Gemini REST client with a persistent response cache.

Design choices (Green Cloud):
* lightweight Flash-Lite model by default,
* every answer is cached on disk keyed by a hash of (model, prompt) so a repeated
  question costs zero tokens,
* low temperature + capped output length,
* graceful fallback: if there is no API key or the free-tier quota (HTTP 429) is
  exhausted, callers fall back to deterministic offline logic.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time

import requests

from . import green

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / ".cache" / "llm_cache.json"
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# tried in order if the configured model is not available to the key
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash",
                   "gemini-2.5-flash-lite", "gemini-2.5-flash"]


class LLMUnavailable(RuntimeError):
    pass


def _secret(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:  # Streamlit secrets, when running inside Streamlit
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def api_key() -> str | None:
    return _secret("GEMINI_API_KEY")


def model_name() -> str:
    return _secret("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def is_configured() -> bool:
    return bool(api_key())


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))


def _key(model: str, prompt: str, system: str) -> str:
    return hashlib.sha256(f"{model}\n{system}\n{prompt}".encode()).hexdigest()


def generate(prompt: str, system: str = "", max_tokens: int = 700,
             temperature: float = 0.1, use_cache: bool = True) -> dict:
    """Return {"text", "model", "cached", "prompt_tokens", "output_tokens"}.

    Raises LLMUnavailable when no key is configured or all models fail.
    """
    key = api_key()
    if not key:
        raise LLMUnavailable("No GEMINI_API_KEY configured")

    models = [model_name()] + [m for m in FALLBACK_MODELS if m != model_name()]
    cache = _load_cache() if use_cache else {}
    for m in models:
        ck = _key(m, prompt, system)
        if ck in cache:
            hit = cache[ck]
            green.record(cache_hits=1, tokens_saved_by_cache=hit.get("prompt_tokens", 0) + hit.get("output_tokens", 0))
            return {**hit, "cached": True}

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    last_err = "unknown error"
    for m in models:
        for attempt in range(2):
            try:
                r = requests.post(API.format(model=m), json=body, timeout=60,
                                  headers={"x-goog-api-key": key, "Content-Type": "application/json"})
            except requests.RequestException as e:
                last_err = str(e)
                break
            if r.status_code == 429:  # free-tier rate limit: brief back-off, then give up
                last_err = "Free-tier rate limit reached (HTTP 429). Wait a minute and retry."
                if attempt == 0:
                    time.sleep(4)
                    continue
                raise LLMUnavailable(last_err)
            if r.status_code in (400, 403) and "API key" in r.text:
                raise LLMUnavailable("The Gemini API key was rejected - check the key in Secrets.")
            if r.status_code == 404:  # model not available -> try next
                last_err = f"model {m} not found"
                break
            if not r.ok:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                break
            data = r.json()
            try:
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            except (KeyError, IndexError):
                last_err = f"empty response ({data.get('promptFeedback', '')})"
                break
            usage = data.get("usageMetadata", {})
            result = {
                "text": text.strip(),
                "model": m,
                "prompt_tokens": int(usage.get("promptTokenCount", len(prompt) // 4)),
                "output_tokens": int(usage.get("candidatesTokenCount", len(text) // 4)),
            }
            green.record(llm_calls=1, prompt_tokens=result["prompt_tokens"], output_tokens=result["output_tokens"])
            if use_cache:
                cache = _load_cache()
                cache[_key(m, prompt, system)] = result
                _save_cache(cache)
            return {**result, "cached": False}
    raise LLMUnavailable(last_err)


def clear_cache() -> None:
    CACHE_FILE.unlink(missing_ok=True)
