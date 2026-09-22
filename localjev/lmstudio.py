"""Thin client for LM Studio's OpenAI-compatible local server.

LocalJev never asks the model to *write* a probability. It constrains the model
to emit a single label token and reads the token log-probabilities that LM Studio
returns, turning them into a real distribution. This module is the only place that
talks to LM Studio.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

# LM Studio serves an OpenAI-compatible API here by default (Developer tab -> Start).
BASE_URL = os.environ.get("LOCALJEV_LMSTUDIO_URL", "http://localhost:1234/v1").rstrip("/")
# LM Studio's native REST API (same host) reports load state + model type, which
# the OpenAI /v1/models list does not. We derive it from BASE_URL's origin.
NATIVE_URL = BASE_URL.rsplit("/v1", 1)[0] + "/api/v0"
_MODEL_OVERRIDE = os.environ.get("LOCALJEV_MODEL")  # optional; else auto-detected
_HTTP_TIMEOUT = float(os.environ.get("LOCALJEV_TIMEOUT", "60"))

_cached_model: Optional[str] = None


class LMStudioError(RuntimeError):
    pass


def _reach_error(exc: Exception) -> "LMStudioError":
    return LMStudioError(
        f"Could not reach LM Studio at {BASE_URL}. Is the local server running? ({exc})"
    )


def model_catalog() -> List[Dict[str, str]]:
    """Return [{id, state, type}] for every model LM Studio knows about.

    Uses LM Studio's native /api/v0/models (which reports `state` and `type`) and
    falls back to the OpenAI /v1/models list (ids only) if that isn't available.
    """
    try:
        r = requests.get(f"{NATIVE_URL}/models", timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        rows = r.json().get("data", [])
        if rows:
            return [
                {"id": m["id"], "state": m.get("state", "unknown"),
                 "type": m.get("type", "llm")}
                for m in rows
            ]
    except (requests.RequestException, ValueError, KeyError):
        pass  # native API unavailable / older LM Studio -> fall back below

    try:
        r = requests.get(f"{BASE_URL}/models", timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network
        raise _reach_error(exc) from exc
    return [{"id": m["id"], "state": "unknown", "type": "llm"}
            for m in r.json().get("data", [])]


def _is_text_llm(m: Dict[str, str]) -> bool:
    # Exclude embeddings; vision (vlm) models usually can't emit answer-token
    # logprobs the way we need, so we de-prioritize them too.
    return m.get("type") not in ("embeddings", "embedding")


def list_models() -> List[str]:
    """Model ids suitable for decisions, loaded ones first, embeddings excluded."""
    catalog = [m for m in model_catalog() if _is_text_llm(m)]
    loaded = [m["id"] for m in catalog if m["state"] == "loaded"]
    others = [m["id"] for m in catalog if m["state"] != "loaded"]
    return loaded + others


def resolve_model() -> str:
    """Pick a sensible default model.

    Priority: env override -> a *loaded* text LLM -> any loaded non-embedding
    model -> first usable model in the catalog. This avoids defaulting to some
    huge not-loaded vision model just because it happens to be listed first.
    """
    global _cached_model
    if _MODEL_OVERRIDE:
        return _MODEL_OVERRIDE
    if _cached_model:
        return _cached_model

    catalog = [m for m in model_catalog() if _is_text_llm(m)]
    if not catalog:
        raise LMStudioError(
            "LM Studio is reachable but has no usable (non-embedding) model. "
            "Load a plain instruct model in the Developer tab (or set LOCALJEV_MODEL)."
        )
    loaded_llms = [m for m in catalog if m["state"] == "loaded" and m["type"] == "llm"]
    loaded_any = [m for m in catalog if m["state"] == "loaded"]
    pick = (loaded_llms or loaded_any or catalog)[0]["id"]
    _cached_model = pick
    return pick


def first_token_logprobs(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    top_logprobs: int = 20,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Generate a single token and return its top candidate log-probabilities.

    Returns a dict with:
      - token:     the argmax token string the model emitted
      - top:       {token_string: logprob} for the top candidates
      - usage:     the raw usage block from LM Studio (input/output tokens)
      - model:     the model id that answered
    """
    model = model or resolve_model()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions", json=payload, timeout=_HTTP_TIMEOUT
        )
        r.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network
        raise LMStudioError(f"LM Studio request failed: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    data = r.json()
    answered_model = data.get("model", model)
    choice = data["choices"][0]
    msg = choice.get("message", {}) or {}
    emitted_text = msg.get("content") or ""
    is_reasoning = ("reasoning_content" in msg) or (
        emitted_text == "" and choice.get("finish_reason") == "length"
    )
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []
    if not content:
        if is_reasoning:
            raise LMStudioError(
                f"Model '{answered_model}' looks like a reasoning/'thinking' model: "
                "its first token goes into the hidden reasoning channel, so there is "
                "no answer token to read a probability from (empty content, no "
                "logprobs). LocalJev needs a plain instruct model. Load a non-thinking "
                "instruct GGUF in LM Studio (e.g. Llama-3.x-Instruct, Qwen2.5-Instruct "
                "non-thinking, Gemma-2-it, Phi-3.5-mini) and set LOCALJEV_MODEL to it, "
                "or disable the model's thinking mode."
            )
        raise LMStudioError(
            f"Model '{answered_model}' did not return token logprobs. Use a model/"
            "runtime that supports logprobs — llama.cpp GGUF models in LM Studio do; "
            "some MLX builds do not. Pin one with LOCALJEV_MODEL."
        )

    first = content[0]
    top: Dict[str, float] = {}
    # Always include the emitted token itself.
    top[first["token"]] = float(first["logprob"])
    for cand in first.get("top_logprobs", []) or []:
        top[cand["token"]] = float(cand["logprob"])

    return {
        "token": first["token"],
        "top": top,
        "usage": data.get("usage", {}),
        "model": data.get("model", model),
        "latency_ms": latency_ms,
    }


def generate(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Free-form completion — the *baseline* path a normal LLM app would take.

    Returns {text, usage, model, latency_ms}. No logprobs, no constraint; the
    caller must parse whatever the model wrote (and cope when it isn't valid).
    """
    model = model or resolve_model()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions", json=payload, timeout=_HTTP_TIMEOUT
        )
        r.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network
        raise LMStudioError(f"LM Studio request failed: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    data = r.json()
    msg = data["choices"][0].get("message", {}) or {}
    return {
        "text": msg.get("content") or "",
        "usage": data.get("usage", {}),
        "model": data.get("model", model),
        "latency_ms": latency_ms,
    }
