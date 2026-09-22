"""Thin client for LM Studio's OpenAI-compatible local server.

LocalJev never asks the model to *write* a probability. It constrains the model
to emit a single label token and reads the token log-probabilities that LM Studio
returns, turning them into a real distribution. This module is the only place that
talks to LM Studio.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

# LM Studio serves an OpenAI-compatible API here by default (Developer tab -> Start).
BASE_URL = os.environ.get("LOCALJEV_LMSTUDIO_URL", "http://localhost:1234/v1").rstrip("/")
_MODEL_OVERRIDE = os.environ.get("LOCALJEV_MODEL")  # optional; else auto-detected
_HTTP_TIMEOUT = float(os.environ.get("LOCALJEV_TIMEOUT", "60"))

_cached_model: Optional[str] = None


class LMStudioError(RuntimeError):
    pass


def list_models() -> List[str]:
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network
        raise LMStudioError(
            f"Could not reach LM Studio at {BASE_URL}. Is the local server running? ({exc})"
        ) from exc
    return [m["id"] for m in r.json().get("data", [])]


def resolve_model() -> str:
    """Pick a model: env override, else the first model LM Studio has loaded."""
    global _cached_model
    if _MODEL_OVERRIDE:
        return _MODEL_OVERRIDE
    if _cached_model:
        return _cached_model
    models = list_models()
    if not models:
        raise LMStudioError(
            "LM Studio is running but no model is loaded. Load a model in the "
            "Developer tab (or set LOCALJEV_MODEL)."
        )
    _cached_model = models[0]
    return _cached_model


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
    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions", json=payload, timeout=_HTTP_TIMEOUT
        )
        r.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network
        raise LMStudioError(f"LM Studio request failed: {exc}") from exc

    data = r.json()
    choice = data["choices"][0]
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []
    if not content:
        raise LMStudioError(
            "The model did not return token logprobs. Use a model/runtime that "
            "supports logprobs (most llama.cpp GGUF models in LM Studio do)."
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
    }
