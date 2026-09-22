"""Shared fixtures: a fake LM Studio logprob backend so tests need no server."""
import math

import pytest

from localjev import engine, lmstudio


def fake_backend(logprob_by_digit, emitted=None, usage=None):
    """Build a stand-in for lmstudio.first_token_logprobs.

    `logprob_by_digit` maps a digit string ("0","1",...) to a natural-log prob.
    """
    def _fake(messages, **kwargs):
        tok = emitted or max(logprob_by_digit, key=logprob_by_digit.get)
        return {
            "token": tok,
            "top": dict(logprob_by_digit),
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 1},
            "model": "fake-model",
        }
    return _fake


@pytest.fixture
def patch_backend(monkeypatch):
    """Return a helper that installs a fake backend for the duration of a test."""
    def _install(logprob_by_digit, emitted=None, usage=None):
        monkeypatch.setattr(lmstudio, "resolve_model", lambda: "fake-model")
        monkeypatch.setattr(
            lmstudio, "first_token_logprobs",
            fake_backend(logprob_by_digit, emitted, usage),
        )
    return _install


def probs(*ps):
    """Convenience: turn plain probabilities into a digit->logprob dict."""
    return {str(i): math.log(p) for i, p in enumerate(ps)}
