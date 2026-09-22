"""Unit tests for the LM Studio client: response parsing + actionable errors.

These mock `requests.post`/`requests.get` so no server is needed. They cover the
two real-world failures we hit: a reasoning model (empty content, null logprobs)
and a runtime that omits logprobs.
"""
import pytest

from localjev import lmstudio


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _install_post(monkeypatch, payload):
    monkeypatch.setattr(lmstudio.requests, "post", lambda *a, **k: FakeResp(payload))


def test_parses_logprobs_from_a_normal_model(monkeypatch):
    _install_post(monkeypatch, {
        "model": "bonsai-1.7b",
        "choices": [{
            "message": {"content": "1"},
            "finish_reason": "length",
            "logprobs": {"content": [{
                "token": "1", "logprob": -0.01,
                "top_logprobs": [{"token": "1", "logprob": -0.01},
                                 {"token": "0", "logprob": -6.0}],
            }]},
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1},
    })
    res = lmstudio.first_token_logprobs([{"role": "user", "content": "x"}], model="bonsai-1.7b")
    assert res["token"] == "1"
    assert res["top"]["1"] == pytest.approx(-0.01)
    assert res["top"]["0"] == pytest.approx(-6.0)
    assert res["model"] == "bonsai-1.7b"


def test_reasoning_model_gives_actionable_error(monkeypatch):
    # Empty content + reasoning_content + null logprobs -> the muse-glimmer case.
    _install_post(monkeypatch, {
        "model": "muse-glimmer-30b",
        "choices": [{
            "message": {"content": "", "reasoning_content": ""},
            "finish_reason": "length",
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 64, "completion_tokens": 1},
    })
    with pytest.raises(lmstudio.LMStudioError) as exc:
        lmstudio.first_token_logprobs([{"role": "user", "content": "x"}], model="muse-glimmer-30b")
    msg = str(exc.value)
    assert "reasoning" in msg.lower()
    assert "muse-glimmer-30b" in msg


def test_missing_logprobs_gives_actionable_error(monkeypatch):
    _install_post(monkeypatch, {
        "model": "some-mlx-model",
        "choices": [{"message": {"content": "1"}, "finish_reason": "stop", "logprobs": None}],
        "usage": {},
    })
    with pytest.raises(lmstudio.LMStudioError) as exc:
        lmstudio.first_token_logprobs([{"role": "user", "content": "x"}], model="some-mlx-model")
    assert "logprobs" in str(exc.value).lower()


def test_resolve_prefers_loaded_llm_over_first_listed(monkeypatch):
    # Mirrors the real bug: a not-loaded vision model is listed first, but a
    # loaded text LLM should win.
    catalog = [
        {"id": "muse-glimmer-30b", "state": "not-loaded", "type": "vlm"},
        {"id": "bonsai-8b", "state": "loaded", "type": "llm"},
        {"id": "bonsai-1.7b", "state": "loaded", "type": "llm"},
        {"id": "text-embed", "state": "loaded", "type": "embeddings"},
    ]
    monkeypatch.setattr(lmstudio, "model_catalog", lambda: catalog)
    monkeypatch.setattr(lmstudio, "_cached_model", None, raising=False)
    monkeypatch.setattr(lmstudio, "_MODEL_OVERRIDE", None, raising=False)
    assert lmstudio.resolve_model() == "bonsai-8b"


def test_list_models_puts_loaded_first_and_drops_embeddings(monkeypatch):
    catalog = [
        {"id": "muse-glimmer-30b", "state": "not-loaded", "type": "vlm"},
        {"id": "bonsai-8b", "state": "loaded", "type": "llm"},
        {"id": "text-embed", "state": "loaded", "type": "embeddings"},
    ]
    monkeypatch.setattr(lmstudio, "model_catalog", lambda: catalog)
    ids = lmstudio.list_models()
    assert ids[0] == "bonsai-8b"          # loaded first
    assert "muse-glimmer-30b" in ids      # still listed
    assert "text-embed" not in ids        # embeddings excluded


def test_env_override_beats_everything(monkeypatch):
    monkeypatch.setattr(lmstudio, "_cached_model", None, raising=False)
    monkeypatch.setattr(lmstudio, "_MODEL_OVERRIDE", "pinned", raising=False)
    assert lmstudio.resolve_model() == "pinned"


def test_forwards_model_in_request_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["model"] = json.get("model")
        captured["logprobs"] = json.get("logprobs")
        return FakeResp({
            "model": json.get("model"),
            "choices": [{"message": {"content": "0"}, "finish_reason": "length",
                         "logprobs": {"content": [{"token": "0", "logprob": -0.1, "top_logprobs": []}]}}],
            "usage": {},
        })

    monkeypatch.setattr(lmstudio.requests, "post", fake_post)
    lmstudio.first_token_logprobs([{"role": "user", "content": "x"}], model="pinned-model")
    assert captured["model"] == "pinned-model"
    assert captured["logprobs"] is True
