"""Baseline (free-form LLM) path: JSON extraction, coercion, validity flags."""
import pytest

from localjev import baseline, lmstudio

QUESTIONS = {
    "department": {"type": "choice", "instructions": "team?",
                   "criteria": {"billing": "", "technical": "", "sales": ""}},
    "is_urgent": {"type": "noul", "instructions": "urgent?", "criteria": {"true": "", "false": ""}},
    "level": {"type": "score", "instructions": "how much?", "criteria": ["low", "mid", "high"]},
}


def _mock_generate(monkeypatch, text):
    monkeypatch.setattr(lmstudio, "resolve_model", lambda: "m")
    monkeypatch.setattr(lmstudio, "generate",
                        lambda *a, **k: {"text": text, "usage": {"completion_tokens": 20, "prompt_tokens": 80},
                                         "model": "m", "latency_ms": 40.0})


def test_extract_json_from_code_fence():
    obj = baseline._extract_json('```json\n{"a": 1}\n```')
    assert obj == {"a": 1}


def test_extract_json_from_surrounding_prose():
    obj = baseline._extract_json('Sure! Here you go: {"x": {"answer": "billing"}} hope that helps')
    assert obj["x"]["answer"] == "billing"


def test_extract_json_returns_none_on_garbage():
    assert baseline._extract_json("no json here") is None


def test_valid_json_parses_all_fields(monkeypatch):
    _mock_generate(monkeypatch, '{"department":{"answer":"billing","confidence":0.9},'
                                '"is_urgent":{"answer":true,"confidence":0.8},'
                                '"level":{"answer":2,"confidence":0.7}}')
    r = baseline.evaluate("ticket", QUESTIONS)
    a = r["answers"]
    assert a["department"]["choice"] == "billing" and a["department"]["valid"]
    assert a["is_urgent"]["noul"] == pytest.approx(0.8) and a["is_urgent"]["valid"]
    assert a["level"]["score"] == 2.0 and a["level"]["valid"]
    # choice distribution puts confidence on the pick, spills the rest
    assert a["department"]["probabilities"]["billing"] == pytest.approx(0.9)
    assert sum(a["department"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_off_menu_choice_is_marked_invalid(monkeypatch):
    _mock_generate(monkeypatch, '{"department":{"answer":"legal","confidence":0.9},'
                                '"is_urgent":{"answer":"maybe","confidence":0.5},'
                                '"level":{"answer":9,"confidence":0.5}}')
    r = baseline.evaluate("ticket", QUESTIONS)
    a = r["answers"]
    assert a["department"]["valid"] is False   # 'legal' not an option
    assert a["is_urgent"]["valid"] is False     # 'maybe' not yes/no
    assert a["level"]["valid"] is False         # 9 out of range


def test_unparseable_output_is_invalid_but_does_not_crash(monkeypatch):
    _mock_generate(monkeypatch, "I think it's billing, probably urgent.")
    r = baseline.evaluate("ticket", QUESTIONS)
    for name in QUESTIONS:
        assert r["answers"][name]["valid"] is False
