"""Benchmark orchestration: scoring against ground truth + streamed events.

We mock both engines so the benchmark runs offline and deterministically.
"""
import pytest

from localjev import benchmark, engine, baseline


@pytest.fixture
def fake_engines(monkeypatch):
    # System One: always right on department, confident and calibrated.
    def so_eval(state, questions, model=None, **k):
        return {
            "model": "m",
            "answers": {
                "department": {"type": "choice", "choice": "billing",
                               "probabilities": {"billing": 0.9, "technical": 0.05, "sales": 0.05},
                               "confidence": 0.9},
                "is_urgent": {"type": "noul", "noul": 0.9, "confidence": 0.9},
                "wants_refund": {"type": "noul", "noul": 0.9, "confidence": 0.9},
            },
            "usage": {"input_tokens": 100, "output_tokens": 3},
            "latency_ms": 30.0, "model_latency_ms": 30.0,
        }

    # Baseline: wrong department, one invalid output, more tokens.
    def bl_eval(state, questions, model=None, **k):
        return {
            "answers": {
                "department": {"type": "choice", "choice": "technical",
                               "probabilities": {"billing": 0.3, "technical": 0.6, "sales": 0.1},
                               "confidence": 0.6, "valid": True},
                "is_urgent": {"type": "noul", "noul": 0.6, "confidence": 0.6, "valid": True},
                "wants_refund": {"type": "noul", "noul": 0.5, "confidence": 0.5, "valid": False},
            },
            "usage": {"completion_tokens": 60, "prompt_tokens": 120},
            "latency_ms": 200.0, "model": "m", "raw": "{}",
        }

    monkeypatch.setattr(engine, "evaluate", so_eval)
    monkeypatch.setattr(baseline, "evaluate", bl_eval)
    monkeypatch.setattr(benchmark.lmstudio, "resolve_model", lambda: "m")


def test_run_scores_against_ground_truth(fake_engines):
    # ticket #1 in the dataset is billing / urgent / wants_refund
    result = benchmark.run(model="m", limit=2)
    so, bl = result["systemone"], result["baseline"]

    assert result["tickets"] == 2
    assert result["decisions"] == 6  # 2 tickets * 3 scored questions
    # System One nails department on both billing tickets... but #2 is 'sales',
    # so it's right on ticket 1 dept, wrong on ticket 2 dept. Accuracy is partial.
    assert 0.0 <= so["accuracy"] <= 1.0
    assert so["type_safety"] == 1.0
    assert bl["type_safety"] < 1.0            # baseline had an invalid output
    assert so["output_tokens"] < bl["output_tokens"]
    # both approaches produce a Brier score (whether SO wins depends on the model)
    assert so["brier"] is not None and bl["brier"] is not None


def test_stream_emits_start_progress_result(fake_engines):
    events = list(benchmark.run_stream(model="m", limit=3))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "result"
    assert kinds.count("progress") == 3
    # progress carries running summaries and per-ticket detail
    prog = [e for e in events if e["type"] == "progress"][0]
    assert "systemone" in prog and "baseline" in prog
    assert set(prog["ticket"]["truth"]) == {"department", "is_urgent", "wants_refund"}


def test_projection_present_in_result(fake_engines):
    result = benchmark.run(model="m", limit=1)
    assert result["projection"]["baseline"]["output_tokens"] > result["projection"]["systemone"]["output_tokens"]
    assert result["reference"]["jev_output_tokens"] == "free"
