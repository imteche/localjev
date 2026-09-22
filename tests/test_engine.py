"""Engine math: logprobs -> calibrated Choice / Score / Noul decisions."""
import math

import pytest

from localjev import engine
from tests.conftest import probs


def test_choice_recovers_distribution_from_logprobs(patch_backend):
    patch_backend(probs(0.80, 0.15, 0.05))  # billing / technical / sales
    q = {"type": "choice", "instructions": "Which team?",
         "criteria": {"billing": "pay", "technical": "bug", "sales": "price"}}
    r = engine.eval_choice("payouts failing", q)

    assert r["type"] == "choice"
    assert r["choice"] == "billing"
    assert r["probabilities"]["billing"] == pytest.approx(0.80, abs=0.02)
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert r["probabilities"]["billing"] > r["probabilities"]["technical"] > r["probabilities"]["sales"]
    assert 0.0 <= r["confidence"] <= 1.0


def test_choice_argmax_flips_with_distribution(patch_backend):
    patch_backend(probs(0.10, 0.85, 0.05))  # now technical dominates
    q = {"type": "choice", "instructions": "Which team?",
         "criteria": {"billing": "pay", "technical": "bug", "sales": "price"}}
    assert engine.eval_choice("API 500s in prod", q)["choice"] == "technical"


def test_score_is_probability_weighted_expected_level(patch_backend):
    patch_backend(probs(0.02, 0.08, 0.70, 0.20))
    q = {"type": "score", "instructions": "How frustrated?",
         "criteria": ["Calm", "Mild", "Frustrated", "Very angry"]}
    r = engine.eval_score("this is unacceptable", q)

    expected = 0 * 0.02 + 1 * 0.08 + 2 * 0.70 + 3 * 0.20  # 2.08
    assert r["score"] == pytest.approx(expected, abs=0.03)
    assert r["legend"] == {"0": "Calm", "1": "Mild", "2": "Frustrated", "3": "Very angry"}
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_noul_is_probability_of_yes(patch_backend):
    patch_backend({"0": math.log(0.10), "1": math.log(0.90)})  # 1 = yes/true
    q = {"type": "noul", "instructions": "Urgent?",
         "criteria": {"true": "time-sensitive", "false": "not"}}
    r = engine.eval_noul("fix this NOW", q)

    assert r["noul"] == pytest.approx(0.90, abs=0.02)
    assert r["confidence"] == pytest.approx(0.90, abs=0.02)


def test_confidence_higher_when_mass_concentrates(patch_backend):
    patch_backend(probs(0.98, 0.01, 0.01))
    q = {"type": "choice", "instructions": "?",
         "criteria": {"a": "", "b": "", "c": ""}}
    sharp = engine.eval_choice("s", q)["confidence"]

    patch_backend(probs(0.34, 0.33, 0.33))
    flat = engine.eval_choice("s", q)["confidence"]

    assert sharp > flat
    assert flat == pytest.approx(0.0, abs=0.02)


def test_off_menu_output_still_returns_a_declared_label(patch_backend):
    # Model emits a non-digit token -> type-safety must still hold.
    patch_backend({"hello": math.log(0.9), "world": math.log(0.1)}, emitted="hello")
    q = {"type": "choice", "instructions": "?",
         "criteria": {"a": "", "b": "", "c": ""}}
    r = engine.eval_choice("s", q)

    assert r["choice"] in {"a", "b", "c"}
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("qtype,n", [("choice", 1), ("score", 11)])
def test_rejects_out_of_range_option_counts(patch_backend, qtype, n):
    patch_backend(probs(0.5, 0.5))
    crit = {str(i): "" for i in range(n)} if qtype == "choice" else [str(i) for i in range(n)]
    q = {"type": qtype, "instructions": "?", "criteria": crit}
    with pytest.raises(ValueError):
        (engine.eval_choice if qtype == "choice" else engine.eval_score)("s", q)


def test_evaluate_threads_model_through_to_backend(monkeypatch):
    # Regression: the request-level `model` must reach first_token_logprobs,
    # not be silently dropped in favour of the first loaded model.
    from localjev import lmstudio
    seen = []

    def fake(messages, model=None, **kwargs):
        seen.append(model)
        import math
        return {"token": "1", "top": {"0": math.log(0.4), "1": math.log(0.6)},
                "usage": {}, "model": model}

    monkeypatch.setattr(lmstudio, "resolve_model", lambda: "default-model")
    monkeypatch.setattr(lmstudio, "first_token_logprobs", fake)
    res = engine.evaluate("s", {
        "q": {"type": "noul", "instructions": "?", "criteria": {"true": "y", "false": "n"}},
    }, model="pinned-model")
    assert res["model"] == "pinned-model"
    assert seen == ["pinned-model"]  # not "default-model"


def test_evaluate_returns_jev_shaped_response(patch_backend):
    patch_backend(probs(0.7, 0.3), usage={"prompt_tokens": 120, "completion_tokens": 1})
    res = engine.evaluate("some ticket", {
        "refund": {"type": "noul", "instructions": "refund?",
                   "criteria": {"true": "yes", "false": "no"}},
    })
    assert set(res.keys()) == {"model", "answers", "usage", "latency_ms"}
    assert res["usage"]["input_tokens"] == 120
    assert res["answers"]["refund"]["type"] == "noul"
    assert isinstance(res["latency_ms"], float)
