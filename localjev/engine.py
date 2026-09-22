"""LocalJev engine: turn LM Studio token log-probabilities into typed,
calibrated System One decisions (Choice / Score / Noul).

Design faithful to TypeSafe's Jev contract (https://docs.typesafe.ai/api.md):
"unstructured state in, typed probabilistic decisions out."

The trick: we never let the model free-write. For a question with N labels we
force it to answer with a single digit 0..N-1, then read the model's
log-probabilities for those digit tokens and softmax them into a real
distribution over the labels. Type-safety is structural — the answer can only
ever be one of the declared labels, so hallucinated categories are impossible.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from . import lmstudio

# Calibration temperature applied to the logits (T>1 softens, T<1 sharpens).
# System One models are trained for calibration; locally we expose a simple
# temperature-scaling knob so the probabilities can be tuned against outcomes.
CALIB_T = float(os.environ.get("LOCALJEV_CALIB_T", "1.0"))

# Digits are single tokens in essentially every BPE/SentencePiece vocab, which
# makes them ideal, low-bias label anchors for the logprob read.
_DIGITS = [str(i) for i in range(10)]

_SYSTEM_PROMPT = (
    "You are a fast, deterministic classification function inside a larger program. "
    "You read STATE and a QUESTION with numbered OPTIONS, then output the single "
    "digit of the best option and nothing else — no words, no punctuation, no "
    "explanation. Output exactly one digit."
)


def _softmax(logits: Dict[str, float], temperature: float) -> Dict[str, float]:
    if not logits:
        return {}
    t = max(temperature, 1e-6)
    scaled = {k: v / t for k, v in logits.items()}
    hi = max(scaled.values())
    exp = {k: math.exp(v - hi) for k, v in scaled.items()}
    z = sum(exp.values()) or 1.0
    return {k: v / z for k, v in exp.items()}


def _entropy(probs: List[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def _distribution_over_labels(
    top: Dict[str, float], n_labels: int, emitted: str
) -> Dict[int, float]:
    """Map the model's top token logprobs onto our label indices 0..n_labels-1."""
    # Collect the logprob of each label's digit token, matching on the stripped
    # token string (LM Studio may return " 1" or "1").
    label_logits: Dict[int, float] = {}
    for tok, lp in top.items():
        s = tok.strip()
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < n_labels:
                # keep the highest logprob seen for this label index
                label_logits[idx] = max(label_logits.get(idx, -1e9), lp)

    if not label_logits:
        # Model emitted something off-menu; fall back to the emitted token if it
        # parses, otherwise a uniform distribution. Type-safety still holds:
        # the returned label is always a declared one.
        s = emitted.strip()
        if s.isdigit() and 0 <= int(s) < n_labels:
            return {i: (1.0 if i == int(s) else 0.0) for i in range(n_labels)}
        return {i: 1.0 / n_labels for i in range(n_labels)}

    probs = _softmax({str(k): v for k, v in label_logits.items()}, CALIB_T)
    # Re-key to ints and fill any missing labels with 0.0 for a complete dist.
    dist = {i: 0.0 for i in range(n_labels)}
    for k, v in probs.items():
        dist[int(k)] = v
    return dist


def _build_prompt(state: str, instructions: str, options: List[str]) -> List[Dict[str, str]]:
    lines = [f"{i}) {opt}" for i, opt in enumerate(options)]
    user = (
        f"STATE:\n{state}\n\n"
        f"QUESTION: {instructions}\n"
        f"OPTIONS:\n" + "\n".join(lines) + "\n\n"
        f"Answer with only the digit (0-{len(options) - 1})."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _confidence(dist_values: List[float]) -> float:
    """Confidence in [0,1]: 1 - normalized entropy. High when mass concentrates."""
    n = len(dist_values)
    if n <= 1:
        return 1.0
    max_ent = math.log(n)
    return round(1.0 - (_entropy(dist_values) / max_ent), 4)


def _option_labels(criteria: Any) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Normalize a question's `criteria` into (keys, descriptions, key->desc).

    Accepts a dict {key: description} (Choice/Noul) or a list [level, ...] (Score).
    """
    if isinstance(criteria, dict):
        keys = list(criteria.keys())
        descs = [f"{k}: {criteria[k]}" if criteria[k] else str(k) for k in keys]
        return keys, descs, {k: str(criteria[k]) for k in keys}
    if isinstance(criteria, list):
        keys = [str(c) for c in criteria]
        return keys, keys, {}
    raise ValueError("criteria must be an object or a list")


# --- Primitive evaluators -------------------------------------------------


def eval_choice(state: str, q: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    keys, descs, _ = _option_labels(q.get("criteria", {}))
    if not 2 <= len(keys) <= 10:
        raise ValueError("choice expects between 2 and 10 options")
    messages = _build_prompt(state, q.get("instructions", "Select the best option."), descs)
    res = lmstudio.first_token_logprobs(messages, model=model)
    dist = _distribution_over_labels(res["top"], len(keys), res["token"])
    probabilities = {keys[i]: round(dist[i], 4) for i in range(len(keys))}
    best = max(range(len(keys)), key=lambda i: dist[i])
    return {
        "type": "choice",
        "choice": keys[best],
        "probabilities": probabilities,
        "confidence": _confidence(list(dist.values())),
        "_usage": res["usage"],
    }


def eval_noul(state: str, q: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    crit = q.get("criteria") or {}
    # Present as a 2-option choice: 0 = false, 1 = true; noul = P(true).
    false_desc = crit.get("false", "The answer is no / false.")
    true_desc = crit.get("true", "The answer is yes / true.")
    options = [f"No — {false_desc}", f"Yes — {true_desc}"]
    messages = _build_prompt(state, q.get("instructions", "Answer the yes/no question."), options)
    res = lmstudio.first_token_logprobs(messages, model=model)
    dist = _distribution_over_labels(res["top"], 2, res["token"])
    p_true = round(dist[1], 4)
    return {
        "type": "noul",
        "noul": p_true,
        "confidence": round(max(p_true, 1.0 - p_true), 4),
        "_usage": res["usage"],
    }


def eval_score(state: str, q: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    keys, descs, _ = _option_labels(q.get("criteria", []))
    if not 2 <= len(keys) <= 10:
        raise ValueError("score expects between 2 and 10 ordered levels")
    instr = q.get("instructions", "Rate against the ordered levels.")
    messages = _build_prompt(state, instr, descs)
    res = lmstudio.first_token_logprobs(messages, model=model)
    dist = _distribution_over_labels(res["top"], len(keys), res["token"])
    # Continuous score = probability-weighted expected level (like Jev's 1.05).
    score = sum(i * dist[i] for i in range(len(keys)))
    return {
        "type": "score",
        "score": round(score, 4),
        "legend": {str(i): keys[i] for i in range(len(keys))},
        "probabilities": {str(i): round(dist[i], 4) for i in range(len(keys))},
        "confidence": _confidence(list(dist.values())),
        "_usage": res["usage"],
    }


_EVALUATORS = {"choice": eval_choice, "noul": eval_noul, "score": eval_score}


def evaluate(state: str, questions: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate every question against the state and return a Jev-shaped response."""
    started = time.perf_counter()
    answers: Dict[str, Any] = {}
    in_tokens = 0
    out_tokens = 0
    used_model = model or lmstudio.resolve_model()

    for name, q in questions.items():
        qtype = q.get("type")
        fn = _EVALUATORS.get(qtype)
        if fn is None:
            raise ValueError(f"question '{name}' has unknown type '{qtype}'")
        ans = fn(state, q, used_model)
        usage = ans.pop("_usage", {}) or {}
        in_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        out_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        answers[name] = ans

    latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
    return {
        "model": used_model,
        "answers": answers,
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
        "latency_ms": latency_ms,
    }
