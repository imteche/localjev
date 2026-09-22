"""Baseline path: how a *normal* LLM app answers the same questions.

We ask the model to free-write a JSON object with its answer and a self-reported
confidence for every question, then parse it. This is the honest, common
alternative to a System One model — and the benchmark measures what it costs:
more output tokens, higher latency, occasional invalid/hallucinated output, and
confidences that don't mean much (worse Brier).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from . import lmstudio

_SYSTEM = (
    "You are a support-ticket classification function. Respond with ONLY a single "
    "JSON object and no other text, following the requested schema exactly."
)


def _labels(criteria: Any) -> List[str]:
    if isinstance(criteria, dict):
        return list(criteria.keys())
    if isinstance(criteria, list):
        return [str(c) for c in criteria]
    return []


def _build_prompt(state: str, questions: Dict[str, Any]) -> List[Dict[str, str]]:
    lines = []
    for name, q in questions.items():
        qtype = q.get("type")
        if qtype == "choice":
            opts = " | ".join(_labels(q.get("criteria", {})))
            lines.append(f'- "{name}": one of [{opts}]  ({q.get("instructions","")})')
        elif qtype == "noul":
            lines.append(f'- "{name}": true or false  ({q.get("instructions","")})')
        elif qtype == "score":
            levels = _labels(q.get("criteria", []))
            enumerated = ", ".join(f"{i}={l}" for i, l in enumerate(levels))
            lines.append(f'- "{name}": integer level [{enumerated}]  ({q.get("instructions","")})')
    schema = "\n".join(lines)
    keys = ", ".join(f'"{k}"' for k in questions)
    user = (
        f"TICKET:\n{state}\n\n"
        f"FIELDS:\n{schema}\n\n"
        f"Return a JSON object whose keys are EXACTLY these field names: {keys}. "
        'Each value must be an object {"answer": <one allowed value>, '
        '"confidence": <0.0-1.0>}. Reply with ONLY the JSON object, no other text.'
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort: pull the first balanced {...} object out of the model's text."""
    # Strip code fences if present.
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _coerce_answer(qtype: str, criteria: Any, raw: Any) -> Tuple[Any, bool]:
    """Return (normalized_answer, valid). valid=False means off-schema output."""
    labels = _labels(criteria)
    if qtype == "choice":
        if isinstance(raw, str) and raw in labels:
            return raw, True
        # tolerate case / whitespace
        for l in labels:
            if isinstance(raw, str) and raw.strip().lower() == l.lower():
                return l, True
        return (labels[0] if labels else None), False
    if qtype == "noul":
        if isinstance(raw, bool):
            return raw, True
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in ("true", "yes", "y", "1"):
                return True, True
            if s in ("false", "no", "n", "0"):
                return False, True
        return False, False
    if qtype == "score":
        n = len(labels)
        try:
            idx = int(raw)
            if 0 <= idx < n:
                return idx, True
        except (TypeError, ValueError):
            if isinstance(raw, str):
                for i, l in enumerate(labels):
                    if raw.strip().lower() == l.lower():
                        return i, True
        return 0, False
    return raw, False


def evaluate(state: str, questions: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    """Free-form baseline. Returns answers with a distribution derived from the
    model's self-reported confidence, plus per-answer validity flags."""
    res = lmstudio.generate(_build_prompt(state, questions), model=model, max_tokens=256)
    parsed = _extract_json(res["text"]) or {}
    answers: Dict[str, Any] = {}

    for name, q in questions.items():
        qtype = q.get("type")
        criteria = q.get("criteria", {} if qtype != "score" else [])
        labels = _labels(criteria)
        entry = parsed.get(name)
        raw_ans = entry.get("answer") if isinstance(entry, dict) else entry
        conf = entry.get("confidence") if isinstance(entry, dict) else None
        try:
            conf = float(conf)
            conf = min(max(conf, 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5
        ans, valid = _coerce_answer(qtype, criteria, raw_ans)
        valid = valid and (entry is not None)

        if qtype == "choice":
            others = [l for l in labels if l != ans]
            spill = (1.0 - conf) / len(others) if others else 0.0
            probs = {l: (conf if l == ans else spill) for l in labels}
            answers[name] = {"type": "choice", "choice": ans, "probabilities": probs,
                             "confidence": conf, "valid": valid}
        elif qtype == "noul":
            p_yes = conf if ans else (1.0 - conf)
            answers[name] = {"type": "noul", "noul": round(p_yes, 4),
                             "confidence": conf, "valid": valid}
        elif qtype == "score":
            n = len(labels)
            probs = {str(i): ((conf if i == ans else (1.0 - conf) / (n - 1)))
                     for i in range(n)}
            answers[name] = {"type": "score", "score": float(ans),
                             "legend": {str(i): labels[i] for i in range(n)},
                             "probabilities": probs, "confidence": conf, "valid": valid}

    return {
        "answers": answers,
        "usage": res["usage"],
        "latency_ms": res["latency_ms"],
        "model": res["model"],
        "raw": res["text"],
    }
