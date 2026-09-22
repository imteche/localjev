"""Pure scoring functions for the benchmark — no I/O, so they're easy to test.

We score two things the JEV pitch cares about:
  * accuracy   — did the decision match ground truth?
  * calibration— do the probabilities mean anything? (Brier score; lower better)
plus the operational metrics (latency, output tokens, type-safety) that the
benchmark collects directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def brier_multiclass(probs: Dict[str, float], true_label: str) -> float:
    """Sum_k (p_k - y_k)^2 over the class distribution. Range [0, 2], lower better.

    A perfectly confident correct answer scores 0; a confident wrong answer ~2.
    """
    total = 0.0
    seen_true = False
    for label, p in probs.items():
        y = 1.0 if label == true_label else 0.0
        seen_true = seen_true or (y == 1.0)
        total += (p - y) ** 2
    if not seen_true:  # true label absent from the distribution -> counts as p=0
        total += 1.0
    return total


def brier_binary(p_yes: float, truth: bool) -> float:
    """(p - y)^2 for a yes/no decision. Range [0, 1], lower better."""
    y = 1.0 if truth else 0.0
    return (p_yes - y) ** 2


def summarize(records: List[Dict[str, Any]],
              latency_ms: float,
              output_tokens: int,
              input_tokens: int,
              wall_ms: Optional[float] = None) -> Dict[str, Any]:
    """Aggregate per-decision `records` into a scorecard for one approach.

    Each record: {correct: bool, brier: float, valid: bool}. `latency_ms` is the
    summed model-call latency; `wall_ms` (optional) is measured wall time.
    """
    n = len(records)
    scored = [r for r in records if r.get("correct") is not None]
    correct = sum(1 for r in scored if r["correct"])
    valid = sum(1 for r in records if r.get("valid", True))
    briers = [r["brier"] for r in records if r.get("brier") is not None]
    return {
        "decisions": n,
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "correct": correct,
        "scored": len(scored),
        "valid_outputs": valid,
        "invalid_outputs": n - valid,
        "type_safety": round(valid / n, 4) if n else None,
        "brier": round(sum(briers) / len(briers), 4) if briers else None,
        "output_tokens": output_tokens,
        "input_tokens": input_tokens,
        "avg_latency_ms": round(latency_ms / n, 1) if n else None,
        "total_latency_ms": round(latency_ms, 1),
        "wall_ms": round(wall_ms, 1) if wall_ms is not None else None,
        "output_tokens_per_decision": round(output_tokens / n, 2) if n else None,
    }


def project_cost(summary: Dict[str, Any], scale: int) -> Dict[str, Any]:
    """Extrapolate output-token volume and wall time to `scale` decisions."""
    n = summary["decisions"] or 1
    per_dec_tokens = (summary["output_tokens"] or 0) / n
    per_dec_ms = (summary["total_latency_ms"] or 0) / n
    return {
        "scale": scale,
        "output_tokens": int(per_dec_tokens * scale),
        "hours": round((per_dec_ms * scale) / 3_600_000.0, 2),
    }
