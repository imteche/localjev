"""Head-to-head benchmark: System One (LocalJev) vs a free-form LLM baseline,
on the same LM Studio model, over a labeled ticket dataset.

Measures the things the JEV pitch is about: accuracy, output-token cost, latency,
type-safety, and calibration (Brier). Exposes a streaming generator so the
dashboard can fill in live, and a one-shot `run()` for the CLI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from . import baseline, engine, lmstudio, metrics

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.jsonl"

# The scored questions (each maps to a ground-truth field in the dataset).
BENCH_QUESTIONS: Dict[str, Any] = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "Payments, invoicing, refunds, chargebacks",
            "technical": "Bugs, outages, API/integration problems",
            "sales": "Pricing, upgrades, demos, new accounts",
        },
    },
    "is_urgent": {
        "type": "noul",
        "instructions": "Is this ticket time-sensitive / business-blocking?",
        "criteria": {"true": "urgent or blocking", "false": "no urgency"},
    },
    "wants_refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?",
        "criteria": {"true": "requests money back", "false": "no refund"},
    },
}

# How each question is scored against the dataset row.
_TRUTH_FIELDS = {"department": "department", "is_urgent": "is_urgent", "wants_refund": "wants_refund"}

# Reference figures for a hosted System One decision, from TypeSafe's blog.
REFERENCE = {
    "jev_latency_ms": "70–500",
    "jev_cost_per_decision_usd": 0.0004,
    "jev_output_tokens": "free",
    "source": "https://typesafe.ai/blog/introducing-system-one-models-and-jev",
}


def load_dataset(limit: Optional[int] = None, path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def _score_answer(name: str, ans: Dict[str, Any], truth_row: Dict[str, Any]) -> Dict[str, Any]:
    field = _TRUTH_FIELDS[name]
    truth = truth_row[field]
    if ans["type"] == "choice":
        pred = ans["choice"]
        return {
            "question": name, "predicted": pred, "truth": truth,
            "correct": pred == truth,
            "brier": metrics.brier_multiclass(ans["probabilities"], truth),
            "valid": ans.get("valid", True),
        }
    # noul
    p_yes = ans["noul"]
    pred = p_yes >= 0.5
    return {
        "question": name, "predicted": pred, "truth": bool(truth),
        "correct": pred == bool(truth),
        "brier": metrics.brier_binary(p_yes, bool(truth)),
        "valid": ans.get("valid", True),
    }


def _accumulate(acc: Dict[str, Any], result: Dict[str, Any], truth_row: Dict[str, Any],
                out_tokens: int, in_tokens: int, model_latency: float, wall_ms: float) -> None:
    for name, ans in result["answers"].items():
        if name in _TRUTH_FIELDS:
            acc["records"].append(_score_answer(name, ans, truth_row))
    acc["output_tokens"] += out_tokens
    acc["input_tokens"] += in_tokens
    acc["model_latency_ms"] += model_latency
    acc["wall_ms"] += wall_ms


def _summary(acc: Dict[str, Any]) -> Dict[str, Any]:
    return metrics.summarize(
        acc["records"],
        latency_ms=acc["model_latency_ms"],
        output_tokens=acc["output_tokens"],
        input_tokens=acc["input_tokens"],
        wall_ms=acc["wall_ms"],
    )


def run_stream(model: Optional[str] = None, limit: Optional[int] = None,
               projection_scale: int = 1_000_000) -> Iterator[Dict[str, Any]]:
    """Yield progress events per ticket, then a final result event."""
    rows = load_dataset(limit)
    used_model = model or lmstudio.resolve_model()
    so = {"records": [], "output_tokens": 0, "input_tokens": 0, "model_latency_ms": 0.0, "wall_ms": 0.0}
    bl = {"records": [], "output_tokens": 0, "input_tokens": 0, "model_latency_ms": 0.0, "wall_ms": 0.0}

    yield {"type": "start", "model": used_model, "tickets": len(rows),
           "decisions": len(rows) * len(_TRUTH_FIELDS)}

    for i, row in enumerate(rows, 1):
        state = row["text"]

        t0 = time.perf_counter()
        so_res = engine.evaluate(state, BENCH_QUESTIONS, model=used_model)
        so_wall = (time.perf_counter() - t0) * 1000.0
        _accumulate(so, so_res, row,
                    out_tokens=so_res["usage"]["output_tokens"],
                    in_tokens=so_res["usage"]["input_tokens"],
                    model_latency=so_res.get("model_latency_ms", so_wall),
                    wall_ms=so_wall)

        t0 = time.perf_counter()
        bl_res = baseline.evaluate(state, BENCH_QUESTIONS, model=used_model)
        bl_wall = (time.perf_counter() - t0) * 1000.0
        _accumulate(bl, bl_res, row,
                    out_tokens=int(bl_res["usage"].get("completion_tokens", 0) or 0),
                    in_tokens=int(bl_res["usage"].get("prompt_tokens", 0) or 0),
                    model_latency=bl_res["latency_ms"], wall_ms=bl_wall)

        yield {
            "type": "progress", "done": i, "total": len(rows),
            "ticket": {"id": row["id"], "text": state,
                       "truth": {k: row[v] for k, v in _TRUTH_FIELDS.items()},
                       "systemone": {"department": so_res["answers"]["department"]["choice"]},
                       "baseline": {"department": bl_res["answers"]["department"]["choice"]}},
            "systemone": _summary(so), "baseline": _summary(bl),
        }

    so_sum, bl_sum = _summary(so), _summary(bl)
    yield {
        "type": "result",
        "model": used_model,
        "tickets": len(rows),
        "decisions": len(so["records"]),
        "systemone": so_sum,
        "baseline": bl_sum,
        "projection": {
            "systemone": metrics.project_cost(so_sum, projection_scale),
            "baseline": metrics.project_cost(bl_sum, projection_scale),
        },
        "reference": REFERENCE,
    }


def run(model: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    final = {}
    for event in run_stream(model, limit):
        if event["type"] == "result":
            final = event
    return final
