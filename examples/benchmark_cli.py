"""CLI benchmark: System One vs a free-form LLM on the same LM Studio model.

    python examples/benchmark_cli.py                 # whole dataset, default model
    python examples/benchmark_cli.py --model bonsai-8b --limit 8

Prints a scorecard and the projected output-token volume at 1M decisions.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localjev import benchmark  # noqa: E402


def _fmt(s, key, suffix=""):
    v = s.get(key)
    return "—" if v is None else f"{v}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="LM Studio model id (default: auto)")
    ap.add_argument("--limit", type=int, default=None, help="number of tickets")
    args = ap.parse_args()

    print(f"Running benchmark (model={args.model or 'auto'}, limit={args.limit or 'all'})…\n")
    last = None
    for ev in benchmark.run_stream(args.model, args.limit):
        if ev["type"] == "start":
            print(f"model = {ev['model']}  ·  {ev['tickets']} tickets  ·  {ev['decisions']} decisions\n")
        elif ev["type"] == "progress":
            print(f"  [{ev['done']:>2}/{ev['total']}] ticket #{ev['ticket']['id']}", end="\r")
        elif ev["type"] == "error":
            print("\nERROR:", ev["detail"]); return
        elif ev["type"] == "result":
            last = ev

    if not last:
        return
    so, bl = last["systemone"], last["baseline"]
    rows = [
        ("Accuracy vs truth", _fmt(so, "accuracy"), _fmt(bl, "accuracy")),
        ("Output tokens/decision", _fmt(so, "output_tokens_per_decision"), _fmt(bl, "output_tokens_per_decision")),
        ("Calibration (Brier ↓)", _fmt(so, "brier"), _fmt(bl, "brier")),
        ("Type-safe outputs", _fmt(so, "type_safety"), _fmt(bl, "type_safety")),
        ("Total wall time", _fmt(so, "wall_ms", " ms"), _fmt(bl, "wall_ms", " ms")),
        ("Output tokens (total)", _fmt(so, "output_tokens"), _fmt(bl, "output_tokens")),
    ]
    print("\n\n" + " " * 26 + "System One      Free-form LLM")
    print("  " + "-" * 54)
    for name, a, b in rows:
        print(f"  {name:<24}{str(a):>10}     {str(b):>12}")
    p = last["projection"]
    print("\n  Projected @ 1,000,000 decisions:")
    print(f"    System One : {p['systemone']['output_tokens']:>14,} output tokens")
    print(f"    Free-form  : {p['baseline']['output_tokens']:>14,} output tokens")
    ref = last["reference"]
    print(f"\n  Reference (hosted Jev): {ref['jev_latency_ms']} ms, "
          f"≈${ref['jev_cost_per_decision_usd']}/decision, output tokens {ref['jev_output_tokens']}.")


if __name__ == "__main__":
    main()
