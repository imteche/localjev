"""CLI demo: triage a few support tickets through LocalJev and print the
typed, calibrated decisions. Run the server first (see README), then:

    python examples/triage.py
    python examples/triage.py "My card was charged twice and support won't reply!!"
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localjev.sdk import LocalJev, choice, noul, score  # noqa: E402

QUESTIONS = {
    "department": choice(
        "Which team should handle this ticket?",
        {
            "billing": "Payments, invoicing, refunds, chargebacks",
            "technical": "Bugs, outages, API/integration problems",
            "sales": "Pricing, upgrades, new accounts, demos",
        },
    ),
    "frustration": score(
        "How frustrated is the customer?",
        ["Calm", "Mildly annoyed", "Frustrated", "Very angry"],
    ),
    "is_urgent": noul(
        "Does this ticket convey time-sensitive urgency?",
        true="Explicitly time-sensitive or business-blocking",
        false="No urgency expressed",
    ),
    "wants_refund": noul(
        "Is the customer asking for a refund?",
        true="Requests money back / refund / chargeback",
        false="No refund requested",
    ),
}

SAMPLE_TICKETS = [
    "Help! My payouts have been failing for 3 days and I'm losing customers. Fix this now.",
    "Hi, just wondering how I'd upgrade from the Team plan to Business? No rush.",
    "You charged me twice this month. I want a refund immediately or I'm disputing it.",
]

ESCALATE_BELOW = 0.60  # confidence threshold -> route to a human


def render(state: str, result: dict) -> None:
    print("\n" + "=" * 72)
    print(f"TICKET: {state}")
    print(f"  model={result['model']}  latency={result['latency_ms']}ms  "
          f"tokens_in={result['usage']['input_tokens']}")
    ans = result["answers"]

    dep = ans["department"]
    bars = "  ".join(f"{k}:{v:.2f}" for k, v in dep["probabilities"].items())
    flag = "  ⚠ ESCALATE" if dep["confidence"] < ESCALATE_BELOW else ""
    print(f"  → route:       {dep['choice']:<10} conf={dep['confidence']:.2f}  [{bars}]{flag}")

    fr = ans["frustration"]
    level = fr["legend"][str(round(fr["score"]))]
    print(f"  → frustration: {fr['score']:.2f}/{len(fr['legend'])-1} (~{level})  conf={fr['confidence']:.2f}")

    ur = ans["is_urgent"]
    print(f"  → urgent:      {'YES' if ur['noul'] >= 0.5 else 'no':<10} p(yes)={ur['noul']:.2f}")

    rf = ans["wants_refund"]
    print(f"  → refund:      {'YES' if rf['noul'] >= 0.5 else 'no':<10} p(yes)={rf['noul']:.2f}")


def main() -> None:
    jev = LocalJev()
    tickets = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else SAMPLE_TICKETS
    for t in tickets:
        render(t, jev.evaluate(t, QUESTIONS))
    print()


if __name__ == "__main__":
    main()
