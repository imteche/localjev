"""A tiny client SDK mirroring the ergonomics of TypeSafe's Jev SDK.

    from localjev.sdk import LocalJev, choice, score, noul

    jev = LocalJev()  # talks to your LocalJev server (default http://localhost:8000)
    result = jev.evaluate(
        state="Help! My payouts have been failing for 3 days.",
        questions={
            "department": choice("Which team should handle this?", {
                "billing":   "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales":     "Pricing, upgrades, new accounts",
            }),
            "frustration": score("How frustrated is the customer?",
                                 ["Calm", "Frustrated", "Very angry"]),
            "is_urgent":  noul("Does this convey urgency?",
                               true="Explicitly time-sensitive",
                               false="No urgency expressed"),
        },
    )
    print(result["answers"]["department"]["choice"])       # -> "billing"
    print(result["answers"]["department"]["probabilities"]) # -> {...}
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


def choice(instructions: str, criteria: Dict[str, str]) -> Dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: List[str]) -> Dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def noul(instructions: str, true: str = "", false: str = "") -> Dict[str, Any]:
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": {"true": true, "false": false},
    }


class LocalJev:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (
            base_url or os.environ.get("LOCALJEV_SERVER", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = timeout

    def evaluate(
        self,
        state: str,
        questions: Dict[str, Any],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"state": state, "questions": questions}
        if model:
            payload["model"] = model
        r = requests.post(
            f"{self.base_url}/v1/systemone", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()
