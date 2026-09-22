"""LocalJev — a self-hosted System One decision engine backed by LM Studio.

Speaks TypeSafe Jev's /v1/systemone contract (Choice / Score / Noul) but runs
100% locally against any model loaded in LM Studio, deriving real probability
distributions from token log-probabilities instead of asking the model to make
numbers up.
"""
from .engine import evaluate  # noqa: F401

__version__ = "0.1.0"
