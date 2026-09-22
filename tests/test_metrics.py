"""Pure metric functions: Brier, summarize, projection."""
import pytest

from localjev import metrics


def test_brier_multiclass_bounds():
    assert metrics.brier_multiclass({"a": 1.0, "b": 0.0, "c": 0.0}, "a") == 0.0
    assert metrics.brier_multiclass({"a": 0.0, "b": 1.0, "c": 0.0}, "a") == pytest.approx(2.0)
    mid = metrics.brier_multiclass({"a": 0.5, "b": 0.5}, "a")
    assert 0 < mid < 1


def test_brier_multiclass_true_label_absent_counts_as_zero_prob():
    # true label 'z' not in the distribution -> penalized
    assert metrics.brier_multiclass({"a": 1.0}, "z") == pytest.approx(2.0)


def test_brier_binary():
    assert metrics.brier_binary(1.0, True) == 0.0
    assert metrics.brier_binary(0.0, True) == 1.0
    assert metrics.brier_binary(0.5, False) == pytest.approx(0.25)


def test_summarize_accuracy_and_typesafety():
    records = [
        {"correct": True, "brier": 0.1, "valid": True},
        {"correct": False, "brier": 0.9, "valid": True},
        {"correct": True, "brier": 0.2, "valid": False},
    ]
    s = metrics.summarize(records, latency_ms=300.0, output_tokens=6, input_tokens=90, wall_ms=250.0)
    assert s["decisions"] == 3
    assert s["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["invalid_outputs"] == 1
    assert s["type_safety"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["output_tokens_per_decision"] == pytest.approx(2.0)
    assert s["avg_latency_ms"] == pytest.approx(100.0)


def test_project_cost_scales_linearly():
    s = metrics.summarize([{"correct": True, "brier": 0.0, "valid": True}],
                          latency_ms=10.0, output_tokens=2, input_tokens=5)
    proj = metrics.project_cost(s, 1_000_000)
    assert proj["scale"] == 1_000_000
    assert proj["output_tokens"] == 2_000_000  # 2 tokens/decision * 1e6
