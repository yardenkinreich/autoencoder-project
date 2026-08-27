"""
Unit tests for eval/history.py's run-identity scheme.
Run with: PYTHONPATH=src python -m pytest src/test/test_history.py
"""
import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.history import generate_run_id, flatten_results


def test_generate_run_id_deterministic_given_when():
    when = datetime.datetime(2026, 8, 16, 14, 32, 0)
    a = generate_run_id("logs/mae/foo/bar/models/autoencoder.pth", "mae", when)
    b = generate_run_id("logs/mae/foo/bar/models/autoencoder.pth", "mae", when)
    assert a == b


def test_generate_run_id_distinct_across_when():
    ckpt = "logs/mae/foo/bar/models/autoencoder.pth"
    a = generate_run_id(ckpt, "mae", datetime.datetime(2026, 8, 16, 14, 32, 0))
    b = generate_run_id(ckpt, "mae", datetime.datetime(2026, 8, 16, 14, 33, 0))
    assert a != b


def test_generate_run_id_distinct_across_checkpoints():
    when = datetime.datetime(2026, 8, 16, 14, 32, 0)
    a = generate_run_id("logs/mae/foo/models/autoencoder.pth", "mae", when)
    b = generate_run_id("logs/mae/bar/models/autoencoder.pth", "mae", when)
    assert a != b


def test_generate_run_id_filesystem_safe():
    run_id = generate_run_id("logs/mae/foo/bar/models/autoencoder.pth", "mae",
                             datetime.datetime(2026, 8, 16, 14, 32, 0))
    assert "/" not in run_id
    assert run_id == "mae_20260816-143200_" + run_id.split("_")[-1]
    assert len(run_id.split("_")[-1]) == 8


def test_flatten_results_uses_stored_run_id_when_present():
    all_results = {
        "run_id": "mae_20260816-143200_abcd1234",
        "run_timestamp": "2026-08-16T14:32:00",
        "checkpoint": "logs/mae/foo/bar/models/autoencoder.pth",
        "autoencoder_model": "mae",
        "labeled_sets": {},
        "holdout": None,
    }
    row = flatten_results(all_results)
    assert row["run_id"] == "mae_20260816-143200_abcd1234"
    assert row["timestamp"] == "2026-08-16T14:32:00"


def test_flatten_results_falls_back_without_stored_run_id():
    all_results = {
        "checkpoint": "logs/mae/wac100m_sigma100_noband/d3-10_ep50_mask0.75_n25000/models/autoencoder.pth",
        "autoencoder_model": "mae",
        "labeled_sets": {},
        "holdout": None,
    }
    row = flatten_results(all_results)
    assert row["run_id"] == "mae/wac100m_sigma100_noband/d3-10_ep50_mask0.75_n25000"
    assert row["timestamp"]  # some ISO string was generated


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
