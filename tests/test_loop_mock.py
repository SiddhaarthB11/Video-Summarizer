"""End-to-end smoke test of the round loop using the deterministic mock backend.

No API key, no real video. Verifies the loop wiring, the log schema, the
halted-vs-baseline round difference, and the token accounting.
"""

import os
import sys

os.environ["HALT_VIDEO_MOCK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib  # noqa: E402

import config  # noqa: E402
importlib.reload(config)
assert config.MOCK is True

from loop import run_video  # noqa: E402


def test_halted_run_stops_early_via_no_gain():
    res = run_video("<mock>", halting=True)
    assert res["policy"] == "halted"
    assert res["backend"] == "mock"
    # mock drafts converge by round 3 and the Critic never approves
    assert res["stop_reason"] == "no_gain"
    assert res["rounds_used"] < config.MAX_ROUNDS
    assert res["final_summary"]


def test_baseline_runs_full_schedule():
    res = run_video("<mock>", halting=False)
    assert res["policy"] == "baseline"
    assert res["rounds_used"] == config.MAX_ROUNDS
    assert res["stop_reason"] == "baseline_cap"


def test_halted_is_cheaper_than_baseline():
    h = run_video("<mock>", halting=True)
    b = run_video("<mock>", halting=False)
    assert h["usage"]["total_tokens"] < b["usage"]["total_tokens"]
    assert h["usage"]["calls"] < b["usage"]["calls"]


def test_round_log_schema():
    res = run_video("<mock>", halting=True)
    r1 = res["rounds"][0]
    for key in [
        "round", "draft", "retrieved", "distance", "score", "critique",
        "approved", "would_halt", "cumulative_calls", "cumulative_tokens",
    ]:
        assert key in r1
    assert res["rounds"][0]["distance"] is None  # no d_1
    assert res["rounds"][1]["distance"] is not None
    assert len(r1["retrieved"]) == config.TOP_K


def test_quality_delta_is_small():
    h = run_video("<mock>", halting=True)
    b = run_video("<mock>", halting=False)
    assert abs(h["final_score"] - b["final_score"]) <= 0.05
