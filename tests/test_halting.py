"""Exhaustive tests for the four-level halt cascade (halting.halt_decision)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from halting import halt_decision  # noqa: E402

# defaults under test: MAX_ROUNDS=5, EPSILON=0.05, PATIENCE=2, DELTA=0.02


def test_continue_when_nothing_fires():
    d = halt_decision(2, distances=[0.4], scores=[0.5, 0.7], critic_approved=False)
    assert d.stop is False
    assert d.reason is None


def test_critic_approval_has_top_priority():
    # distances nowhere near converged, but the Critic approves
    d = halt_decision(2, distances=[0.9], scores=[0.5, 0.55], critic_approved=True)
    assert d.stop is True
    assert d.reason == "critic"


def test_critic_beats_failsafe():
    d = halt_decision(5, distances=[0.9, 0.9, 0.9, 0.9], scores=[0.4, 0.4, 0.4, 0.4, 0.4],
                      critic_approved=True)
    assert d.reason == "critic"


def test_no_gain_fires_when_converged_and_score_plateaued():
    # last 2 distances < 0.05, score gain 0.90 -> 0.905 = 0.005 < DELTA
    d = halt_decision(3, distances=[0.2, 0.03, 0.01], scores=[0.7, 0.90, 0.905],
                      critic_approved=False)
    assert d.stop is True
    assert d.reason == "no_gain"


def test_no_gain_suppressed_when_a_recent_distance_exceeds_epsilon():
    # only the very last distance is small; the one before it is not
    d = halt_decision(3, distances=[0.2, 0.09, 0.01], scores=[0.7, 0.90, 0.905],
                      critic_approved=False)
    assert d.stop is False


def test_no_gain_suppressed_when_score_still_improving():
    # distances converged, but score jumped 0.70 -> 0.85 (>= DELTA)
    d = halt_decision(3, distances=[0.2, 0.02, 0.01], scores=[0.6, 0.70, 0.85],
                      critic_approved=False)
    assert d.stop is False


def test_no_gain_counts_a_score_drop_as_stalled():
    d = halt_decision(3, distances=[0.2, 0.02, 0.01], scores=[0.8, 0.90, 0.88],
                      critic_approved=False)
    assert d.stop is True
    assert d.reason == "no_gain"


def test_no_gain_needs_patience_worth_of_distances():
    # only one distance recorded so far -> cannot be "converged for PATIENCE rounds"
    d = halt_decision(2, distances=[0.01], scores=[0.9, 0.905], critic_approved=False)
    assert d.stop is False


def test_failsafe_fires_at_max_rounds():
    d = halt_decision(5, distances=[0.5, 0.5, 0.5, 0.5], scores=[0.4, 0.4, 0.4, 0.4, 0.4],
                      critic_approved=False)
    assert d.stop is True
    assert d.reason == "failsafe"


def test_no_gain_beats_failsafe_at_max_rounds():
    d = halt_decision(5, distances=[0.5, 0.5, 0.01, 0.01], scores=[0.4, 0.5, 0.8, 0.9, 0.905],
                      critic_approved=False)
    assert d.reason == "no_gain"


def test_unpacking_interface():
    stop, reason = halt_decision(2, [0.9], [0.5, 0.5], False)
    assert stop is False and reason is None


def test_custom_thresholds_are_respected():
    d = halt_decision(2, distances=[0.2, 0.2], scores=[0.5, 0.5], critic_approved=False,
                      epsilon=0.5, patience=2, delta=0.1)
    assert d.stop is True
    assert d.reason == "no_gain"
