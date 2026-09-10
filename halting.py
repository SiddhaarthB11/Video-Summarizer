"""The four-level halt cascade.

This is a pure function of the round state -- no I/O, no model calls -- so it
can be unit-tested exhaustively. It mirrors HaltIQ's halt operator H, minus the
`entropy` reason (out of scope for this project).

Priority order (first match wins):

  1. critic   -- the Critic set `approved: true`
  2. no_gain  -- the last PATIENCE cosine distances are all < EPSILON
                 AND the latest Critic-score gain is < DELTA
                 (converged in meaning *and* stopped improving in quality)
  3. failsafe -- reached MAX_ROUNDS
  4. (continue)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import config

CONTINUE = None


@dataclass(frozen=True)
class HaltDecision:
    stop: bool
    reason: str | None  # "critic" | "no_gain" | "failsafe" | None
    detail: str = ""

    def __iter__(self):
        # allows: stop, reason = halt_decision(...)
        yield self.stop
        yield self.reason


def _converged(distances: Sequence[float], epsilon: float, patience: int) -> bool:
    """True when the last `patience` distances all sit below `epsilon`."""
    if len(distances) < patience:
        return False
    return all(d < epsilon for d in distances[-patience:])


def _score_stalled(scores: Sequence[float], delta: float) -> bool:
    """True when the most recent Critic-score improvement is below `delta`.

    A drop in score also counts as stalled (negative gain < delta).
    With fewer than two scores we cannot tell, so treat as *not* stalled.
    """
    if len(scores) < 2:
        return False
    return (scores[-1] - scores[-2]) < delta


def halt_decision(
    round_idx: int,
    distances: Sequence[float],
    scores: Sequence[float],
    critic_approved: bool,
    *,
    max_rounds: int = config.MAX_ROUNDS,
    epsilon: float = config.EPSILON,
    patience: int = config.PATIENCE,
    delta: float = config.DELTA,
) -> HaltDecision:
    """Decide whether the loop should stop after completing `round_idx`.

    Args:
        round_idx:       1-based index of the round just completed.
        distances:       cosine distances d_t, one per round t >= 2 (so this is
                         empty after round 1, length 1 after round 2, ...).
        scores:          Critic quality scores q_t, one per completed round.
        critic_approved: the Critic's `approved` flag for the latest draft.
    """
    if critic_approved:
        return HaltDecision(True, "critic", "Critic approved the draft")

    if _converged(distances, epsilon, patience) and _score_stalled(scores, delta):
        gain = scores[-1] - scores[-2]
        return HaltDecision(
            True,
            "no_gain",
            f"last {patience} distances < {epsilon} and score gain {gain:+.3f} < {delta}",
        )

    if round_idx >= max_rounds:
        return HaltDecision(True, "failsafe", f"reached MAX_ROUNDS ({max_rounds})")

    return HaltDecision(False, None, "")
