"""Central configuration for the semantic-halting video-summarization loop.

All tunable constants live here so the loop, the baseline, and the tests read
the same values. See README.md for the rationale behind each default.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:  # noqa: BLE001 - dotenv is optional
    pass

# --- Models -----------------------------------------------------------------

# Gemini model used for the Writer, the Critic, and the round-1 "fresh eyes"
# description. gemini-2.5-flash is fast, cheap, and video-capable.
MODEL = os.environ.get("HALT_VIDEO_MODEL", "gemini-2.5-flash")

# Local sentence-embedding model (sentence-transformers). Runs on CPU, no API
# cost. 384-dimensional output, matching HaltIQ's embedding map.
EMBED_MODEL = os.environ.get("HALT_VIDEO_EMBED_MODEL", "all-MiniLM-L6-v2")

# --- Halting cascade constants --------------------------------------------

MAX_ROUNDS = 5      # hard failsafe cap on Writer/Critic rounds
EPSILON = 0.05      # cosine-distance threshold: below this = "converged"
PATIENCE = 2        # consecutive converged rounds required before halting
DELTA = 0.02        # minimum Critic-score gain that still counts as "improving"

# --- Retrieval -------------------------------------------------------------

TOP_K = 3           # reference summaries retrieved per round

# If True, one extra call per video writes a small set of reference summaries
# tailored to that clip's subject (from the fresh-eyes description) and adds
# them to the fixed library for that run only. This is what makes retrieval
# useful on subjects the hand-written 25 don't cover -- a clip that's nothing
# like "dog on grass" still gets close, relevant style examples. They're ADDED
# to the static library, never replace it, and every retrieved example is
# tagged static/generated wherever it's shown, since these examples are
# written by the same model that's doing the writing and reviewing -- a
# weaker, self-referential form of grounding than the hand-vetted library.
# Turn off to reproduce the original fixed-library-only behavior.
DYNAMIC_REFERENCES = os.environ.get("HALT_VIDEO_DYNAMIC_REFS", "1") == "1"
DYNAMIC_REFERENCE_COUNT = int(os.environ.get("HALT_VIDEO_DYNAMIC_REF_COUNT", "5"))

# --- Behaviour toggles ---------------------------------------------------

# If True, the Critic call is also given the video (not just the draft text).
CRITIC_WATCHES_VIDEO = os.environ.get("HALT_VIDEO_CRITIC_WATCHES", "0") == "1"

# If True, all Gemini calls are replaced by a deterministic fake backend so the
# loop, cascade, logging, and batch runner can run with no API key and no real
# video files. Used by the test suite and for local smoke checks.
MOCK = os.environ.get("HALT_VIDEO_MOCK", "0") == "1"

# --- Paths ---------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_LIBRARY_PATH = os.path.join(_HERE, "reference_summaries.json")
VIDEOS_DIR = os.path.join(_HERE, "videos")
RESULTS_DIR = os.path.join(_HERE, "results")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv")


def api_key() -> str | None:
    """Return the Gemini API key from the environment, if set."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
