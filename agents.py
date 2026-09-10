"""Writer and Critic agents, plus the round-1 "fresh eyes" description.

Every model call goes through Gemini (`google-genai`). Video clips are uploaded
once via the File API and the handle is reused across all rounds and the
baseline pass -- see `loop.run_video`.

Set HALT_VIDEO_MOCK=1 to swap in a deterministic fake backend (no API key, no
real video needed). The fake is good enough to exercise the loop, the cascade,
the logging, and the batch runner end to end.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import config

# ---------------------------------------------------------------------------
# Token / call accounting
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """Running tally of API cost across a run."""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)

    def add(self, agent: str, prompt: int, output: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.output_tokens += output
        self.total_tokens += prompt + output
        self.by_agent[agent] = self.by_agent.get(agent, 0) + prompt + output

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "by_agent": dict(self.by_agent),
        }


@dataclass
class CriticResult:
    critique: str
    score: float
    approved: bool
    raw: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_WRITER_SYSTEM = (
    "You are a careful video-summarization writer. Watch the clip and write a "
    "factual summary of what actually happens in it: the main subject, the key "
    "actions in order, the setting, and the outcome. Write 2-3 sentences of "
    "plain prose. No markdown, no bullet points, no preamble like 'This video "
    "shows'. Only describe what is visible in THIS clip -- never borrow details "
    "from the reference examples."
)

_CRITIC_SYSTEM = (
    "You are a strict editor judging a draft video summary against a small set "
    "of reference-quality summaries. Judge only: (a) how well the draft matches "
    "the depth, specificity, and plain-prose style of the references, and (b) "
    "whether it reads as complete and self-consistent. You are NOT checking the "
    "draft against the video. Return strict JSON."
)


def _format_examples(retrieved) -> str:
    lines = []
    for i, ex in enumerate(retrieved, 1):
        text = ex["text"] if isinstance(ex, dict) else getattr(ex, "text", str(ex))
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


def build_writer_prompt(retrieved, prev_draft: str | None,
                        critic_feedback: str | None, round_idx: int) -> str:
    parts = [
        "REFERENCE EXAMPLES (for depth and style only -- do not copy their content):",
        _format_examples(retrieved),
        "",
    ]
    if round_idx > 1 and prev_draft:
        parts += [
            "YOUR PREVIOUS DRAFT:",
            prev_draft,
            "",
            "EDITOR FEEDBACK ON THAT DRAFT:",
            critic_feedback or "(none)",
            "",
            "Revise your summary of the clip, addressing the feedback where it is "
            "correct. If the draft is already accurate and complete, make only "
            "small improvements rather than adding speculation.",
        ]
    else:
        parts.append("Write your summary of the clip.")
    return "\n".join(parts)


def build_critic_prompt(draft: str, retrieved) -> str:
    return "\n".join(
        [
            "REFERENCE-QUALITY SUMMARIES:",
            _format_examples(retrieved),
            "",
            "DRAFT TO JUDGE:",
            draft,
            "",
            'Return JSON exactly: {"critique": "<2-4 sentences of specific, '
            'actionable feedback>", "score": <0.0-1.0 how close to reference '
            'quality/completeness>, "approved": <true only if no further '
            "revision is worthwhile>}",
        ]
    )


# ---------------------------------------------------------------------------
# Real Gemini backend
# ---------------------------------------------------------------------------


class GeminiBackend:
    name = "gemini"

    def __init__(self) -> None:
        from google import genai  # imported lazily so mock mode needs no package

        key = config.api_key()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it, put it in a .env file, "
                "or run with HALT_VIDEO_MOCK=1 for the offline fake backend."
            )
        self.client = genai.Client(api_key=key)

    def upload_video(self, path: str):
        f = self.client.files.upload(file=path)
        deadline = time.time() + 300
        while _state(f) == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError(f"video still processing after 5 min: {path}")
            time.sleep(3)
            f = self.client.files.get(name=f.name)
        if _state(f) == "FAILED":
            raise RuntimeError(f"Gemini failed to process video: {path}")
        return f

    def _generate(self, contents, system: str, usage: Usage, agent: str,
                  json_out: bool = False) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.4,
            response_mime_type="application/json" if json_out else "text/plain",
        )
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=config.MODEL, contents=contents, config=cfg
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        meta = getattr(resp, "usage_metadata", None)
        usage.add(
            agent,
            getattr(meta, "prompt_token_count", 0) or 0,
            getattr(meta, "candidates_token_count", 0) or 0,
        )
        return (getattr(resp, "text", "") or "").strip()

    def fresh_eyes(self, video, usage: Usage) -> str:
        return self._generate(
            [video, "Describe this video in one plain sentence."],
            "You describe videos in a single factual sentence.",
            usage,
            "fresh_eyes",
        )

    def writer(self, video, prompt: str, usage: Usage, round_idx: int) -> str:
        return self._generate([video, prompt], _WRITER_SYSTEM, usage, "writer")

    def critic(self, video, prompt: str, usage: Usage, round_idx: int) -> str:
        contents = [video, prompt] if (video is not None and config.CRITIC_WATCHES_VIDEO) else [prompt]
        return self._generate(contents, _CRITIC_SYSTEM, usage, "critic", json_out=True)


def _state(f) -> str:
    s = getattr(f, "state", "")
    return getattr(s, "name", str(s)).upper()


# ---------------------------------------------------------------------------
# Deterministic mock backend
# ---------------------------------------------------------------------------


class MockBackend:
    """Fake Gemini: canned, deterministic, converges by round 3.

    Drafts get more detailed for the first few rounds, then repeat verbatim so
    the embedding-distance signal fires. Critic scores rise then plateau, and
    the Critic never approves -- so a mock run halts via `no_gain`.
    """

    name = "mock"

    _THIN = ("A dog runs across a grassy field. It moves quickly from one side "
             "to the other.")
    _FULL = ("A golden retriever sprints across a bright open grass field, ears "
             "flat and tongue out, chasing a ball. It skids as it reaches the "
             "ball, grabs it, and turns back toward a person off-screen.")
    # round 1 thin, then the draft stabilises -> d_3, d_4 ~ 0 -> no_gain at r4
    _DRAFTS = [_THIN, _FULL, _FULL, _FULL, _FULL]
    _SCORES = [0.55, 0.85, 0.90, 0.905, 0.905]

    def upload_video(self, path: str):
        return {"mock_video": os.path.basename(path) if path else "mock"}

    def fresh_eyes(self, video, usage: Usage) -> str:
        usage.add("fresh_eyes", 120, 15)
        return "A dog running across a grassy field."

    def writer(self, video, prompt: str, usage: Usage, round_idx: int) -> str:
        i = min(max(round_idx - 1, 0), len(self._DRAFTS) - 1)
        usage.add("writer", 400 + i * 60, 60)
        return self._DRAFTS[i]

    def critic(self, video, prompt: str, usage: Usage, round_idx: int) -> str:
        i = min(max(round_idx - 1, 0), len(self._SCORES) - 1)
        usage.add("critic", 300, 40)
        score = self._SCORES[i]
        return json.dumps(
            {
                "critique": (
                    f"Round {round_idx}: draft is "
                    + ("thin -- add the setting and the outcome."
                       if i < 2 else "close to reference quality; only minor wording left.")
                ),
                "score": score,
                "approved": bool(score >= 0.95),
            }
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_backend():
    return MockBackend() if config.MOCK else GeminiBackend()


def parse_critic_json(raw: str) -> CriticResult:
    """Parse the Critic's JSON, tolerating code fences and stray prose."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return CriticResult(critique=raw[:500], score=0.0, approved=False, raw=raw)
    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = min(1.0, max(0.0, score))
    return CriticResult(
        critique=str(data.get("critique", "")).strip(),
        score=score,
        approved=bool(data.get("approved", False)),
        raw=raw,
    )
