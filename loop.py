"""The Writer/Critic round loop for a single video.

`run_video(path, halting=True)` runs the semantic-halting policy.
`run_video(path, halting=False)` runs the fixed-iteration baseline: identical
loop, but the halt cascade is ignored and it always runs MAX_ROUNDS.

There is no fixed reference library: every run writes its own reference
summaries, tailored to that clip's subject, from a single LLM call. Retrieval
for the whole run works over that per-clip set. Callers that run both policies
for one clip should pass a shared `backend` and a pre-uploaded `video` handle
to avoid re-uploading the clip, and pass the first call's `generated_references`
into the second so both compare against the same examples.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import config
from agents import (
    Usage,
    build_critic_prompt,
    build_writer_prompt,
    get_backend,
    parse_critic_json,
    parse_generated_references,
)
from halting import halt_decision
from retrieval import Embedder, ReferenceLibrary, cosine_distance


class ReferenceGenerationError(RuntimeError):
    """Raised when the clip-specific reference examples couldn't be written
    (and there's no fixed library to fall back to)."""


def run_video(
    video_path: str,
    halting: bool = True,
    *,
    backend=None,
    embedder: Embedder | None = None,
    video=None,
    verbose: bool = False,
    on_event: Callable[[dict], None] | None = None,
    generated_references: list[dict] | None = None,
) -> dict[str, Any]:
    """
    generated_references: reuse a previously-generated clip-specific reference
    set instead of writing a new one (e.g. so the halted and baseline runs of
    the same clip compare against identical examples). Leave as `None` to
    generate a fresh set for this call.
    """
    def emit(kind: str, **data: Any) -> None:
        if on_event is not None:
            on_event({"event": kind, **data})

    backend = backend or get_backend()
    embedder = embedder or Embedder()
    if video is None:
        emit("upload", status="start", message="Sending the clip to Gemini…")
        video = backend.upload_video(video_path)
        emit("upload", status="done")

    usage = Usage()
    policy = "halted" if halting else "baseline"
    name = os.path.basename(video_path) if video_path else "mock"

    if verbose:
        print(f"\n=== {name}  [{policy}] ===")

    # Round-1 retrieval query: a cheap one-line "fresh eyes" description.
    emit("fresh_eyes", status="start", message="Watching the clip for a first-pass description…")
    fresh_eyes_query = backend.fresh_eyes(video, usage)
    query = fresh_eyes_query
    query_source = "fresh-eyes description of the clip"
    emit("fresh_eyes", status="done", text=fresh_eyes_query)

    if generated_references is None:
        emit("generate_refs", status="start",
             message="Writing tailored examples for this clip…")
        generated_references = []
        for attempt in range(2):
            try:
                raw = backend.generate_references(
                    fresh_eyes_query, usage, config.DYNAMIC_REFERENCE_COUNT
                )
                generated_references = parse_generated_references(raw)
            except Exception:  # noqa: BLE001
                generated_references = []
            if generated_references:
                break
        if generated_references:
            emit("generate_refs", status="done",
                 examples=[{"category": e["category"], "text": e["text"]}
                           for e in generated_references])

    if not generated_references:
        raise ReferenceGenerationError(
            "No reference examples to retrieve against for this clip -- "
            "there's no fixed library to fall back to. Try again."
        )
    run_library = ReferenceLibrary(generated_references, embedder)

    rounds: list[dict[str, Any]] = []
    distances: list[float] = []   # d_t for t >= 2
    scores: list[float] = []      # q_t for every completed round
    prev_emb = None
    prev_draft: str | None = None
    prev_critique: str | None = None
    stop_reason = None

    for t in range(1, config.MAX_ROUNDS + 1):
        emit("round", status="start", round=t, max_rounds=config.MAX_ROUNDS)

        emit("retrieve", status="start", round=t,
             query=query, query_source=query_source,
             message=f"Retrieving top-{config.TOP_K} reference summaries…")
        retrieved = run_library.top_k_similar(query, k=config.TOP_K)
        emit("retrieve", status="done", round=t,
             query=query, query_source=query_source,
             retrieved=[r.as_dict() for r in retrieved])

        emit("write", status="start", round=t,
             message=("Writer is drafting the summary…" if t == 1
                      else "Writer is revising with the Critic's feedback…"))
        draft = backend.writer(
            video,
            build_writer_prompt(retrieved, prev_draft, prev_critique, t),
            usage,
            t,
        )
        emit("write", status="done", round=t, draft=draft)

        emb = embedder.encode(draft)
        d_t = cosine_distance(emb, prev_emb) if prev_emb is not None else None
        if d_t is not None:
            distances.append(d_t)
        emit("measure", status="done", round=t,
             distance=None if d_t is None else round(d_t, 4),
             epsilon=config.EPSILON,
             converged=(d_t is not None and d_t < config.EPSILON))

        emit("critique", status="start", round=t, message="Critic is scoring the draft…")
        critic_raw = backend.critic(video, build_critic_prompt(draft, retrieved), usage, t)
        critic = parse_critic_json(critic_raw)
        scores.append(critic.score)
        prev_score = scores[-2] if len(scores) >= 2 else None
        emit("critique", status="done", round=t,
             critique=critic.critique, score=round(critic.score, 4),
             approved=critic.approved,
             delta=None if prev_score is None else round(critic.score - prev_score, 4))

        decision = halt_decision(t, distances, scores, critic.approved)

        rounds.append(
            {
                "round": t,
                "retrieval_query": query,
                "retrieval_query_source": query_source,
                "retrieved": [r.as_dict() for r in retrieved],
                "draft": draft,
                "distance": None if d_t is None else round(d_t, 4),
                "score": round(critic.score, 4),
                "critique": critic.critique,
                "approved": critic.approved,
                "would_halt": decision.stop,
                "halt_reason_if_stopped": decision.reason,
                "halt_detail": decision.detail,
                "cumulative_calls": usage.calls,
                "cumulative_tokens": usage.total_tokens,
            }
        )

        will_halt = halting and decision.stop
        emit("decide", status="done", round=t,
             halts=decision.stop, reason=decision.reason, detail=decision.detail,
             applied=will_halt,
             next_action=("halt" if will_halt else "continue"),
             cumulative_calls=usage.calls, cumulative_tokens=usage.total_tokens)

        if verbose:
            dd = "  --" if d_t is None else f"{d_t:6.3f}"
            print(f"  r{t}: d={dd}  q={critic.score:.2f}  approved={critic.approved}"
                  f"  halt={decision.stop} ({decision.reason})")

        if halting and decision.stop:
            stop_reason = decision.reason
            break

        prev_emb = emb
        prev_draft = draft
        prev_critique = critic.critique
        query = draft  # next round retrieves against the latest draft
        query_source = f"Writer's draft from round {t}"

    if stop_reason is None:
        # A halted run always breaks (failsafe fires at MAX_ROUNDS), so this is
        # the baseline path: it ran the full fixed schedule.
        stop_reason = "baseline_cap"

    result = {
        "video": name,
        "video_path": video_path,
        "policy": policy,
        "backend": getattr(backend, "name", "unknown"),
        "fresh_eyes_query": fresh_eyes_query,
        "generated_references": generated_references,
        "rounds_used": len(rounds),
        "max_rounds": config.MAX_ROUNDS,
        "stop_reason": stop_reason,
        "final_summary": rounds[-1]["draft"] if rounds else "",
        "final_score": rounds[-1]["score"] if rounds else None,
        "distances": [round(d, 4) for d in distances],
        "scores": [round(s, 4) for s in scores],
        "usage": usage.as_dict(),
        "config": {
            "MAX_ROUNDS": config.MAX_ROUNDS,
            "EPSILON": config.EPSILON,
            "PATIENCE": config.PATIENCE,
            "DELTA": config.DELTA,
            "TOP_K": config.TOP_K,
            "MODEL": config.MODEL,
            "EMBED_MODEL": config.EMBED_MODEL,
            "DYNAMIC_REFERENCE_COUNT": config.DYNAMIC_REFERENCE_COUNT,
        },
        "rounds": rounds,
    }
    emit("done", status="done",
         stop_reason=stop_reason, rounds_used=len(rounds),
         final_summary=result["final_summary"], final_score=result["final_score"],
         usage=result["usage"])
    return result
