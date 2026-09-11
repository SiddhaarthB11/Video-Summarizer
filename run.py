"""Batch runner: for every clip in videos/, run the halted policy and the
fixed-iteration baseline, write per-run JSON logs to results/, and print a
comparison table.

    python run.py                 # all clips, both policies
    python run.py --mock          # deterministic fake backend, no key/videos
    python run.py --only halted    # skip the baseline pass
    python run.py --videos-dir path/to/clips
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import config


def discover_videos(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    out = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(config.VIDEO_EXTENSIONS)
    ]
    return out


def _pct(base: float, new: float) -> str:
    if not base:
        return "  n/a"
    return f"{100.0 * (base - new) / base:5.1f}%"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", default=config.VIDEOS_DIR)
    ap.add_argument("--results-dir", default=config.RESULTS_DIR)
    ap.add_argument("--only", choices=["halted", "baseline", "both"], default="both")
    ap.add_argument("--mock", action="store_true", help="use the deterministic fake backend")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.mock:
        os.environ["HALT_VIDEO_MOCK"] = "1"
        import importlib
        importlib.reload(config)

    # imported after the MOCK env flag is set
    from agents import get_backend
    from loop import run_video
    from retrieval import Embedder, ReferenceLibrary

    videos = discover_videos(args.videos_dir)
    if args.mock and not videos:
        videos = ["mock-clip-1.mp4", "mock-clip-2.mp4"]  # names only; mock ignores path

    if not videos:
        print(f"No videos found in {args.videos_dir!r}. Add a few .mp4 clips "
              f"(or run with --mock).")
        return 1

    os.makedirs(args.results_dir, exist_ok=True)
    embedder = Embedder()
    library = ReferenceLibrary.from_json(embedder=embedder)
    backend = get_backend()

    policies = {"both": [True, False], "halted": [True], "baseline": [False]}[args.only]

    summary_rows = []
    all_results = {}

    for vpath in videos:
        name = os.path.basename(vpath)
        real_path = vpath if os.path.exists(vpath) else ""
        try:
            video = backend.upload_video(real_path)
        except Exception as e:  # noqa: BLE001
            print(f"!! {name}: upload failed: {e}")
            continue

        per_clip = {}
        generated_refs = None  # generate once per clip, reuse for both policies
        for halting in policies:
            tag = "halted" if halting else "baseline"
            try:
                res = run_video(
                    real_path or vpath,
                    halting=halting,
                    backend=backend,
                    library=library,
                    embedder=embedder,
                    video=video,
                    verbose=args.verbose,
                    generated_references=generated_refs,
                )
            except Exception:  # noqa: BLE001
                print(f"!! {name} [{tag}] failed:\n{traceback.format_exc()}")
                continue
            generated_refs = res.get("generated_references")
            per_clip[tag] = res
            out_path = os.path.join(args.results_dir, f"{name}__{tag}.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2)

        all_results[name] = per_clip
        h = per_clip.get("halted")
        b = per_clip.get("baseline")
        summary_rows.append(
            {
                "video": name,
                "halted_rounds": h["rounds_used"] if h else None,
                "baseline_rounds": b["rounds_used"] if b else None,
                "stop_reason": h["stop_reason"] if h else None,
                "halted_q": h["final_score"] if h else None,
                "baseline_q": b["final_score"] if b else None,
                "halted_tokens": h["usage"]["total_tokens"] if h else None,
                "baseline_tokens": b["usage"]["total_tokens"] if b else None,
                "halted_calls": h["usage"]["calls"] if h else None,
                "baseline_calls": b["usage"]["calls"] if b else None,
            }
        )

    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"rows": summary_rows, "results": all_results}, fh, indent=2)

    _print_table(summary_rows)
    return 0


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results.")
        return
    print()
    header = (
        f"{'clip':<22} {'rounds h/b':>11} {'round save':>10} "
        f"{'stop reason':>14} {'q h/b':>12} {'tok save':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        hr, br = r["halted_rounds"], r["baseline_rounds"]
        rounds = f"{hr if hr is not None else '-'}/{br if br is not None else '-'}"
        rsave = _pct(br, hr) if (hr is not None and br) else "  n/a"
        q = (
            f"{(r['halted_q'] if r['halted_q'] is not None else float('nan')):.2f}"
            f"/{(r['baseline_q'] if r['baseline_q'] is not None else float('nan')):.2f}"
        )
        tsave = (
            _pct(r["baseline_tokens"], r["halted_tokens"])
            if (r["halted_tokens"] is not None and r["baseline_tokens"])
            else "  n/a"
        )
        print(f"{r['video'][:22]:<22} {rounds:>11} {rsave:>10} "
              f"{str(r['stop_reason']):>14} {q:>12} {tsave:>9}")
    print()


if __name__ == "__main__":
    sys.exit(main())
