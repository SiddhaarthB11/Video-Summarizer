# Semantic Halting for Iterative Video Summarization

An applied extension of **HaltIQ** / *"Semantic Early-Stopping for Iterative LLM
Agent Loops"* (Shrivastava, 2026) to a new task and modality: iterative video
summarization.

A **Writer** agent watches a short clip and drafts a summary. A **Critic** agent
compares that draft against a small RAG-retrieved set of reference-quality
summaries and returns feedback plus a quality score. The Writer revises. The
loop repeats until a **semantic-halting signal** fires or a hard round cap is
reached — instead of always running a fixed `max_iterations`.

## Research question

Does a free, embedding-based semantic-convergence signal — paired with a
Critic's qualitative judgment — reliably identify when an iterative
video-summarization loop has stopped improving, so that halting early saves
compute/API cost without a meaningful drop in summary quality, relative to a
fixed-iteration baseline?

## How the loop works

```
fresh-eyes description of the clip  ─┐
                                    ▼
   ┌─────────────── ROUND t = 1..MAX_ROUNDS ───────────────┐
   │  retrieve top-k reference summaries (query = prev draft │
   │      on round > 1, else the fresh-eyes description)     │
   │  WRITER(clip, prev draft, critic feedback, retrieved) → x_t
   │  e_t = embed(x_t);  d_t = 1 - cos(e_t, e_{t-1})   [HaltIQ Eq. 1]
   │  CRITIC(x_t, retrieved) → {critique, score q_t, approved}
   │  HALT CASCADE:
   │    1. approved?                              → HALT (critic)
   │    2. last PATIENCE distances < EPSILON
   │       AND q_t - q_{t-1} < DELTA              → HALT (no_gain)
   │    3. t == MAX_ROUNDS                        → HALT (failsafe)
   │    4. else                                   → CONTINUE
   └────────────────────────────────────────────────────────┘
                                    ▼
        final summary + full per-round log  →  compare vs. fixed-round baseline
```

The "narrowing" of retrieval across rounds is not special logic: as the draft
becomes more specific, the retrieval query (the draft itself) becomes more
specific, so retrieved examples get more targeted.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your key in it
# or: export GEMINI_API_KEY=...
```

Get a Gemini API key from https://aistudio.google.com/apikey.

Drop 3–5 short (~30s) clips into `videos/` (`.mp4`, `.mov`, `.webm`, ...).

## Run

### Interactive — upload a clip, watch the loop

```bash
python app.py
```

Open http://localhost:5000. Drop in a video (or click one of the **bundled
clips** — whatever is in `videos/`), hit **Run the loop**. Every step streams to
the page as it happens: the first-glance description, each attempt's lookup and
the example summaries it found, the summary it wrote, how much it changed, the
reviewer's notes and rating, and the decision to keep going or stop. First start
takes ~1 min (loads the embedding model). ffmpeg, if installed, downscales the
upload before it goes to Gemini. Each run is saved to `results/<clip>__halted.json`
(with a preview in `assets/`), so it appears at `/dashboard` — the walkthrough of
past runs on the same server.

### Batch — all clips, halted vs. baseline

```bash
python run.py                    # every clip in videos/, halted + baseline
python run.py --only halted       # skip the baseline pass
python run.py --verbose           # per-round trace to stdout
python run.py --mock              # deterministic fake backend: no key, no videos
```

Per-run logs are written to `results/<clip>__halted.json` and
`results/<clip>__baseline.json`, plus a combined `results/summary.json`. The
script also prints a comparison table:

### Visual walkthrough

```bash
python build_dashboard.py
```

Reads `results/` + the preview clips in `assets/` and writes a self-contained
`dashboard.html` (open in a browser, or visit `/dashboard` on the running app)
plus `dashboard.artifact.html` for publishing. It's written in plain language for
a non-technical reader: per clip and per run it shows each attempt — the example
summaries it looked up, the summary it wrote, how much it changed, the reviewer's
notes and rating, and why it stopped. The distances/scores/thresholds themselves
stay in the `results/*.json` files.

```
clip                    rounds h/b  round save    stop reason        q h/b  tok save
------------------------------------------------------------------------------------
dog_running.mp4                3/5      40.0%          no_gain    0.90/0.91    41.3%
street_scene.mp4               5/5       0.0%         failsafe    0.83/0.84     3.1%
```

## Tuning (`config.py`)

| Constant   | Default | Meaning |
|------------|---------|---------|
| `MAX_ROUNDS` | 5 | hard failsafe cap on Writer/Critic rounds |
| `EPSILON`    | 0.05 | cosine-distance threshold below which a round counts as "converged" |
| `PATIENCE`   | 2 | consecutive converged rounds required before `no_gain` can fire |
| `DELTA`      | 0.02 | minimum Critic-score gain that still counts as "improving" |
| `TOP_K`      | 3 | reference summaries retrieved per round |
| `CRITIC_WATCHES_VIDEO` | False | also give the Critic the clip, not just the draft text |

## Files

| File | Role |
|------|------|
| `config.py` | all tunable constants, model names, paths |
| `retrieval.py` | local embedder + reference library + `top_k_similar` + `cosine_distance` |
| `reference_summaries.json` | 25 hand-written reference summaries (4 categories) — **review/edit these** |
| `agents.py` | Writer, Critic, fresh-eyes description; real Gemini + deterministic mock backend |
| `halting.py` | the four-level halt cascade (pure function) |
| `loop.py` | `run_video(path, halting)` — one clip, one policy |
| `app.py` | interactive Flask runner — upload a clip, watch the loop live; also serves `/dashboard` |
| `run.py` | batch runner over `videos/`, writes `results/`, prints the table |
| `build_dashboard.py` | turns `results/` into a plain-language HTML walkthrough (`dashboard.html` + `dashboard.artifact.html`) |
| `tests/` | `pytest` — cascade logic, retrieval ranking, mock end-to-end |

Generated / local-only (git-ignored): `.env`, `dashboard*.html`, `results/*.json`,
`videos/*`, `assets/`. A fresh clone needs your `GEMINI_API_KEY` in `.env` and a
few clips in `videos/`; `assets/` holds small preview copies the dashboard embeds
(created with ffmpeg, or just drop 480p copies there yourself).

## Tests

```bash
pytest -q                              # all tests
HALT_VIDEO_MOCK=1 python run.py --mock   # full pipeline, fake backend
```

`test_halting.py` and `test_loop_mock.py` need no API key. `test_retrieval.py`
downloads `all-MiniLM-L6-v2` (~90 MB) on first run.

## Evaluation plan

For each of the 3–5 test videos, run both the halted policy and the
fixed-iteration baseline, then report:

- Rounds used (halted vs. baseline) and % reduction.
- Final Critic score `q_t` for both policies (does halting early cost quality?).
- A brief manual read of the final summaries against the video (does the halted
  version still capture the key content?).
- One or two per-video notes on why the halt fired when it did (e.g. "converged
  after round 2 because the scene was simple"; "ran to the failsafe cap on a
  busier/ambiguous clip").

## Scope notes

**In scope / implemented:**

* Two-agent Writer/Critic loop (not a single self-revising agent)
* RAG grounding via a hand-built reference-summary library with cosine-similarity
  retrieval, re-queried each round
* The free embedding-distance convergence signal (HaltIQ's Eq. 1)
* A second, Critic-driven quality signal, combined with the distance signal in a
  4-level halt cascade
* A genuine fixed-iteration baseline for comparison

**Out of scope (these do not exist in this project):**

* HaltIQ's formal, machine-checked termination proofs (§V of their paper) — this
  project only observes empirical behavior, makes no formal guarantees
* HaltIQ's paired statistical significance / non-inferiority testing across a
  large test split — this project uses 3–5 videos and manual quality checks, not
  a statistically powered study
* A judge-efficient token-accounting harness that separates operational vs.
  evaluation tokens and caches judge calls — this project logs simple call
  counts, not a full cost-accounting system
