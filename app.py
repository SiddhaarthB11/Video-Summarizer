"""Interactive runner: upload a clip, watch the halting loop work in real time.

    python app.py           # then open http://localhost:5000

Runs the semantic-halting policy (Writer / Critic / RAG / cascade) on one
uploaded video and streams every step to the browser over Server-Sent Events.
Each completed run is written to results/<clip>__halted.json (same shape as
run.py) with a small preview copy in assets/, so it shows up at /dashboard.
Needs GEMINI_API_KEY (see .env). ffmpeg, if present, downscales the upload.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid

import config  # loads .env
from flask import Flask, Response, request, send_file

# heavy imports (torch etc.) happen here, once, at startup
from agents import get_backend
from loop import run_video
from retrieval import Embedder, ReferenceLibrary

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads

JOBS: dict[str, queue.Queue] = {}
_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "halt_video_uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

print("Loading embedding model and reference library (first run downloads ~90 MB)…")
EMBEDDER = Embedder()
EMBEDDER.encode("warmup")
LIBRARY = ReferenceLibrary.from_json(embedder=EMBEDDER)
BACKEND = get_backend()
print(f"Ready. {len(LIBRARY)} reference summaries, backend={getattr(BACKEND,'name','?')}.")


def _scale_to(src: str, dst: str, height: int, keep_audio: bool, crf: int = 28) -> str:
    """Re-encode `src` to at most `height` px tall. Returns `dst` on success,
    else `src` (missing ffmpeg or an encode error)."""
    if not shutil.which("ffmpeg"):
        return src
    audio = ["-c:a", "aac", "-b:a", "96k"] if keep_audio else ["-an"]
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", src,
        "-vf", f"scale=-2:'min({height},ih)'", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", str(crf), *audio, "-movflags", "+faststart", dst,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=180)
        return dst if os.path.exists(dst) else src
    except Exception:  # noqa: BLE001
        return src


def _downscale(src: str) -> str:
    """720p copy (with audio) for the upload to Gemini, or `src` unchanged."""
    return _scale_to(src, src + ".720.mp4", 720, keep_audio=True)


def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "clip"


def _save_run(result: dict, original_name: str, small_path: str) -> str:
    """Write the run to results/ (same shape as run.py) and stash a preview
    copy in assets/ so it shows up in the dashboard. Returns the json filename."""
    stem = _safe_stem(original_name)
    result = dict(result)
    result["video"] = original_name
    result["video_path"] = f"uploaded via app · {original_name}"

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, f"{stem}__halted.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    try:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        asset = os.path.join(ASSETS_DIR, f"{stem}.mp4")
        made = _scale_to(small_path, asset + ".tmp.mp4", 480, keep_audio=False, crf=30)
        if made != small_path and os.path.exists(made):
            os.replace(made, asset)
        elif os.path.exists(small_path):  # no ffmpeg -> just copy
            shutil.copyfile(small_path, asset)
    except OSError:
        pass
    return os.path.basename(out)


def _worker(job_id: str, path: str, original_name: str) -> None:
    q = JOBS[job_id]

    def emit(ev: dict) -> None:
        q.put(ev)

    video = None
    small = path
    try:
        emit({"event": "prepare", "message": "Downscaling the clip for upload…"})
        small = _downscale(path)
        emit({"event": "upload", "status": "start", "message": "Sending the clip to Gemini…"})
        video = BACKEND.upload_video(small)
        emit({"event": "upload", "status": "done"})
        result = run_video(
            small,
            halting=True,
            backend=BACKEND,
            library=LIBRARY,
            embedder=EMBEDDER,
            video=video,
            on_event=emit,
        )
        saved = _save_run(result, original_name, small)
        emit({"event": "saved", "file": saved})
    except Exception as exc:  # noqa: BLE001
        emit({"event": "error", "message": str(exc),
              "trace": traceback.format_exc()})
    finally:
        if video is not None:
            BACKEND.delete_file(video)
        emit({"event": "_end"})
        for p in {path, small, path + ".720.mp4"}:
            try:
                os.remove(p)
            except OSError:
                pass


@app.post("/run")
def run():
    f = request.files.get("clip")
    if not f or not f.filename:
        return {"error": "no file"}, 400
    ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
    job_id = uuid.uuid4().hex
    path = os.path.join(_UPLOAD_DIR, job_id + ext)
    f.save(path)
    JOBS[job_id] = queue.Queue()
    threading.Thread(target=_worker, args=(job_id, path, f.filename), daemon=True).start()
    return {"job_id": job_id, "filename": f.filename}


@app.get("/events/<job_id>")
def events(job_id: str):
    q = JOBS.get(job_id)
    if q is None:
        return {"error": "unknown job"}, 404

    def stream():
        yield f"retry: 2000\n\ndata: {json.dumps({'event': 'config', **CONFIG_PAYLOAD})}\n\n"
        while True:
            ev = q.get()
            if ev.get("event") == "_end":
                yield "data: {\"event\": \"_end\"}\n\n"
                break
            yield f"data: {json.dumps(ev)}\n\n"
        JOBS.pop(job_id, None)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/dashboard")
def dashboard():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    if not os.path.exists(p):
        return ("dashboard.html not built yet — run `python build_dashboard.py`", 404)
    return send_file(p, mimetype="text/html", conditional=True)


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


CONFIG_PAYLOAD = {
    "MAX_ROUNDS": config.MAX_ROUNDS, "EPSILON": config.EPSILON,
    "PATIENCE": config.PATIENCE, "DELTA": config.DELTA, "TOP_K": config.TOP_K,
    "MODEL": config.MODEL, "EMBED_MODEL": config.EMBED_MODEL,
}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Halting Loop — Live Runner</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --ground:#f5f7f9; --surface:#fff; --surface-2:#eef1f5;
  --ink:#1a1f2b; --ink-soft:#5a6472; --ink-faint:#8b94a3;
  --line:#e0e4ea; --line-strong:#c9d0da;
  --accent:#166b78; --accent-soft:#e0eef0;
  --ok:#3f7d58; --warn:#a9761c; --stop:#a5402f; --neutral:#6b7280;
  --shadow:0 1px 2px rgba(20,30,45,.05), 0 8px 24px rgba(20,30,45,.06);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0f1218; --surface:#161b23; --surface-2:#1d2430;
  --ink:#e7eaf0; --ink-soft:#9aa4b3; --ink-faint:#6b7585;
  --line:#242c39; --line-strong:#333d4e;
  --accent:#54b9c6; --accent-soft:#12333a;
  --ok:#5fae7f; --warn:#d19a4a; --stop:#d4715f; --neutral:#8b95a5;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --ground:#0f1218; --surface:#161b23; --surface-2:#1d2430;
  --ink:#e7eaf0; --ink-soft:#9aa4b3; --ink-faint:#6b7585;
  --line:#242c39; --line-strong:#333d4e; --accent:#54b9c6; --accent-soft:#12333a;
  --ok:#5fae7f; --warn:#d19a4a; --stop:#d4715f; --neutral:#8b95a5;
}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.55}
.wrap{max-width:840px;margin:0 auto;padding:40px 24px 120px}
h1{font-size:26px;font-weight:600;letter-spacing:-.015em;margin:8px 0 8px;text-wrap:balance}
.sub{color:var(--ink-soft);font-size:14.5px;margin:0;max-width:62ch}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;
  letter-spacing:.16em;text-transform:uppercase;color:var(--accent)}
header.top{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:26px}

/* config pills */
.cfg{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px}
.cfg span{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-soft);
  background:var(--surface-2);border:1px solid var(--line);border-radius:999px;padding:3px 10px}
.cfg span b{color:var(--ink);font-weight:600}

/* dropzone */
.uploader{display:flex;flex-direction:column;gap:14px}
.dropzone{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;
  width:100%;min-height:168px;border:1.5px dashed var(--line-strong);border-radius:14px;
  padding:28px 24px;text-align:center;background:var(--surface);cursor:pointer;
  transition:border-color .15s,background .15s,box-shadow .15s}
.dropzone:hover,.dropzone.over{border-color:var(--accent);background:var(--accent-soft)}
.dropzone.over{box-shadow:0 0 0 4px color-mix(in srgb,var(--accent) 18%,transparent)}
.dropzone svg{width:26px;height:26px;color:var(--ink-faint)}
.dropzone .big{font-size:15px;color:var(--ink);font-weight:500}
.dropzone .hint{font-size:12.5px;color:var(--ink-faint)}
.dropzone.has-file{border-style:solid;border-color:var(--accent);
  background:color-mix(in srgb,var(--accent) 7%,var(--surface))}
.dropzone.has-file svg{color:var(--accent)}
.filechip{display:inline-flex;align-items:center;gap:8px;font-family:"IBM Plex Mono",monospace;
  font-size:12.5px;color:var(--ink);background:var(--surface);border:1px solid var(--line-strong);
  border-radius:8px;padding:6px 12px}
.filechip .sz{color:var(--ink-faint)}
#file{display:none}

.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
button.run{font:inherit;font-weight:600;font-size:14px;cursor:pointer;border:0;border-radius:10px;
  padding:11px 24px;background:var(--accent);color:#fff;transition:opacity .15s,transform .05s}
button.run:hover:not(:disabled){filter:brightness(1.06)}
button.run:active:not(:disabled){transform:translateY(1px)}
button.run:disabled{opacity:.4;cursor:not-allowed}
button.run:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.ghost{background:transparent;border:1px solid var(--line-strong);color:var(--ink-soft)}

.status{margin:22px 0 4px;padding:12px 16px;border-radius:10px;background:var(--surface-2);
  border:1px solid var(--line);font-size:13px;font-family:"IBM Plex Mono",monospace;color:var(--ink-soft);
  display:flex;align-items:center;gap:11px;min-height:44px}
.status.idle{background:transparent;border-style:dashed}
.status.err{background:color-mix(in srgb,var(--stop) 12%,transparent);border-color:var(--stop);color:var(--stop)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--accent);flex:none;animation:pulse 1s infinite}
.status.idle .dot{background:var(--ink-faint);animation:none}
.status.done .dot{background:var(--ok);animation:none}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
@media (prefers-reduced-motion:reduce){.dot{animation:none!important}}

/* how one round works — resting content */
.howto{margin-top:30px}
.howto h2{font-size:12px;font-family:"IBM Plex Mono",monospace;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-soft);margin:0 0 14px}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
@media (max-width:640px){.flow{grid-template-columns:1fr 1fr}}
.flow .fstep{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 12px 14px}
.flow .fstep .n{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--accent);font-weight:600}
.flow .fstep .t{font-size:12.5px;font-weight:600;margin:3px 0 4px}
.flow .fstep .d{font-size:11px;color:var(--ink-soft);line-height:1.45}

.fresh{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:8px;padding:12px 15px;margin:14px 0;font-size:13.5px}
.fresh .lab{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-faint);display:block;margin-bottom:4px}

.round{background:var(--surface);border:1px solid var(--line);border-radius:12px;margin:14px 0;
  overflow:hidden;box-shadow:var(--shadow)}
.round > .rhead{padding:12px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--line)}
.round .rtitle{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:13px}
.round .rchips{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}
.chip{font-family:"IBM Plex Mono",monospace;font-size:11px;padding:3px 8px;border-radius:5px;
  background:var(--surface-2);color:var(--ink-soft);border:1px solid var(--line);font-variant-numeric:tabular-nums}
.chip.halt{background:color-mix(in srgb,var(--stop) 14%,transparent);color:var(--stop);border-color:var(--stop)}
.chip.go{background:color-mix(in srgb,var(--ok) 12%,transparent);color:var(--ok);border-color:var(--ok)}

.step{display:grid;grid-template-columns:92px 1fr;gap:14px;padding:13px 16px;border-top:1px solid var(--line)}
.step:first-child{border-top:0}
.step .slab{font-family:"IBM Plex Mono",monospace;font-size:10px;font-weight:600;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink-faint);padding-top:2px}
.step.pending .sbody{color:var(--ink-faint);font-style:italic;font-size:13px}
@media (max-width:560px){.step{grid-template-columns:1fr;gap:5px}}

.draft{font-size:13.5px;line-height:1.6;background:var(--surface-2);border:1px solid var(--line);
  border-radius:8px;padding:11px 13px}
.qsrc{font-size:12px;color:var(--ink-soft);margin-bottom:7px}
.qsrc b{color:var(--ink)}
.qtext{font-family:"IBM Plex Mono",monospace;font-size:12px;background:var(--surface-2);
  border:1px solid var(--line);border-radius:7px;padding:8px 10px;margin-bottom:9px}
.docs{display:flex;flex-direction:column;gap:7px}
.doc{border:1px solid var(--line);border-radius:8px;padding:9px 11px}
.doc.top{border-color:var(--accent);background:var(--accent-soft)}
.doc .dh{display:flex;gap:8px;align-items:center;margin-bottom:4px;flex-wrap:wrap}
.doc .did{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;color:var(--accent)}
.doc .dcat{font-family:"IBM Plex Mono",monospace;font-size:10px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--ink-faint)}
.doc .dsim{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--ink-soft);
  display:flex;align-items:center;gap:6px}
.simbar{width:54px;height:5px;border-radius:3px;background:var(--line-strong);overflow:hidden}
.simbar i{display:block;height:100%;background:var(--accent)}
.doc .dtext{font-size:12px;color:var(--ink-soft);line-height:1.5}
.tag{font-size:9px;font-weight:700;letter-spacing:.06em;color:#fff;background:var(--accent);
  padding:2px 5px;border-radius:4px}
.tag.gen{background:var(--warn)}

.gauge svg{width:100%;max-width:380px;height:auto;display:block;overflow:visible}
.note{font-size:12.5px;color:var(--ink-soft);margin-top:5px}
.note b{color:var(--ink)}

.checks{display:flex;flex-direction:column;gap:6px}
.check{display:grid;grid-template-columns:18px 130px 1fr;gap:9px;font-size:12px;align-items:start}
.check .cm{font-family:"IBM Plex Mono",monospace;font-weight:700}
.check .cn{font-family:"IBM Plex Mono",monospace;color:var(--ink-soft)}
.check .cw{color:var(--ink-faint)}
.check.hit .cm{color:var(--stop)} .check.hit .cn{color:var(--ink)}
.check.miss .cm{color:var(--ink-faint)}
.verdict{margin-top:9px;font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:12.5px}
.verdict.halt{color:var(--stop)} .verdict.go{color:var(--ok)}

.final{margin-top:22px;border:1px solid var(--accent);background:var(--accent-soft);border-radius:12px;padding:18px}
.final .lab{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent);font-weight:600}
.final p{margin:7px 0 12px;font-size:15px;line-height:1.6}
.final .oc{display:flex;gap:7px;flex-wrap:wrap}
.reason-critic{background:color-mix(in srgb,var(--ok) 16%,transparent);border-color:var(--ok);color:var(--ok)}
.reason-no_gain{background:color-mix(in srgb,var(--warn) 16%,transparent);border-color:var(--warn);color:var(--warn)}
.reason-failsafe{background:color-mix(in srgb,var(--stop) 14%,transparent);border-color:var(--stop);color:var(--stop)}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="eyebrow">HaltIQ · live</div>
    <h1>Watch the halting loop run</h1>
    <p class="sub">Upload a short clip. The Writer drafts a summary, RAG pulls reference
    summaries, the draft is embedded and measured against the last one, the Critic scores
    it, and the cascade decides whether to keep going — every step shown as it happens.</p>
    <div class="cfg" id="cfg"></div>
  </header>

  <div class="uploader">
    <label class="dropzone" id="dz" for="file">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
        stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 16V4"/><path d="m6 10 6-6 6 6"/><path d="M4 20h16"/>
      </svg>
      <span class="big" id="dzlabel">Drop a video here, or click to choose</span>
      <span class="hint" id="dzhint">.mp4 · .mov · .webm — a 4K clip is downscaled to 720p before upload</span>
      <input type="file" id="file" accept="video/*">
    </label>

    <div class="controls">
      <button class="run" id="go" disabled>Run the loop</button>
      <button class="run ghost" id="again" hidden>Run another clip</button>
    </div>
  </div>

  <div class="status idle" id="status"><span class="dot"></span><span id="statustext">Waiting for a clip.</span></div>

  <section class="howto" id="howto">
    <h2>How one round works</h2>
    <div class="flow">
      <div class="fstep"><div class="n">01</div><div class="t">Retrieve</div>
        <div class="d">Embed the query, pull the top-k nearest reference summaries.</div></div>
      <div class="fstep"><div class="n">02</div><div class="t">Write</div>
        <div class="d">Writer drafts (or revises) the summary from the clip + feedback.</div></div>
      <div class="fstep"><div class="n">03</div><div class="t">Measure</div>
        <div class="d">Embed the draft; cosine distance dₜ from the previous draft.</div></div>
      <div class="fstep"><div class="n">04</div><div class="t">Critique</div>
        <div class="d">Critic scores it qₜ against the references and may approve.</div></div>
      <div class="fstep"><div class="n">05</div><div class="t">Decide</div>
        <div class="d">Cascade: critic → no_gain → failsafe → continue.</div></div>
    </div>
  </section>

  <div id="feed"></div>
</div>

<script>
const $ = s => document.querySelector(s);
const el = (t,a={},...k)=>{const n=document.createElement(t);
  for(const x in a){ if(x==="class")n.className=a[x]; else if(x==="html")n.innerHTML=a[x]; else n.setAttribute(x,a[x]); }
  for(const c of k){ if(c!=null) n.append(c.nodeType?c:document.createTextNode(c)); } return n;};
const fx=(v,d=3)=> v==null?"—":Number(v).toFixed(d);
const esc=s=>(s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

let CFG={MAX_ROUNDS:5,EPSILON:0.05,PATIENCE:2,DELTA:0.02,TOP_K:3};
let chosen=null, rounds={}, dists=[], scores=[];

const dz=$("#dz"), fileInput=$("#file");
dz.addEventListener("dragover",e=>{e.preventDefault();dz.classList.add("over");});
dz.addEventListener("dragleave",()=>dz.classList.remove("over"));
dz.addEventListener("drop",e=>{e.preventDefault();dz.classList.remove("over");
  if(e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);});
fileInput.addEventListener("change",()=>{ if(fileInput.files[0]) setFile(fileInput.files[0]); });
function setFile(f){ chosen=f; dz.classList.add("has-file");
  $("#dzlabel").textContent=f.name;
  $("#dzhint").textContent=(f.size/1048576).toFixed(1)+" MB · click to choose another";
  $("#go").disabled=false; }

$("#go").addEventListener("click",start);
$("#again").addEventListener("click",reset);

function setStatus(text,cls){ const s=$("#status"); s.className="status "+(cls||"");
  $("#statustext").textContent=text; }

function reset(){
  $("#feed").innerHTML=""; rounds={}; dists=[]; scores=[];
  $("#again").hidden=true; $("#go").hidden=false; $("#go").disabled=(chosen==null);
  $("#howto").hidden=false; dz.style.pointerEvents="";
  setStatus(chosen? "Ready — "+chosen.name : "Waiting for a clip.","idle");
  window.scrollTo({top:0,behavior:"smooth"});
}

async function start(){
  if(!chosen) return;
  $("#go").disabled=true; $("#go").textContent="Running…"; dz.style.pointerEvents="none";
  $("#howto").hidden=true;
  $("#feed").innerHTML=""; rounds={}; dists=[]; scores=[];
  setStatus("Uploading "+chosen.name+" …","");
  const fd=new FormData(); fd.append("clip",chosen);
  let r;
  try{ r=await (await fetch("/run",{method:"POST",body:fd})).json(); }
  catch(e){ endRun(); return setStatus("Upload failed: "+e,"err"); }
  if(r.error){ endRun(); return setStatus("Error: "+r.error,"err"); }
  const es=new EventSource("/events/"+r.job_id);
  es.onmessage=ev=>{ const d=JSON.parse(ev.data); handle(d);
    if(d.event==="_end"){ es.close(); endRun(); } };
  es.onerror=()=>{ setStatus("Connection lost.","err"); es.close(); endRun(); };
}

function endRun(){
  $("#go").hidden=true; $("#go").textContent="Run the loop";
  $("#again").hidden=false; dz.style.pointerEvents="";
}

function handle(d){
  switch(d.event){
    case "config": CFG=d; renderCfg(d); break;
    case "prepare": setStatus(d.message,""); break;
    case "upload": setStatus(d.status==="done"?"Clip uploaded. Processing…":d.message,""); break;
    case "fresh_eyes":
      if(d.status==="start"){ setStatus("Round 0 · "+d.message,""); }
      else{ $("#feed").append(el("div",{class:"fresh"},
        el("span",{class:"lab"},"fresh-eyes description · the round 1 retrieval query"),
        "“"+d.text+"”")); }
      break;
    case "generate_refs":
      if(d.status==="start"){ setStatus("Round 0 · "+d.message,""); break; }
      if(!d.examples || !d.examples.length) break;
      { const box=el("div",{class:"fresh"});
        box.append(el("span",{class:"lab"},
          `${d.examples.length} reference examples generated for this clip · added to the library, tagged "generated" wherever shown`));
        d.examples.forEach(e=> box.append(el("div",{style:"font-size:12.5px;margin-top:6px"},
          el("b",{},e.category+": "), e.text)));
        $("#feed").append(box); }
      break;
    case "round": ensureRound(d.round); setStatus(`Round ${d.round} of ${d.max_rounds} …`,""); break;
    case "retrieve": onRetrieve(d); break;
    case "write": onWrite(d); break;
    case "measure": onMeasure(d); break;
    case "critique": onCritique(d); break;
    case "decide": onDecide(d); break;
    case "done": onDone(d); break;
    case "saved":
      { const f=document.querySelector(".final");
        if(f) f.append(el("div",{style:"margin-top:12px;font-size:11.5px;color:var(--ink-faint)"},
          "Saved to results/"+d.file+" — also in the dashboard at /dashboard")); }
      break;
    case "error": {
      setStatus("Something went wrong — see below","err");
      const box = el("div",{class:"round",style:"border-color:var(--stop)"});
      box.append(el("div",{class:"steps"},
        el("div",{class:"step"},
          el("div",{class:"slab",style:"color:var(--stop)"}, "Error"),
          el("div",{style:"font-size:13.5px;color:var(--ink)"}, d.message))));
      if(d.trace){
        const det = el("details",{style:"padding:0 18px 14px"});
        det.append(el("summary",{style:"cursor:pointer;font-size:12px;color:var(--ink-faint)"}, "technical details"));
        det.append(el("pre",{style:"font-size:11px;color:var(--ink-soft);white-space:pre-wrap;overflow-x:auto"}, d.trace));
        box.append(det);
      }
      $("#feed").append(box);
      break;
    }
  }
}

function ensureRound(n){
  if(rounds[n]) return rounds[n];
  const steps=["1 · retrieve","2 · write","3 · measure","4 · critique","5 · decide"];
  const box=el("div",{class:"round"});
  box.append(el("div",{class:"rhead"},
    el("span",{class:"rtitle"},"Round "+n),
    el("div",{class:"rchips",id:"chips-"+n})));
  const map={};
  for(const s of steps){
    const key=s.split(" ")[2];
    const row=el("div",{class:"step pending",id:`step-${n}-${key}`},
      el("div",{class:"slab"},s),
      el("div",{class:"sbody"},"waiting…"));
    box.append(row); map[key]=row;
  }
  $("#feed").append(box);
  rounds[n]={box,map};
  return rounds[n];
}
function fill(n,key,node){ const r=ensureRound(n); const row=r.map[key];
  row.classList.remove("pending"); const b=row.querySelector(".sbody");
  b.innerHTML=""; b.append(node); }
function chip(n,html,cls){ $("#chips-"+n).append(el("span",{class:"chip "+(cls||""),html:html})); }

function onRetrieve(d){
  if(d.status==="start"){ setStatus(`Round ${d.round} · ${d.message}`,"");
    fill(d.round,"retrieve", el("div",{class:"qsrc",html:
      `query &larr; <b>${esc(d.query_source)}</b>`})); return; }
  const wrap=el("div",{});
  wrap.append(el("div",{class:"qsrc",html:`query &larr; <b>${esc(d.query_source)}</b>`}));
  wrap.append(el("div",{class:"qtext"},"“"+d.query+"”"));
  const docs=el("div",{class:"docs"});
  d.retrieved.forEach((doc,i)=>{
    const dd=el("div",{class:"doc"+(i===0?" top":"")});
    const h=el("div",{class:"dh"});
    h.append(el("span",{class:"did"},doc.id));
    h.append(el("span",{class:"dcat"},doc.category));
    if(doc.source==="generated") h.append(el("span",{class:"tag gen"},"WRITTEN FOR THIS CLIP"));
    if(i===0) h.append(el("span",{class:"tag"},"NEAREST"));
    const sim=el("span",{class:"dsim"});
    sim.append(el("span",{class:"simbar"}, el("i",{style:`width:${Math.max(3,Math.round(doc.similarity*100))}%`})));
    sim.append("cos "+fx(doc.similarity,2));
    h.append(sim); dd.append(h);
    dd.append(el("div",{class:"dtext"},doc.text));
    docs.append(dd);
  });
  wrap.append(docs);
  fill(d.round,"retrieve",wrap);
}

function onWrite(d){
  if(d.status==="start"){ setStatus(`Round ${d.round} · ${d.message}`,"");
    fill(d.round,"write",el("span",{},"drafting…")); return; }
  fill(d.round,"write",el("div",{class:"draft"},d.draft));
}

function onMeasure(d){
  const r=ensureRound(d.round);
  if(d.distance==null){ fill(d.round,"measure",el("span",{class:"note"},"first draft — no distance yet")); return; }
  dists.push(d.distance);
  const eps=d.epsilon, max=Math.max(0.4,d.distance*1.25), W=380,H=44,pL=8,pR=8;
  const x=v=>pL+v/max*(W-pL-pR);
  const below=d.distance<eps;
  const svg=`<svg viewBox="0 0 ${W} ${H}">
    <line x1="${pL}" y1="26" x2="${W-pR}" y2="26" stroke="var(--line-strong)" stroke-width="2"/>
    <line x1="${x(eps)}" y1="14" x2="${x(eps)}" y2="38" stroke="var(--warn)" stroke-width="2"/>
    <text x="${x(eps)}" y="11" text-anchor="middle" font-size="9" fill="var(--warn)" font-family="IBM Plex Mono,monospace">ε ${eps}</text>
    <circle cx="${x(d.distance)}" cy="26" r="6" fill="${below?'var(--ok)':'var(--accent)'}" stroke="var(--surface)" stroke-width="2"/>
    <text x="${x(d.distance)}" y="42" text-anchor="middle" font-size="9.5" fill="var(--ink)" font-family="IBM Plex Mono,monospace">dₜ ${fx(d.distance)}</text>
  </svg>`;
  const wrap=el("div",{class:"gauge",html:svg});
  wrap.append(el("div",{class:"note",html: below
    ? `moved <b>${fx(d.distance)}</b> — <b>below ε</b>, counts as converged this round`
    : `moved <b>${fx(d.distance)}</b> in meaning from the round ${d.round-1} draft — still above ε`}));
  fill(d.round,"measure",wrap);
  chip(d.round,"dₜ "+fx(d.distance));
}

function onCritique(d){
  if(d.status==="start"){ setStatus(`Round ${d.round} · ${d.message}`,"");
    fill(d.round,"critique",el("span",{},"scoring…")); return; }
  scores.push(d.score);
  const wrap=el("div",{});
  wrap.append(el("div",{class:"draft",style:"background:transparent;border-style:dashed"},d.critique||"—"));
  wrap.append(el("div",{class:"note",html:
    `score <b>qₜ = ${fx(d.score,2)}</b>`+
    (d.delta!=null?` · Δq ${d.delta>=0?"+":""}${fx(d.delta,3)} vs previous`:"")+
    ` · approved: <b>${d.approved}</b>`}));
  fill(d.round,"critique",wrap);
  chip(d.round,"qₜ "+fx(d.score,2));
}

function onDecide(d){
  const lastK=dists.slice(-CFG.PATIENCE);
  const converged=lastK.length>=CFG.PATIENCE && lastK.every(x=>x<CFG.EPSILON);
  const gain=scores.length>=2?scores[scores.length-1]-scores[scores.length-2]:null;
  const stalled=gain!=null && gain<CFG.DELTA;
  const checks=[
    ["1 critic", d.reason==="critic",
      d.reason==="critic"?"Critic approved":"Critic did not approve"],
    ["2 no_gain", d.reason==="no_gain",
      !converged?`only ${lastK.filter(x=>x<CFG.EPSILON).length}/${CFG.PATIENCE} recent distances < ε`
       :!stalled?`score still gaining (Δq ${gain>=0?"+":""}${fx(gain,3)} ≥ δ)`
       :`distances < ε for ${CFG.PATIENCE} rounds and Δq < δ`],
    ["3 failsafe", d.reason==="failsafe",
      d.reason==="failsafe"?`round ${CFG.MAX_ROUNDS} reached`:`round ${d.round} of ${CFG.MAX_ROUNDS}`],
  ];
  const box=el("div",{class:"checks"});
  for(const [n,hit,why] of checks)
    box.append(el("div",{class:"check "+(hit?"hit":"miss")},
      el("span",{class:"cm"},hit?"◉":"○"), el("span",{class:"cn"},n), el("span",{class:"cw"},why)));
  const halted=d.next_action==="halt";
  box.append(el("div",{class:"verdict "+(halted?"halt":"go")},
    halted?`cascade → HALT · ${d.reason}`:`cascade → continue to round ${d.round+1}`));
  fill(d.round,"decide",box);
  chip(d.round, halted?"HALT · "+d.reason:"continue", halted?"halt":"go");
  chip(d.round, d.cumulative_tokens.toLocaleString()+" tok");
}

function onDone(d){
  setStatus(`Done — ${d.rounds_used} round${d.rounds_used>1?"s":""}, stopped: ${d.stop_reason}`,"done");
  const f=el("div",{class:"final"});
  f.append(el("div",{class:"lab"},"final summary · round "+d.rounds_used));
  f.append(el("p",{},d.final_summary));
  const oc=el("div",{class:"oc"});
  oc.append(el("span",{class:"chip reason-"+d.stop_reason},"stopped: "+d.stop_reason));
  oc.append(el("span",{class:"chip"},"final qₜ "+fx(d.final_score,2)));
  oc.append(el("span",{class:"chip"},d.usage.calls+" API calls"));
  oc.append(el("span",{class:"chip"},d.usage.total_tokens.toLocaleString()+" tokens"));
  f.append(oc);
  $("#feed").append(f);
  f.scrollIntoView({behavior:"smooth",block:"nearest"});
}

function renderCfg(c){
  const cfg=$("#cfg"); cfg.innerHTML="";
  const rows=[["MAX_ROUNDS",c.MAX_ROUNDS],["ε",c.EPSILON],["patience",c.PATIENCE],
    ["δ",c.DELTA],["top-k",c.TOP_K]];
  for(const [k,v] of rows) cfg.append(el("span",{html:`${k} <b>${v}</b>`}));
  if(c.MODEL) cfg.append(el("span",{html:`<b>${c.MODEL}</b>`}));
  if(c.EMBED_MODEL) cfg.append(el("span",{html:`embed <b>${c.EMBED_MODEL}</b>`}));
}
renderCfg(CFG);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
