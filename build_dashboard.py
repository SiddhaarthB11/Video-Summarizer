"""Generate a self-contained HTML walkthrough of the halting runs in results/.

    python build_dashboard.py

Reads every results/<clip>__<policy>.json plus the small preview clips in
assets/, and writes dashboard.html with everything inlined (data + video as
base64). No server, no external assets. Also writes dashboard.artifact.html
(body-only) for publishing as an Artifact.
"""

from __future__ import annotations

import base64
import datetime as dt
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "dashboard.html")
OUT_ARTIFACT = os.path.join(HERE, "dashboard.artifact.html")


def _clip_id(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def load() -> dict:
    runs: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        base = os.path.basename(path)
        if base == "summary.json":
            continue
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cid = _clip_id(data["video"])
        runs.setdefault(cid, {"id": cid, "name": data["video"], "policies": {}})
        runs[cid]["policies"][data["policy"]] = data

    clips = []
    for cid, entry in runs.items():
        uri = ""
        for ext in (".mp4", ".webm", ".mov"):
            vp = os.path.join(ASSETS, cid + ext)
            if os.path.exists(vp):
                mime = "video/mp4" if ext == ".mp4" else f"video/{ext[1:]}"
                b64 = base64.b64encode(open(vp, "rb").read()).decode("ascii")
                uri = f"data:{mime};base64,{b64}"
                break
        any_policy = next(iter(entry["policies"].values()))
        clips.append(
            {
                "id": cid,
                "name": entry["name"],
                "video": uri,
                "fresh_eyes": any_policy.get("fresh_eyes_query", ""),
                "policies": entry["policies"],
            }
        )

    cfg = next(iter(clips[0]["policies"].values()))["config"] if clips else {}
    return {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": cfg,
        "clips": clips,
    }


def main() -> None:
    data = load()
    if not data["clips"]:
        raise SystemExit("no results found - run `python run.py` first")
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data))

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)

    head = html.split("<!--HEAD_START-->", 1)[1].split("<!--HEAD_END-->", 1)[0]
    body = html.split("<!--BODY_START-->", 1)[1].split("<!--BODY_END-->", 1)[0]
    with open(OUT_ARTIFACT, "w", encoding="utf-8") as fh:
        fh.write(head.strip() + "\n" + body.strip() + "\n")

    for p in (OUT, OUT_ARTIFACT):
        print(f"wrote {os.path.basename(p)}  ({os.path.getsize(p)/1024:.0f} KB, "
              f"{len(data['clips'])} clips)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!--HEAD_START-->
<title>How the AI Summarized Each Clip</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{
  --ground:#f5f7f9; --surface:#ffffff; --surface-2:#eef1f5;
  --ink:#1a1f2b; --ink-soft:#5a6472; --ink-faint:#8b94a3;
  --line:#e0e4ea; --line-strong:#c9d0da;
  --accent:#166b78; --accent-soft:#e0eef0;
  --ok:#3f7d58; --ok-soft:#e3efe7; --warn:#a9761c; --stop:#a5402f;
  --shadow:0 1px 2px rgba(20,30,45,.05), 0 8px 24px rgba(20,30,45,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0f1218; --surface:#161b23; --surface-2:#1d2430;
    --ink:#e7eaf0; --ink-soft:#9aa4b3; --ink-faint:#6b7585;
    --line:#242c39; --line-strong:#333d4e;
    --accent:#54b9c6; --accent-soft:#12333a;
    --ok:#5fae7f; --ok-soft:#173226; --warn:#d19a4a; --stop:#d4715f;
    --shadow:0 1px 2px rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0f1218; --surface:#161b23; --surface-2:#1d2430;
  --ink:#e7eaf0; --ink-soft:#9aa4b3; --ink-faint:#6b7585;
  --line:#242c39; --line-strong:#333d4e;
  --accent:#54b9c6; --accent-soft:#12333a;
  --ok:#5fae7f; --ok-soft:#173226; --warn:#d19a4a; --stop:#d4715f;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  line-height:1.6; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:880px; margin:0 auto; padding:40px 24px 110px}
h1,h2,h3{text-wrap:balance; line-height:1.25; margin:0}
.eyebrow{
  font-family:"IBM Plex Mono",monospace; font-size:11px; font-weight:600;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent);
}
header.top{border-bottom:1px solid var(--line); padding-bottom:24px; margin-bottom:8px}
header.top h1{font-size:28px; font-weight:600; letter-spacing:-.015em; margin:10px 0 10px}
header.top p{margin:0; color:var(--ink-soft); max-width:64ch; font-size:15px}
.gen{margin-top:14px; font-size:12px; color:var(--ink-faint)}

section{margin:34px 0}
section > h2{
  font-size:12px; font-family:"IBM Plex Mono",monospace; font-weight:600;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink-soft);
  padding-bottom:8px; border-bottom:1px solid var(--line); margin-bottom:20px;
}

/* how it works */
.how{display:grid; grid-template-columns:repeat(4,1fr); gap:10px}
@media (max-width:680px){.how{grid-template-columns:1fr 1fr}}
.how .card{background:var(--surface); border:1px solid var(--line); border-radius:11px; padding:14px}
.how .card .n{font-family:"IBM Plex Mono",monospace; font-size:10px; font-weight:600; color:var(--accent); letter-spacing:.06em}
.how .card .t{font-size:13.5px; font-weight:600; margin:4px 0 5px}
.how .card .d{font-size:12px; color:var(--ink-soft); line-height:1.5}
.stopwhen{margin-top:14px; font-size:13px; color:var(--ink-soft); background:var(--surface-2);
  border:1px solid var(--line); border-radius:10px; padding:12px 15px}
.stopwhen b{color:var(--ink)}

/* clip switcher */
.tabs{display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px}
.tab{
  font:inherit; font-size:12.5px; cursor:pointer;
  background:var(--surface); color:var(--ink-soft);
  border:1px solid var(--line-strong); border-radius:999px; padding:6px 15px;
}
.tab[aria-selected="true"]{background:var(--ink); color:var(--ground); border-color:var(--ink)}
.tab:focus-visible{outline:2px solid var(--accent); outline-offset:2px}

.toggle{display:inline-flex; border:1px solid var(--line-strong); border-radius:9px; overflow:hidden; margin-bottom:22px}
.toggle button{
  font:inherit; font-size:12.5px; cursor:pointer; padding:8px 16px; border:0;
  background:var(--surface); color:var(--ink-soft);
}
.toggle button[aria-pressed="true"]{background:var(--accent); color:#fff}
.toggle button:focus-visible{outline:2px solid var(--accent); outline-offset:-2px}

/* clip header */
.cliphead{display:grid; grid-template-columns:300px 1fr; gap:24px; margin-bottom:22px}
@media (max-width:760px){.cliphead{grid-template-columns:1fr}}
.cliphead video{width:100%; border-radius:10px; border:1px solid var(--line); background:#000; display:block}
.cliphead .fname{font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--ink-faint); margin-top:8px; word-break:break-all}
.firstlook{margin:0 0 14px; font-size:13.5px; color:var(--ink-soft)}
.firstlook b{color:var(--ink); font-weight:600}
.outcome{display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px}
.chip{
  font-size:12px; font-weight:500;
  padding:5px 11px; border-radius:7px; background:var(--surface-2); color:var(--ink-soft);
  border:1px solid var(--line); display:inline-flex; gap:6px; align-items:center;
}
.chip b{color:var(--ink); font-weight:600}
.chip.good{background:var(--ok-soft); border-color:var(--ok); color:var(--ok)}
.chip.cap{background:color-mix(in srgb,var(--warn) 14%,transparent); border-color:var(--warn); color:var(--warn)}
.verdict{
  font-size:13.5px; padding:12px 15px; border-radius:9px; background:var(--accent-soft);
  border:1px solid color-mix(in srgb,var(--accent) 30%,transparent); color:var(--ink);
}

/* rating chart */
.ratingbox{background:var(--surface); border:1px solid var(--line); border-radius:11px;
  padding:15px 16px 10px; margin-bottom:26px; box-shadow:var(--shadow)}
.ratingbox .rt{font-size:12.5px; color:var(--ink-soft); margin-bottom:4px; font-weight:500}
.ratingbox svg{width:100%; height:auto; display:block; overflow:visible}

/* rounds */
.round{
  background:var(--surface); border:1px solid var(--line); border-radius:12px;
  margin-bottom:14px; overflow:hidden; box-shadow:var(--shadow);
}
.round > summary{
  list-style:none; cursor:pointer; padding:14px 18px; display:flex;
  align-items:center; gap:12px; border-bottom:1px solid transparent; flex-wrap:wrap;
}
.round[open] > summary{border-bottom-color:var(--line)}
.round > summary::-webkit-details-marker{display:none}
.round .rlabel{font-weight:600; font-size:14px}
.round .rmeta{margin-left:auto; display:flex; gap:7px; flex-wrap:wrap}
.mchip{
  font-size:11px; padding:3px 9px; border-radius:6px;
  background:var(--surface-2); color:var(--ink-soft); border:1px solid var(--line);
}
.mchip.stop{background:color-mix(in srgb,var(--stop) 14%,transparent); color:var(--stop); border-color:var(--stop)}
.mchip.go{background:var(--ok-soft); color:var(--ok); border-color:var(--ok)}

.steps{padding:4px 18px 16px}
.step{padding:15px 0; border-top:1px solid var(--line)}
.step:first-child{border-top:0}
.step .slab{
  font-size:11px; font-weight:600; letter-spacing:.02em;
  color:var(--ink-faint); text-transform:uppercase; margin-bottom:8px;
}
.summary-text{
  font-size:14px; line-height:1.6; background:var(--surface-2); border:1px solid var(--line);
  border-radius:9px; padding:12px 14px; color:var(--ink);
}
.lookup{font-size:12.5px; color:var(--ink-soft); margin-bottom:8px}
.lookup b{color:var(--ink)}
.docs{display:flex; flex-direction:column; gap:8px}
.doc{border:1px solid var(--line); border-radius:9px; padding:10px 12px; background:var(--surface)}
.doc.top{border-color:var(--accent); background:var(--accent-soft)}
.doc .dhead{display:flex; align-items:center; gap:8px; margin-bottom:4px; flex-wrap:wrap}
.doc .dcat{font-size:11px; font-weight:600; color:var(--accent); text-transform:capitalize}
.doc .closest{font-size:9.5px; font-weight:700; letter-spacing:.06em; color:#fff; background:var(--accent); padding:2px 6px; border-radius:4px}
.doc .dtext{font-size:12.5px; color:var(--ink-soft); line-height:1.5}
.note{font-size:12.5px; color:var(--ink-soft); margin-top:8px}
.note b{color:var(--ink)}

.rating-inline{display:flex; align-items:center; gap:10px; margin-top:10px; font-size:12.5px; color:var(--ink-soft)}
.rating-inline .bar{flex:1; max-width:220px; height:7px; border-radius:4px; background:var(--surface-2); overflow:hidden; border:1px solid var(--line)}
.rating-inline .bar i{display:block; height:100%; background:var(--accent)}
.rating-inline .pct{font-weight:600; color:var(--ink)}
.approved{color:var(--ok); font-weight:600}

.decision{margin-top:4px; font-size:13px; font-weight:500; padding:11px 14px; border-radius:9px}
.decision.go{background:var(--surface-2); color:var(--ink-soft); border:1px solid var(--line)}
.decision.stop{background:var(--ok-soft); color:var(--ok); border:1px solid var(--ok)}

.finalsum{
  margin-top:14px; border:1px solid var(--accent); background:var(--accent-soft);
  border-radius:11px; padding:16px 18px;
}
.finalsum .flab{font-size:10.5px; letter-spacing:.09em; text-transform:uppercase; color:var(--accent); font-weight:600}
.finalsum p{margin:7px 0 0; font-size:15px; line-height:1.62}

.foot{margin-top:56px; padding-top:20px; border-top:1px solid var(--line); font-size:12px; color:var(--ink-faint)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
video{max-width:100%}
</style>
<!--HEAD_END-->
</head>
<body>
<!--BODY_START-->
<div class="wrap">
<header class="top">
  <div class="eyebrow">Video summarizer</div>
  <h1>How the AI summarized each clip</h1>
  <p>For every clip, the AI wrote a summary, checked it against strong human-written
  examples, had a reviewer give notes, and rewrote it &mdash; stopping once the summary
  had settled. Here is each attempt, in order.</p>
  <div class="gen" id="gen"></div>
</header>

<section>
  <h2>How it works</h2>
  <div class="how">
    <div class="card"><div class="n">STEP 1</div><div class="t">Look up examples</div>
      <div class="d">Find a few strong, human-written summaries of similar clips to use as a style guide.</div></div>
    <div class="card"><div class="n">STEP 2</div><div class="t">Write</div>
      <div class="d">Draft the summary from the clip &mdash; or rewrite it using the reviewer&rsquo;s notes.</div></div>
    <div class="card"><div class="n">STEP 3</div><div class="t">Compare</div>
      <div class="d">Check how much the summary changed from the previous version.</div></div>
    <div class="card"><div class="n">STEP 4</div><div class="t">Review</div>
      <div class="d">A reviewer rates the summary against the examples and can approve it.</div></div>
  </div>
  <div class="stopwhen">
    It <b>stops rewriting</b> when the reviewer approves the summary, or when the summary
    stops changing and the rating stops going up, or after 5 rewrites at most.
  </div>
</section>

<section>
  <h2>Each clip, attempt by attempt</h2>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div class="toggle" role="group" aria-label="Which run">
    <button id="btn-halted" aria-pressed="true">Stop when settled</button>
    <button id="btn-baseline" aria-pressed="false">Always rewrite 5&times;</button>
  </div>
  <div id="clipview"></div>
</section>

<div class="foot" id="foot"></div>
</div>

<script>
const DATA = /*__DATA__*/;
const C = DATA.config || {};
const EPS = C.EPSILON ?? 0.05, MAXR = C.MAX_ROUNDS ?? 5;
let curClip = DATA.clips[0].id;
let curPolicy = "halted";

const $ = (s,r=document)=>r.querySelector(s);
const el = (t,a={},...kids)=>{const n=document.createElement(t);
  for(const k in a){ if(k==="class")n.className=a[k]; else if(k==="html")n.innerHTML=a[k]; else n.setAttribute(k,a[k]); }
  for(const c of kids){ if(c!=null) n.append(c.nodeType?c:document.createTextNode(c)); } return n;};
const esc=s=>(s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = s => s==null ? "—" : Math.round(s*100)+"%";

$("#gen").textContent = `${DATA.clips.length} clips · generated ${DATA.generated}`;

// how much did the summary move, in words (no numbers)
function changeLabel(d){
  if(d==null) return "First draft";
  if(d < EPS) return "Barely changed";
  if(d < 0.15) return "Changed a little";
  return "Changed a lot";
}
function changeSentence(d, prevRound){
  if(d==null) return "This is the first attempt.";
  if(d < EPS) return `Almost identical to attempt ${prevRound} — just small wording tweaks.`;
  if(d < 0.15) return `A little different from attempt ${prevRound}.`;
  return `Quite different from attempt ${prevRound} — the AI added or reworked real content.`;
}
// plain-language stop reason
function stopPlain(reason){
  return ({
    critic: "the reviewer approved it",
    no_gain: "the summary settled and the rating stopped rising",
    failsafe: "it hit the 5-rewrite limit",
    baseline_cap: "this run always does all 5 rewrites",
  })[reason] || reason;
}
// did THIS run stop at this round?
function stoppedHere(run, i){ return i === run.rounds.length - 1; }

// tabs
const tabsEl = $("#tabs");
DATA.clips.forEach((c,i)=>{
  const b = el("button",{class:"tab",role:"tab",id:"tab-"+c.id,"aria-selected":String(i===0)},
    c.name.replace(/\.(mp4|webm|mov)$/i,""));
  b.onclick = ()=>{ curClip=c.id; syncTabs(); render(); };
  tabsEl.append(b);
});
function syncTabs(){ [...tabsEl.children].forEach(b=> b.setAttribute("aria-selected", String(b.id==="tab-"+curClip))); }

$("#btn-halted").onclick = ()=>setPolicy("halted");
$("#btn-baseline").onclick = ()=>setPolicy("baseline");
function setPolicy(p){ curPolicy=p;
  $("#btn-halted").setAttribute("aria-pressed",String(p==="halted"));
  $("#btn-baseline").setAttribute("aria-pressed",String(p==="baseline"));
  render();
}

// simple 0-100% rating line across attempts
function ratingChart(scores, stopIdx){
  const W=560,H=120,pL=36,pR=14,pT=14,pB=24;
  const n=scores.length;
  const x=i=> pL + (n<=1?0:i/(n-1))*(W-pL-pR);
  const y=v=> H-pB - v*(H-pT-pB);
  let g="";
  for(let i=0;i<=2;i++){ const v=i/2;
    g+=`<line x1="${pL}" y1="${y(v)}" x2="${W-pR}" y2="${y(v)}" stroke="var(--line)"/>`
      +`<text x="${pL-8}" y="${y(v)+3}" text-anchor="end" font-size="9.5" fill="var(--ink-faint)">${v*100}%</text>`;
  }
  scores.forEach((s,i)=> g+=`<text x="${x(i)}" y="${H-7}" text-anchor="middle" font-size="9.5" fill="var(--ink-faint)">#${i+1}</text>`);
  const path = scores.map((s,i)=> (i?"L":"M")+x(i)+" "+y(s)).join(" ");
  g+=`<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>`;
  scores.forEach((s,i)=>{ const hl=i===stopIdx;
    g+=`<circle cx="${x(i)}" cy="${y(s)}" r="${hl?5:3.5}" fill="${hl?'var(--ok)':'var(--accent)'}" stroke="var(--surface)" stroke-width="1.5"/>`;
    g+=`<text x="${x(i)}" y="${y(s)-10}" text-anchor="middle" font-size="9.5" font-weight="600" fill="var(--ink)">${Math.round(s*100)}%</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Reviewer rating across attempts">${g}</svg>`;
}

function render(){
  const clip = DATA.clips.find(c=>c.id===curClip);
  const run = clip.policies[curPolicy];
  const other = clip.policies[curPolicy==="halted"?"baseline":"halted"];
  const view = $("#clipview");
  view.innerHTML = "";

  // header
  const head = el("div",{class:"cliphead"});
  const vwrap = el("div",{});
  if(clip.video) vwrap.append(el("video",{src:clip.video,controls:"",muted:"",loop:"",playsinline:""}));
  vwrap.append(el("div",{class:"fname"}, clip.name));
  head.append(vwrap);

  const info = el("div",{});
  info.append(el("p",{class:"firstlook",html:
    `<b>First glance at the clip:</b> &ldquo;${esc(clip.fresh_eyes)}&rdquo;`}));

  const oc = el("div",{class:"outcome"});
  const nR = run.rounds_used;
  oc.append(el("span",{class:"chip"}, el("b",{},String(nR)), nR===1?" attempt":" attempts"));
  if(curPolicy==="baseline"){
    oc.append(el("span",{class:"chip"}, "no early stopping"));
  } else {
    oc.append(el("span",{class:"chip "+(run.stop_reason==="critic"?"good":run.stop_reason==="failsafe"?"cap":"")},
      "stopped because ", el("b",{}, stopPlain(run.stop_reason))));
  }
  oc.append(el("span",{class:"chip"}, "final rating ", el("b",{}, pct(run.final_score))));
  info.append(oc);

  if(other){
    const dr=run.rounds_used, db=other.rounds_used;
    let msg;
    if(curPolicy==="halted"){
      if(dr<db){
        msg = `It stopped after <b>${dr}</b> ${dr===1?"attempt":"attempts"} instead of ${db} &mdash; about ${Math.round((db-dr)/db*100)}% less work &mdash; because ${stopPlain(run.stop_reason)}. Final rating ${pct(run.final_score)} vs ${pct(other.final_score)} for the always-rewrite run.`;
      } else if(run.stop_reason==="critic"){
        msg = `It went the full ${dr} attempts here &mdash; the reviewer only approved the summary on the last pass. Final rating ${pct(run.final_score)} vs ${pct(other.final_score)}.`;
      } else {
        msg = `It went the full ${dr} attempts &mdash; the summary kept changing enough that it never clearly settled, so it ran to the limit. Final rating ${pct(run.final_score)} vs ${pct(other.final_score)}.`;
      }
    } else {
      const oR = other.rounds_used;
      msg = oR<dr
        ? `This run always rewrites 5 times, whatever the reviewer says. The &ldquo;stop when settled&rdquo; run finished this clip in just <b>${oR}</b> ${oR===1?"attempt":"attempts"} &mdash; ${stopPlain(other.stop_reason)}.`
        : `This run always rewrites 5 times, whatever the reviewer says. The &ldquo;stop when settled&rdquo; run also went the full ${oR} here (${stopPlain(other.stop_reason)}).`;
    }
    info.append(el("div",{class:"verdict",html:msg}));
  }
  head.append(info);
  view.append(head);

  // rating chart
  const scores = run.rounds.map(r=>r.score);
  view.append(el("div",{class:"ratingbox"},
    el("div",{class:"rt"}, "Reviewer rating, attempt by attempt"),
    el("div",{html: ratingChart(scores, run.rounds_used-1)})));

  // rounds
  run.rounds.forEach((r,i)=>{
    const dt = el("details",{class:"round"});
    dt.setAttribute("open","");
    const stopped = (curPolicy==="baseline") ? (i===run.rounds.length-1) : stoppedHere(run,i);

    const sum = el("summary",{});
    sum.append(el("span",{class:"rlabel"}, i===0 ? "First draft" : "Rewrite "+i));
    const meta = el("div",{class:"rmeta"});
    meta.append(el("span",{class:"mchip"}, changeLabel(r.distance)));
    meta.append(el("span",{class:"mchip"}, "rated "+pct(r.score)));
    meta.append(el("span",{class:"mchip "+(stopped?"stop":"go")}, stopped ? "stopped here" : "kept going"));
    sum.append(meta);
    dt.append(sum);

    const steps = el("div",{class:"steps"});

    // examples it looked up
    const docs = el("div",{class:"docs"});
    r.retrieved.forEach((doc,di)=>{
      const dd = el("div",{class:"doc"+(di===0?" top":"")});
      const h = el("div",{class:"dhead"});
      h.append(el("span",{class:"dcat"}, String(doc.category||"example").replace(/-/g," ")));
      if(di===0) h.append(el("span",{class:"closest"}, "CLOSEST MATCH"));
      dd.append(h);
      dd.append(el("div",{class:"dtext"}, doc.text));
      docs.append(dd);
    });
    const lookedFor = i===0 ? "what it saw in the clip" : "the previous summary";
    steps.append(el("div",{class:"step"},
      el("div",{class:"slab"}, "Examples it looked up"),
      el("div",{class:"lookup",html:`It searched for strong summaries similar to <b>${lookedFor}</b>, and found:`}),
      docs));

    // the summary it wrote
    steps.append(el("div",{class:"step"},
      el("div",{class:"slab"}, i===0 ? "The summary it wrote" : "The rewrite"),
      el("div",{class:"summary-text"}, r.draft),
      el("div",{class:"note"}, changeSentence(r.distance, r.round-1))));

    // reviewer
    const rv = el("div",{class:"step"},
      el("div",{class:"slab"}, "Reviewer’s notes"),
      el("div",{class:"summary-text",style:"background:transparent;border-style:dashed"}, r.critique||"—"));
    const ri = el("div",{class:"rating-inline"});
    ri.append(el("span",{}, "Rating"));
    ri.append(el("span",{class:"bar"}, el("i",{style:`width:${Math.round(r.score*100)}%`})));
    ri.append(el("span",{class:"pct"}, pct(r.score)));
    if(r.approved) ri.append(el("span",{class:"approved"}, "✓ approved"));
    rv.append(ri);
    steps.append(rv);

    // decision
    let decTxt, decCls;
    if(stopped){
      decCls = "stop";
      decTxt = curPolicy==="baseline" && i===run.rounds.length-1 && run.stop_reason==="baseline_cap"
        ? "Done — this run always stops after 5 rewrites."
        : "Stopped here — " + stopPlain(run.stop_reason) + ".";
    } else {
      decCls = "go";
      decTxt = r.approved
        ? "The reviewer approved this, but this run ignores approval and keeps rewriting."
        : "Not done yet — the summary is still improving, so it gets another pass.";
    }
    steps.append(el("div",{class:"step"},
      el("div",{class:"slab"}, "Decision"),
      el("div",{class:"decision "+decCls}, decTxt)));

    dt.append(steps);
    view.append(dt);
  });

  // final
  view.append(el("div",{class:"finalsum"},
    el("div",{class:"flab"}, "Final summary"),
    el("p",{}, run.final_summary)));
}

$("#foot").textContent =
  "The examples come from a small hand-written library of strong summaries. "+
  "Ratings are the reviewer AI's own judgement and can be noisy. "+
  "This is an exploratory test on a few clips, not a formal study.";

render();
</script>
<!--BODY_END-->
</body>
</html>
"""


if __name__ == "__main__":
    main()
