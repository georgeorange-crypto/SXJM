"""Render recorded runs as a single self-contained HTML file.

Design choices:

* **No server, no build step.** The trace data is embedded as a JSON blob in a
  ``<script type="application/json">`` tag and the page draws it with vanilla
  canvas 2D. Double-click the file; it works offline. This dodges every
  file://-origin CORS problem a fetch-based viewer would hit on Windows.
* **One canvas, many layers.** Bottom to top: arena disc, true sources (grey
  until cleared, then their channel colour), detection fans (a translucent
  wedge per scan showing the +/-1 deg bearing uncertainty), the robot route
  poly-line, and the robot marker. A time slider / play button reveals frames
  incrementally so the search unfolds.
* **Algorithm switch.** A dropdown selects which recorded run is shown; when the
  runs share a seed the jammer layout is identical, so switching compares routes
  on the same map.

The whole viewer is authored here as a template string. It is deliberately
dependency-free (no CDN, no framework) so it renders identically in the judges'
browser and inside the supporting-material archive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .recorder import RunRecord

_TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0e1116; --panel: #161b22; --ink: #e6edf3; --muted: #8b949e;
    --line: #30363d; --accent: #58a6ff; --grey: #6e7681;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  }
  header { padding: 14px 18px; border-bottom: 1px solid var(--line); }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  header p { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
  .wrap { display: flex; gap: 16px; padding: 16px; flex-wrap: wrap; }
  .stage { flex: 1 1 620px; min-width: 320px; }
  canvas { width: 100%; height: auto; background: #0a0d12;
           border: 1px solid var(--line); border-radius: 8px; display: block; }
  .side { flex: 0 0 300px; display: flex; flex-direction: column; gap: 14px; }
  .card { background: var(--panel); border: 1px solid var(--line);
          border-radius: 8px; padding: 12px 14px; }
  .card h2 { margin: 0 0 8px; font-size: 12px; letter-spacing: .04em;
             text-transform: uppercase; color: var(--muted); }
  label { display: block; font-size: 12px; color: var(--muted); margin: 8px 0 2px; }
  select, button {
    background: #21262d; color: var(--ink); border: 1px solid var(--line);
    border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer;
  }
  select { width: 100%; }
  .controls { display: flex; gap: 8px; align-items: center; margin-top: 6px; }
  input[type=range] { width: 100%; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  td { padding: 3px 0; }
  td:last-child { text-align: right; font-variant-numeric: tabular-nums; color: #fff; }
  .legend { display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
  .legend .row { display: flex; align-items: center; gap: 8px; }
  .sw { width: 14px; height: 14px; border-radius: 3px; flex: none; }
  .sw.fan { background: linear-gradient(90deg,#58a6ff88,#58a6ff11); }
  .sw.route { background: none; border-top: 3px solid #f0883e; height: 0; border-radius: 0; }
  .sw.grey { background: var(--grey); border-radius: 50%; }
  .sw.cleared { background: #3fb950; border-radius: 50%; }
  .sw.near { background: #f85149; border-radius: 50%; }
  .hint { color: var(--muted); font-size: 11px; margin-top: 4px; }
  .frameinfo { font-variant-numeric: tabular-nums; }
  code { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <p>灰色 = 未被发现的干扰源 · 彩色扇形 = 机器狗探测痕迹（示向度 ±1°）· 橙线 = 前进路线 · ✕/绿点 = 已清除</p>
</header>
<div class="wrap">
  <div class="stage">
    <canvas id="cv" width="900" height="900"></canvas>
    <div class="controls">
      <button id="play">▶ 播放</button>
      <input id="slider" type="range" min="0" max="0" value="0" step="1">
    </div>
    <div class="hint frameinfo" id="frameinfo"></div>
  </div>
  <div class="side">
    <div class="card">
      <h2>算法</h2>
      <select id="algo"></select>
      <label>叠加对比</label>
      <div class="controls">
        <label style="margin:0"><input type="checkbox" id="ghost"> 显示其它算法路线（淡）</label>
      </div>
    </div>
    <div class="card">
      <h2>本局统计</h2>
      <table id="stats"></table>
    </div>
    <div class="card">
      <h2>图例</h2>
      <div class="legend">
        <div class="row"><span class="sw grey"></span>未发现的干扰源（真值）</div>
        <div class="row"><span class="sw cleared"></span>已清除的干扰源</div>
        <div class="row"><span class="sw fan"></span>探测扇形（有示向度）</div>
        <div class="row"><span class="sw near"></span>近距离 / 无示向度检测</div>
        <div class="row"><span class="sw route"></span>机器狗前进路线</div>
      </div>
      <div class="hint">扇形从检测点沿示向度方向张开 ±1°，长度截到有效可视范围。定向源画出 180° 覆盖扇区。</div>
    </div>
  </div>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const algoSel = document.getElementById("algo");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("play");
const ghostChk = document.getElementById("ghost");
const statsEl = document.getElementById("stats");
const frameInfo = document.getElementById("frameinfo");

// distinct, colour-blind-friendlyish palette keyed by channel (1..20)
const CH_COLORS = ["#58a6ff","#f0883e","#3fb950","#db61a2","#e3b341","#a371f7",
  "#39c5cf","#f85149","#7ee787","#ff9bce","#79c0ff","#ffa657","#56d364",
  "#bc8cff","#d29922","#ff7b72","#2ea043","#e685b5","#1f6feb","#f2cc60"];
const chColor = (c) => CH_COLORS[((c|0) - 1 + CH_COLORS.length*3) % CH_COLORS.length];

let R = DATA.runs.length ? DATA.runs[0].arena_radius : 1800;
let cur = 0;          // current run index
let frame = 0;        // frames revealed
let timer = null;

// ---- world<->screen transform (square canvas, [-R,R] both axes; y up) ----
function tf() {
  const S = cv.width, pad = 24, span = 2*R;
  const k = (S - 2*pad) / span;
  return {
    x: (wx) => pad + (wx + R) * k,
    y: (wy) => S - pad - (wy + R) * k,   // flip: north is up
    k,
  };
}

function drawArena(T) {
  ctx.save();
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  // arena disc
  ctx.beginPath();
  ctx.arc(T.x(0), T.y(0), R*T.k, 0, 2*Math.PI);
  ctx.stroke();
  // axes
  ctx.strokeStyle = "#20262d";
  ctx.beginPath();
  ctx.moveTo(T.x(-R), T.y(0)); ctx.lineTo(T.x(R), T.y(0));
  ctx.moveTo(T.x(0), T.y(-R)); ctx.lineTo(T.x(0), T.y(R));
  ctx.stroke();
  ctx.fillStyle = "#586069"; ctx.font = "11px sans-serif";
  ctx.fillText("E +x", T.x(R)-34, T.y(0)-6);
  ctx.fillText("N +y", T.x(0)+6, T.y(R)+14);
  ctx.restore();
}

// which channels are cleared by frame f (inclusive)
function clearedByFrame(run, f) {
  const s = new Set();
  for (let i = 0; i < f && i < run.frames.length; i++) {
    const fr = run.frames[i];
    if (fr.type === "clear" && fr.clear_success) s.add(fr.channel);
  }
  return s;
}

function drawTruth(run, T, cleared) {
  if (!run.truth) return;
  const halfArc = (DATA.meta.directional_half_angle_deg || 90) * Math.PI/180;
  for (const j of run.truth.jammers) {
    const px = T.x(j.x), py = T.y(j.y);
    const isCleared = cleared.has(j.channel);
    // directional coverage sector (faint), only while still relevant
    if (j.kind === "dir" && j.direction_deg != null) {
      const a0 = j.direction_deg * Math.PI/180;
      ctx.save();
      ctx.fillStyle = isCleared ? "#3fb95015" : "#8b949e18";
      ctx.beginPath();
      ctx.moveTo(px, py);
      // canvas angles are clockwise & y-down; negate to match math/y-up
      ctx.arc(px, py, Math.min(j.r_eff, R)*T.k, -(a0+halfArc), -(a0-halfArc));
      ctx.closePath(); ctx.fill();
      ctx.restore();
    }
    // source marker
    ctx.save();
    if (isCleared) {
      ctx.strokeStyle = chColor(j.channel); ctx.lineWidth = 2.5;
      const r = 7;
      ctx.beginPath();
      ctx.moveTo(px-r,py-r); ctx.lineTo(px+r,py+r);
      ctx.moveTo(px+r,py-r); ctx.lineTo(px-r,py+r);
      ctx.stroke();
      ctx.fillStyle = chColor(j.channel);
      ctx.beginPath(); ctx.arc(px,py,3,0,2*Math.PI); ctx.fill();
    } else {
      ctx.fillStyle = "#6e7681";
      ctx.beginPath(); ctx.arc(px,py,6,0,2*Math.PI); ctx.fill();
      ctx.strokeStyle = "#0a0d12"; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.fillStyle = isCleared ? chColor(j.channel) : "#8b949e";
    ctx.font = "10px sans-serif";
    ctx.fillText("CH"+j.channel, px+8, py-8);
    ctx.restore();
  }
}

function drawFans(run, T, upto) {
  const err = run.bearing_error_deg || 1.0;
  const reach = Math.min(1500, R) * T.k;   // visual fan length
  for (let i = 0; i < upto && i < run.frames.length; i++) {
    const fr = run.frames[i];
    if (fr.type !== "scan") continue;
    const ox = T.x(fr.pose[0]), oy = T.y(fr.pose[1]);
    if (fr.result === "signal" && fr.bearing != null) {
      const b = fr.bearing * Math.PI/180;
      const de = err * Math.PI/180;
      const col = chColor(fr.channel);
      const g = ctx.createRadialGradient(ox,oy,0,ox,oy,reach);
      g.addColorStop(0, col+"55"); g.addColorStop(1, col+"05");
      ctx.save();
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.arc(ox, oy, reach, -(b+de), -(b-de));
      ctx.closePath(); ctx.fill();
      // center bearing line
      ctx.strokeStyle = col+"aa"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ox,oy);
      ctx.lineTo(ox + reach*Math.cos(b), oy - reach*Math.sin(b));
      ctx.stroke();
      ctx.restore();
    } else if (fr.result === "near") {
      ctx.save();
      ctx.fillStyle = "#f8514966";
      ctx.beginPath(); ctx.arc(ox,oy,7,0,2*Math.PI); ctx.fill();
      ctx.restore();
    } else {
      // no_signal: a small hollow tick so the viewer sees the probe happened
      ctx.save();
      ctx.strokeStyle = "#484f58"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(ox,oy,3,0,2*Math.PI); ctx.stroke();
      ctx.restore();
    }
  }
}

function drawRoute(run, T, upto, ghost) {
  ctx.save();
  ctx.strokeStyle = ghost ? "#f0883e33" : "#f0883e";
  ctx.lineWidth = ghost ? 1 : 2;
  ctx.beginPath();
  ctx.moveTo(T.x(0), T.y(0));   // starts at origin
  for (let i = 0; i < upto && i < run.frames.length; i++) {
    const p = run.frames[i].pose;
    ctx.lineTo(T.x(p[0]), T.y(p[1]));
  }
  ctx.stroke();
  if (!ghost) {
    // clear-attempt markers along the way
    for (let i = 0; i < upto && i < run.frames.length; i++) {
      const fr = run.frames[i];
      if (fr.type !== "clear") continue;
      const px = T.x(fr.pose[0]), py = T.y(fr.pose[1]);
      ctx.fillStyle = fr.clear_success ? "#3fb95000" : "#f8514933";
      ctx.strokeStyle = fr.clear_success ? "#3fb950" : "#f85149";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px,py,9,0,2*Math.PI); ctx.stroke();
    }
    // robot head at the last revealed pose
    let hx = 0, hy = 0;
    if (upto > 0) {
      const p = run.frames[Math.min(upto, run.frames.length)-1].pose;
      hx = p[0]; hy = p[1];
    }
    ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(T.x(hx), T.y(hy), 5, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = "#f0883e"; ctx.lineWidth = 2; ctx.stroke();
  }
  ctx.restore();
}

function render() {
  const run = DATA.runs[cur];
  R = run.arena_radius || 1800;
  const T = tf();
  ctx.clearRect(0,0,cv.width,cv.height);
  drawArena(T);
  if (ghostChk.checked) {
    DATA.runs.forEach((r, i) => { if (i !== cur) drawRoute(r, T, r.frames.length, true); });
  }
  const cleared = clearedByFrame(run, frame);
  drawTruth(run, T, cleared);
  drawFans(run, T, frame);
  drawRoute(run, T, frame, false);
  updateFrameInfo(run);
}

function updateFrameInfo(run) {
  const n = run.frames.length;
  let txt = `帧 ${frame}/${n}`;
  if (frame > 0 && frame <= n) {
    const fr = run.frames[frame-1];
    let d = `${fr.type} CH${fr.channel} → ${fr.result}`;
    if (fr.bearing != null) d += ` @ ${fr.bearing}°`;
    txt += ` · ${d} · t=${fr.vt}s`;
  }
  frameInfo.textContent = txt;
}

function fillStats(run) {
  const s = run.summary;
  const rows = [
    ["清除 / 总数", `${s.sources_cleared} / ${s.sources_total}`],
    ["清除比例", (s.clear_ratio*100).toFixed(1)+"%"],
    ["定位清除总时间 (虚拟)", s.virtual_time_s+" s"],
    ["平均定位清除时间", s.avg_clear_time_s==null?"—":s.avg_clear_time_s+" s"],
    ["行进距离", s.route_distance_m+" m"],
    ["检测次数", s.num_scans],
    ["清除尝试 (失败)", `${s.num_clears} (${s.failed_clears})`],
    ["切换频道次数", s.num_switches],
    ["动作步数", s.steps],
  ];
  statsEl.innerHTML = rows.map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

function selectRun(i) {
  cur = i;
  const run = DATA.runs[cur];
  slider.max = run.frames.length;
  frame = run.frames.length;      // show the whole run by default
  slider.value = frame;
  fillStats(run);
  render();
}

function stopPlay() { if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "▶ 播放"; } }

playBtn.onclick = () => {
  if (timer) { stopPlay(); return; }
  const run = DATA.runs[cur];
  if (frame >= run.frames.length) { frame = 0; }
  playBtn.textContent = "⏸ 暂停";
  timer = setInterval(() => {
    frame++;
    slider.value = frame;
    render();
    if (frame >= DATA.runs[cur].frames.length) stopPlay();
  }, 180);
};
slider.oninput = () => { stopPlay(); frame = +slider.value; render(); };
algoSel.onchange = () => { stopPlay(); selectRun(+algoSel.value); };
ghostChk.onchange = render;

// init
DATA.runs.forEach((r, i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = `${r.label} — 清除 ${r.summary.sources_cleared}/${r.summary.sources_total}, t=${r.summary.virtual_time_s}s`;
  algoSel.appendChild(o);
});
if (DATA.runs.length) selectRun(0);
</script>
</body>
</html>
"""


def render_html(records: Iterable[RunRecord], *, title: str = "干扰源定位清除 · 运行可视化") -> str:
    """Serialize records into a standalone HTML document (as a string)."""
    runs = [r.to_dict() for r in records]
    meta = runs[0]["meta"] if runs else {}
    payload = {"runs": runs, "meta": meta}
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # guard against a literal </script> inside data closing the tag early
    blob = blob.replace("</", "<\\/")
    return (
        _TEMPLATE
        .replace("__TITLE__", title)
        .replace("__DATA__", blob)
    )


def write_report(
    records: Iterable[RunRecord],
    path: str | Path,
    *,
    title: str = "干扰源定位清除 · 运行可视化",
) -> Path:
    """Render and write the HTML report; returns the output path."""
    records = list(records)
    html = render_html(records, title=title)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
