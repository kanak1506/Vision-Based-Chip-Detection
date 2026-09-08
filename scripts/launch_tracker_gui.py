# -*- coding: utf-8 -*-
"""
=============================================================
  5-Waypoint (0%, 25%, 50%, 75%, 100%) Tool Calibration GUI
  Vision-Based Chip Detection Project
=============================================================
  USAGE:
    python scripts/launch_tracker_gui.py

  Opens a browser tab at http://localhost:5050
  - Dedicated 5-Waypoint Anchor Buttons: 0%, 25%, 50%, 75%, 100%
  - Anchor pills with instant [✕] delete buttons
  - Set / Delete / Clear anchors easily
  - Saves directly to data/metadata/tracker_config.json
=============================================================
"""

import sys
import io
import os
import json
import base64
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import RAW_VIDEOS_DIR, TRACKER_CONFIG_PATH

PORT = 5050

all_files = sorted([
    f for f in os.listdir(RAW_VIDEOS_DIR)
    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
])


def load_config():
    if os.path.exists(TRACKER_CONFIG_PATH):
        with open(TRACKER_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_config(cfg):
    os.makedirs(os.path.dirname(TRACKER_CONFIG_PATH), exist_ok=True)
    with open(TRACKER_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_frame(fname, frame_idx=None):
    fpath = os.path.join(RAW_VIDEOS_DIR, fname)
    cap = cv2.VideoCapture(fpath)
    if not cap.isOpened():
        return None, 0, 0, 0, 0, 0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = round(cap.get(cv2.CAP_PROP_FPS), 2)
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_idx is None:
            frame_idx = 0
        frame_idx = max(0, min(frame_idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret or frame is None:
        return None, total, fps, w, h, frame_idx
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("utf-8")
    return b64, total, fps, w, h, frame_idx


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>5-Waypoint Tool Tip Calibration</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; display: flex; height: 100vh; overflow: hidden; }
  #sidebar { width: 330px; background: #1a1d27; border-right: 1px solid #2d3142; display: flex; flex-direction: column; flex-shrink: 0; }
  #sidebar-header { padding: 16px 18px; border-bottom: 1px solid #2d3142; }
  #sidebar-header h1 { font-size: 14px; font-weight: 700; color: #60a5fa; letter-spacing: 0.5px; }
  #sidebar-header p { font-size: 11px; color: #64748b; margin-top: 4px; }
  #progress-bar-wrap { margin-top: 10px; background: #2d3142; border-radius: 99px; height: 6px; overflow: hidden; }
  #progress-bar { background: linear-gradient(90deg, #3b82f6, #10b981); height: 100%; width: 0%; transition: width 0.3s; }
  #vid-list { overflow-y: auto; flex: 1; padding: 8px; }
  .vid-item { padding: 9px 11px; border-radius: 8px; cursor: pointer; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between; font-size: 11px; transition: background 0.15s; }
  .vid-item:hover { background: #242838; }
  .vid-item.active { background: #1e3a5f; border-left: 3px solid #3b82f6; }
  .vid-item .vname { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 8px; }
  .badge { font-size: 10px; padding: 2px 7px; border-radius: 99px; font-weight: 600; flex-shrink: 0; }
  .badge-done { background: #064e3b; color: #34d399; }
  .badge-skip { background: #451a03; color: #fb923c; }
  .badge-todo { background: #1e293b; color: #64748b; }
  
  #main { flex: 1; display: flex; flex-direction: column; background: #0f1117; overflow: hidden; }
  #topbar { padding: 10px 20px; background: #1a1d27; border-bottom: 1px solid #2d3142; display: flex; align-items: center; justify-content: space-between; font-size: 12px; }
  #topbar .title { font-weight: 600; color: #f1f5f9; font-size: 13px; }
  #topbar .meta { color: #64748b; font-size: 11px; margin-left: 12px; }
  
  #waypoint-bar { background: #11141e; padding: 8px 20px; border-bottom: 1px solid #2d3142; display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-shrink: 0; }
  .wp-group { display: flex; align-items: center; gap: 8px; }
  .wp-btn { padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; background: #1e293b; border: 1px solid #3b82f6; color: #93c5fd; transition: all 0.15s; }
  .wp-btn:hover { background: #2563eb; color: #fff; }
  .wp-btn.active { background: #2563eb; color: #fff; border-color: #60a5fa; box-shadow: 0 0 8px rgba(59,130,246,0.5); }
  .wp-btn.tagged { border-color: #10b981; color: #34d399; }
  .wp-btn.tagged::after { content: " ✓"; font-weight: bold; }
  
  #anchor-pills-bar { background: #141824; padding: 6px 20px; border-bottom: 1px solid #2d3142; display: flex; align-items: center; gap: 8px; flex-shrink: 0; min-height: 40px; overflow-x: auto; }
  .anchor-pill { background: #1e293b; border: 1px solid #3b82f6; border-radius: 5px; padding: 3px 8px; font-size: 11px; color: #60a5fa; display: flex; align-items: center; gap: 6px; cursor: pointer; white-space: nowrap; }
  .anchor-pill.active { background: #2563eb; color: #fff; font-weight: bold; }
  .anchor-pill .del-btn { color: #f87171; font-weight: bold; padding: 0 4px; border-radius: 3px; cursor: pointer; }
  .anchor-pill .del-btn:hover { background: #991b1b; color: #fff; }
  
  #canvas-container { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; background: #0a0c10; user-select: none; }
  #canvas-wrap { position: relative; display: inline-block; cursor: crosshair; }
  #main-img { display: block; max-height: calc(100vh - 270px); max-width: calc(100vw - 370px); object-fit: contain; }
  #overlay-canvas { position: absolute; top: 0; left: 0; pointer-events: all; }
  
  #guide-box { position: absolute; top: 12px; right: 12px; background: rgba(26,29,39,0.9); border: 1px solid #2d3142; border-radius: 8px; padding: 12px 16px; font-size: 11px; color: #94a3b8; line-height: 1.6; pointer-events: none; backdrop-filter: blur(4px); }
  #guide-box b { color: #60a5fa; }
  
  #controls { padding: 10px 20px; background: #1a1d27; border-top: 1px solid #2d3142; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-shrink: 0; }
  .btn-group { display: flex; align-items: center; gap: 8px; }
  button { padding: 7px 13px; border-radius: 6px; border: none; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s; display: flex; align-items: center; gap: 6px; }
  .btn-primary { background: #2563eb; color: #fff; }
  .btn-primary:hover { background: #1d4ed8; }
  .btn-success { background: #059669; color: #fff; }
  .btn-success:hover { background: #047857; }
  .btn-secondary { background: #2d3142; color: #cbd5e1; }
  .btn-secondary:hover { background: #3d4358; }
  .btn-danger { background: #7f1d1d; color: #fca5a5; }
  .btn-danger:hover { background: #991b1b; }
  .btn-warning { background: #b45309; color: #fef3c7; }
  .btn-warning:hover { background: #92400e; }
  
  .frame-nav { display: flex; align-items: center; gap: 6px; color: #94a3b8; font-size: 11px; }
  .frame-nav input[type="number"] { width: 65px; background: #0f1117; border: 1px solid #2d3142; color: #f1f5f9; padding: 4px 6px; border-radius: 5px; text-align: center; font-size: 12px; }
  .frame-nav input[type="range"] { width: 120px; cursor: pointer; }
  #info-strip { font-size: 11px; color: #64748b; display: flex; gap: 14px; }
  #info-strip span { color: #94a3b8; font-weight: 600; }
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">
    <h1>5-WAYPOINT CALIBRATION</h1>
    <p id="progress-text">0 / 0 calibrated</p>
    <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
  </div>
  <div id="vid-list"></div>
</div>

<div id="main">
  <div id="topbar">
    <div><span class="title" id="top-vname">Select a video</span><span class="meta" id="top-meta"></span></div>
    <div id="info-strip">
      <div>Tip: <span id="info-tip">none</span></div>
      <div>Box: <span id="info-box">none</span></div>
      <div>Frame: <span id="info-frame">0 / 0</span></div>
    </div>
  </div>

  <div id="waypoint-bar">
    <div class="wp-group">
      <span style="font-size:12px; font-weight:600; color:#94a3b8; margin-right:4px;">5-Point Anchors:</span>
      <button class="wp-btn" id="btn-p0" onclick="jumpPct(0.00)">0% (Start)</button>
      <button class="wp-btn" id="btn-p25" onclick="jumpPct(0.25)">25%</button>
      <button class="wp-btn" id="btn-p50" onclick="jumpPct(0.50)">50% (Mid)</button>
      <button class="wp-btn" id="btn-p75" onclick="jumpPct(0.75)">75%</button>
      <button class="wp-btn" id="btn-p100" onclick="jumpPct(1.00)">100% (End)</button>
    </div>
    <div class="btn-group">
      <button class="btn-warning" onclick="setAnchorAtCurrentFrame()">&#9875; Mark / Save Anchor (K)</button>
      <button class="btn-danger" onclick="deleteCurrentAnchor()">&#10005; Delete Anchor Here</button>
    </div>
  </div>

  <div id="anchor-pills-bar">
    <span style="font-size:11px; font-weight:600; color:#64748b;">Active Anchors:</span>
    <div id="anchor-pills-list" style="display:flex; gap:6px; align-items:center;"></div>
  </div>

  <div id="canvas-container">
    <div id="canvas-wrap">
      <img id="main-img" alt="Frame preview"/>
      <canvas id="overlay-canvas"></canvas>
    </div>
    <div id="guide-box">
      <b>Calibration Controls:</b><br/>
      1. Click <b>[0%]</b> to <b>[100%]</b> buttons to navigate.<br/>
      2. <b>Left Click</b> on tool tip ➔ Press <b>K</b> to save anchor.<br/>
      3. Click <b>"Delete Anchor Here"</b> or <b>[✕]</b> on any pill to remove unwanted anchors.<br/>
      4. Click <b>"Save Video Config &rarr;"</b> (Space/Enter) when done.
    </div>
  </div>

  <div id="controls">
    <div class="btn-group">
      <button class="btn-secondary" onclick="stepFrame(-10)">&lsaquo; -10</button>
      <button class="btn-secondary" onclick="stepFrame(-1)">-1</button>
      <div class="frame-nav">
        <input type="number" id="frame-input" onchange="jumpToFrame(this.value)"/>
        <input type="range" id="frame-slider" min="0" max="100" value="0" oninput="jumpToFrame(this.value)"/>
      </div>
      <button class="btn-secondary" onclick="stepFrame(1)">+1</button>
      <button class="btn-secondary" onclick="stepFrame(10)">+10 &rsaquo;</button>
    </div>
    <div class="btn-group">
      <button class="btn-success" onclick="saveCurrentVideo()">Save Video Config &rarr;</button>
    </div>
  </div>
</div>

<script>
let files = [];
let config = {};
let curIdx = 0;
let curFrame = 0;
let totalFrames = 0;
let imgNaturalW = 1;
let imgNaturalH = 1;

let tipPoint = null;
let box = null;
let isDrawing = false;
let startX = 0, startY = 0;
let keyframes = [];

const img = document.getElementById('main-img');
const canvas = document.getElementById('overlay-canvas');
const ctx = canvas.getContext('2d');

async function init() {
  const [fRes, cRes] = await Promise.all([fetch('/api/files'), fetch('/api/config')]);
  files = await fRes.json();
  config = await cRes.json();
  renderSidebar();
  if (files.length > 0) loadVideo(0);
}

function renderSidebar() {
  const list = document.getElementById('vid-list');
  list.innerHTML = '';
  let done = 0;
  files.forEach((f, idx) => {
    const item = document.createElement('div');
    item.className = 'vid-item' + (idx === curIdx ? ' active' : '');
    const entry = config[f];
    let badge = '<span class="badge badge-todo">TODO</span>';
    const kfs = (entry && entry.keyframes && entry.keyframes.length > 0) ? entry.keyframes : (entry && entry.bbox ? [entry] : []);
    if (kfs.length >= 3) {
      badge = `<span class="badge badge-done">${kfs.length} ANCHORS</span>`;
      done++;
    } else if (kfs.length > 0) {
      badge = `<span class="badge badge-done" style="background:#1e3a8a;">${kfs.length} ANCHOR</span>`;
      done++;
    }
    item.innerHTML = `<span class="vname" title="${f}">${f}</span>${badge}`;
    item.onclick = () => loadVideo(idx);
    list.appendChild(item);
  });
  const pct = Math.round((done / (files.length || 1)) * 100);
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = `${done} / ${files.length} calibrated (${pct}%)`;
}

function renderAnchorPills() {
  const list = document.getElementById('anchor-pills-list');
  list.innerHTML = '';
  if (!keyframes || keyframes.length === 0) {
    list.innerHTML = '<span style="color:#64748b; font-size:11px;">No anchors yet. Mark frame 0!</span>';
    return;
  }
  keyframes.sort((a, b) => a.frame_idx - b.frame_idx);
  keyframes.forEach((kf, idx) => {
    const pill = document.createElement('div');
    const isAct = (kf.frame_idx === curFrame);
    pill.className = 'anchor-pill' + (isAct ? ' active' : '');
    pill.innerHTML = `<span>Frame ${kf.frame_idx} (${kf.tool_tip ? kf.tool_tip.join(',') : 'N/A'})</span><span class="del-btn" onclick="event.stopPropagation(); deleteAnchorAtIdx(${idx});" title="Delete this anchor">&times;</span>`;
    pill.onclick = () => fetchFrame(files[curIdx], kf.frame_idx);
    list.appendChild(pill);
  });
}

function updateWaypointButtons() {
  const pcts = [0.00, 0.25, 0.50, 0.75, 1.00];
  const ids  = ['btn-p0', 'btn-p25', 'btn-p50', 'btn-p75', 'btn-p100'];
  
  pcts.forEach((p, idx) => {
    const btn = document.getElementById(ids[idx]);
    const targetF = Math.round(p * (totalFrames - 1));
    const isAct = Math.abs(curFrame - targetF) <= Math.max(1, totalFrames * 0.03);
    btn.className = 'wp-btn' + (isAct ? ' active' : '');
    
    const exists = keyframes.some(k => Math.abs(k.frame_idx - targetF) <= Math.max(2, totalFrames * 0.03));
    if (exists) {
      btn.classList.add('tagged');
    }
  });
  renderAnchorPills();
}

function jumpPct(pct) {
  if (totalFrames <= 0) return;
  const f = Math.round(pct * (totalFrames - 1));
  fetchFrame(files[curIdx], f);
}

async function loadVideo(idx) {
  curIdx = idx;
  const fname = files[idx];
  const entry = config[fname] || {};
  
  if (entry.keyframes && entry.keyframes.length > 0) {
    keyframes = JSON.parse(JSON.stringify(entry.keyframes));
  } else if (entry.bbox) {
    const tip = entry.tool_tip || [entry.bbox[0], entry.bbox[1]];
    keyframes = [{
      frame_idx: entry.start_frame || 0,
      tool_tip: tip,
      bbox: entry.bbox
    }];
  } else {
    keyframes = [];
  }
  
  curFrame = 0;
  loadCurrentFrameState();
  await fetchFrame(fname, curFrame);
  renderSidebar();
}

function loadCurrentFrameState() {
  const existing = keyframes.find(k => k.frame_idx === curFrame);
  if (existing) {
    tipPoint = existing.tool_tip ? [...existing.tool_tip] : null;
    box = existing.bbox ? [...existing.bbox] : null;
  } else {
    const interp = getInterpolatedState(curFrame);
    if (interp) {
      tipPoint = [Math.round(interp.tip_x), Math.round(interp.tip_y)];
      box = [Math.round(interp.bbox_x), Math.round(interp.bbox_y), interp.bbox_w, interp.bbox_h];
    } else {
      tipPoint = null;
      box = null;
    }
  }
}

function getInterpolatedState(f) {
  if (!keyframes || keyframes.length === 0) return null;
  if (keyframes.length === 1 || f <= keyframes[0].frame_idx) {
    const k0 = keyframes[0];
    return { tip_x: k0.tool_tip[0], tip_y: k0.tool_tip[1], bbox_x: k0.bbox[0], bbox_y: k0.bbox[1], bbox_w: k0.bbox[2], bbox_h: k0.bbox[3] };
  }
  if (f >= keyframes[keyframes.length - 1].frame_idx) {
    const kLast = keyframes[keyframes.length - 1];
    return { tip_x: kLast.tool_tip[0], tip_y: kLast.tool_tip[1], bbox_x: kLast.bbox[0], bbox_y: kLast.bbox[1], bbox_w: kLast.bbox[2], bbox_h: kLast.bbox[3] };
  }
  for (let i = 0; i < keyframes.length - 1; i++) {
    const k0 = keyframes[i];
    const k1 = keyframes[i + 1];
    if (k0.frame_idx <= f && f <= k1.frame_idx) {
      const alpha = (f - k0.frame_idx) / (k1.frame_idx - k0.frame_idx);
      return {
        tip_x: (1 - alpha) * k0.tool_tip[0] + alpha * k1.tool_tip[0],
        tip_y: (1 - alpha) * k0.tool_tip[1] + alpha * k1.tool_tip[1],
        bbox_x: (1 - alpha) * k0.bbox[0] + alpha * k1.bbox[0],
        bbox_y: (1 - alpha) * k0.bbox[1] + alpha * k1.bbox[1],
        bbox_w: k0.bbox[2],
        bbox_h: k0.bbox[3]
      };
    }
  }
  return null;
}

async function fetchFrame(fname, frameIdx) {
  const res = await fetch(`/api/frame?file=${encodeURIComponent(fname)}&frame=${frameIdx}`);
  const data = await res.json();
  totalFrames = data.total_frames;
  curFrame = data.frame_idx;
  document.getElementById('top-vname').textContent = fname;
  document.getElementById('top-meta').textContent = `${data.w}x${data.h} @ ${data.fps}fps (${totalFrames} frames)`;
  document.getElementById('info-frame').textContent = `${curFrame} / ${totalFrames}`;
  document.getElementById('frame-input').value = curFrame;
  const slider = document.getElementById('frame-slider');
  slider.max = totalFrames - 1;
  slider.value = curFrame;
  imgNaturalW = data.w;
  imgNaturalH = data.h;
  loadCurrentFrameState();
  img.onload = () => { syncCanvas(); redraw(); };
  img.src = 'data:image/jpeg;base64,' + data.img_b64;
  updateWaypointButtons();
}

function syncCanvas() {
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + 'px';
  canvas.style.height = img.clientHeight + 'px';
}

function imgToCanvas(x, y) {
  const sx = canvas.width / imgNaturalW;
  const sy = canvas.height / imgNaturalH;
  return [x * sx, y * sy];
}

function canvasToImg(x, y) {
  const sx = imgNaturalW / canvas.width;
  const sy = imgNaturalH / canvas.height;
  return [Math.round(x * sx), Math.round(y * sy)];
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (box) {
    const [bx, by, bw, bh] = box;
    const [cx, cy] = imgToCanvas(bx, by);
    const [cw, ch] = imgToCanvas(bw, bh);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.strokeRect(cx, cy, cw, ch);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
    ctx.fillRect(cx, cy, cw, ch);
    document.getElementById('info-box').textContent = `[${bx}, ${by}, ${bw}, ${bh}]`;
  } else {
    document.getElementById('info-box').textContent = 'none';
  }
  
  if (tipPoint) {
    const [tx, ty] = tipPoint;
    const [cx, cy] = imgToCanvas(tx, ty);
    
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#2563eb';
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(cx, cy, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    
    document.getElementById('info-tip').textContent = `(${tx}, ${ty})`;
  } else {
    document.getElementById('info-tip').textContent = 'none';
  }
}

canvas.addEventListener('mousedown', (e) => {
  const r = canvas.getBoundingClientRect();
  const [ix, iy] = canvasToImg(e.clientX - r.left, e.clientY - r.top);
  startX = ix; startY = iy;
  tipPoint = [ix, iy];
  isDrawing = true;
  redraw();
});

canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const r = canvas.getBoundingClientRect();
  const [ix, iy] = canvasToImg(e.clientX - r.left, e.clientY - r.top);
  const w = Math.abs(ix - startX);
  const h = Math.abs(iy - startY);
  if (w > 3 || h > 3) {
    box = [Math.min(startX, ix), Math.min(startY, iy), w, h];
  }
  redraw();
});

canvas.addEventListener('mouseup', () => {
  isDrawing = false;
  if (!box && tipPoint) {
    box = [tipPoint[0] - 8, tipPoint[1] - 8, 16, 16];
  }
  redraw();
});

function setAnchorAtCurrentFrame() {
  if (!tipPoint) {
    alert('Please click on the cutting tool tip first!');
    return;
  }
  if (!box) {
    box = [tipPoint[0] - 8, tipPoint[1] - 8, 16, 16];
  }
  const existingIdx = keyframes.findIndex(k => k.frame_idx === curFrame);
  const newAnchor = {
    frame_idx: curFrame,
    tool_tip: [...tipPoint],
    bbox: [...box]
  };
  if (existingIdx >= 0) {
    keyframes[existingIdx] = newAnchor;
  } else {
    keyframes.push(newAnchor);
  }
  keyframes.sort((a, b) => a.frame_idx - b.frame_idx);
  updateWaypointButtons();
  renderSidebar();
}

function deleteCurrentAnchor() {
  // Find nearest anchor within +/- 10 frames of curFrame, or exact match
  const idx = keyframes.findIndex(k => Math.abs(k.frame_idx - curFrame) <= Math.max(5, totalFrames * 0.03));
  if (idx >= 0) {
    keyframes.splice(idx, 1);
    tipPoint = null;
    box = null;
    redraw();
    updateWaypointButtons();
    renderSidebar();
  } else if (keyframes.length > 0) {
    // If not near, delete the last keyframe if at end
    if (curFrame >= keyframes[keyframes.length - 1].frame_idx) {
      keyframes.pop();
      tipPoint = null;
      box = null;
      redraw();
      updateWaypointButtons();
      renderSidebar();
    }
  }
}

function deleteAnchorAtIdx(idx) {
  keyframes.splice(idx, 1);
  loadCurrentFrameState();
  redraw();
  updateWaypointButtons();
  renderSidebar();
}

async function saveCurrentVideo() {
  if (!keyframes || keyframes.length === 0) {
    alert('Please add at least one keyframe anchor before saving!');
    return;
  }
  const fname = files[curIdx];
  const oldEntry = config[fname] || {};
  keyframes.sort((a, b) => a.frame_idx - b.frame_idx);
  
  const firstKf = keyframes[0];
  config[fname] = {
    video_index: oldEntry.video_index || (curIdx + 1),
    material: oldEntry.material || "Unknown",
    start_frame: firstKf.frame_idx,
    bbox: firstKf.bbox,
    tool_tip: firstKf.tool_tip,
    offset_from_bbox: [firstKf.tool_tip[0] - firstKf.bbox[0], firstKf.tool_tip[1] - firstKf.bbox[1]],
    keyframes: keyframes
  };
  
  await fetch('/api/save', { method: 'POST', body: JSON.stringify(config) });
  renderSidebar();
  if (curIdx < files.length - 1) {
    loadVideo(curIdx + 1);
  } else {
    alert('All video configurations saved successfully!');
  }
}

function stepFrame(delta) {
  const next = curFrame + delta;
  if (next >= 0 && next < totalFrames) fetchFrame(files[curIdx], next);
}
function jumpToFrame(v) { fetchFrame(files[curIdx], parseInt(v)); }

window.addEventListener('keydown', (e) => {
  if (['input', 'textarea'].includes(e.target.tagName.toLowerCase())) return;
  if (e.key === 'k' || e.key === 'K') { setAnchorAtCurrentFrame(); }
  else if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); saveCurrentVideo(); }
  else if (e.key === 'ArrowLeft') stepFrame(e.shiftKey ? -10 : -1);
  else if (e.key === 'ArrowRight') stepFrame(e.shiftKey ? 10 : 1);
});

window.addEventListener('resize', () => { syncCanvas(); redraw(); });
init();
</script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        pr = urlparse(self.path)
        qs = parse_qs(pr.query)

        if pr.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif pr.path == "/api/files":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(all_files).encode("utf-8"))

        elif pr.path == "/api/config":
            cfg = load_config()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(cfg).encode("utf-8"))

        elif pr.path == "/api/frame":
            fname = qs.get("file", [""])[0]
            f_idx = int(qs.get("frame", [0])[0])
            b64, total, fps, w, h, actual_idx = get_frame(fname, f_idx)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "img_b64": b64 or "",
                "total_frames": total,
                "fps": fps,
                "w": w,
                "h": h,
                "frame_idx": actual_idx
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        pr = urlparse(self.path)
        if pr.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            cfg = json.loads(body.decode("utf-8"))
            save_config(cfg)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def launch_server():
    server = HTTPServer(("127.0.0.1", PORT), RequestHandler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n=======================================================")
    print(f"  5-WAYPOINT CALIBRATION SERVER RUNNING")
    print(f"  URL: {url}")
    print(f"  Press Ctrl+C to stop.")
    print(f"=======================================================\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down calibration server.")
        server.server_close()


if __name__ == "__main__":
    launch_server()
