"""Interactive well control drilling simulator & teaching console web server.

Uses Python's standard library http.server (zero third-party dependencies required):
- SVG analog and digital gauges (SPP, Casing Pressure, Pit Volume, Pump SPM, Flow Paddle)
- Real-time HTML5 Canvas strip charts (rolling history of SPP, CICP, and Pit Gain)
- Remote hydraulic choke panel with jog buttons and automated PID Auto-Choke mode
- Web Audio API real-time acoustic alarms (synthesized pit gain alert & pump rumble)
- Rig controls (pump throttle, tool joint space-out, annular BOP rams, HCR valve)
- Live IADC WellSharp lessons with step validation and kill sheet automated grading
- Markdown Kill Sheet report export endpoint (/api/killsheet/export)
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from wellsim.core.transient import TransientSimulator, UserControls
from wellsim.core.well_control import KillSheetInputs, calculate_kill_sheet
from wellsim.teaching.scenarios import (
    LessonID,
    LessonScenario,
    get_all_lessons,
    get_lesson,
    grade_kill_sheet_submission,
)

logger = logging.getLogger(__name__)


# Global active simulator and active lesson session
class SimulatorSession:
    def __init__(self) -> None:
        self.sim = TransientSimulator()
        self.current_lesson: LessonScenario | None = get_lesson(LessonID.KICK_DETECTION)
        if self.current_lesson:
            self.sim = self.current_lesson.initial_simulator_setup()

    def set_lesson(self, lesson_id: str) -> None:
        les = get_lesson(lesson_id)
        if les:
            self.current_lesson = get_lesson(lesson_id)
            self.sim = self.current_lesson.initial_simulator_setup()

    def step(self, dt: float = 1.0) -> dict[str, Any]:
        st = self.sim.step(dt=dt)
        if self.current_lesson:
            for idx in range(len(self.current_lesson.steps)):
                self.current_lesson.evaluate_step(idx, st)
        data = st.as_dict()
        if self.current_lesson:
            data["lesson"] = {
                "id": self.current_lesson.lesson_id.value,
                "title": self.current_lesson.title,
                "progress_pct": round(self.current_lesson.progress_pct, 1),
                "steps": [
                    {
                        "step_number": s.step_number,
                        "title": s.title,
                        "instruction": s.instruction,
                        "hint": s.hint,
                        "passed": s.passed,
                    }
                    for s in self.current_lesson.steps
                ],
            }
        return data


SESSION = SimulatorSession()


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SLB Drillbench v2026.1 // Dynamic Well Control & Blowout Simulator</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg-app: #090d13;
      --bg-ribbon: #111722;
      --bg-panel: #141c28;
      --bg-card: #182232;
      --bg-surface: #0d131d;
      --border-dim: #1f2b3e;
      --border-bright: #2d3e58;
      --slb-blue: #0077c8;
      --slb-cyan: #00c0f0;
      --slb-glow: rgba(0, 192, 240, 0.25);
      --color-spp: #00c0f0;
      --color-casing: #ff9800;
      --color-pit: #10b981;
      --color-flow: #f43f5e;
      --color-choke: #ffd54f;
      --color-target: #f59e0b;
      --color-danger: #ef4444;
      --color-good: #10b981;
      --color-warn: #f59e0b;
      --text-main: #f1f5f9;
      --text-dim: #94a3b8;
      --text-muted: #64748b;
      --font-mono: "JetBrains Mono", "SF Mono", "Roboto Mono", Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      height: 100vh;
      max-height: 100vh;
      overflow: hidden;
      background-color: var(--bg-app);
      color: var(--text-main);
      font-family: var(--font-sans);
      display: flex;
      flex-direction: column;
      user-select: none;
    }

    /* TOP COMMAND RIBBON */
    #ribbon {
      height: 48px;
      min-height: 48px;
      background: var(--bg-ribbon);
      border-bottom: 1px solid var(--border-dim);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      gap: 12px;
      z-index: 100;
    }
    .brand-cluster {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .slb-badge {
      background: var(--slb-blue);
      color: #fff;
      font-weight: 900;
      font-size: 11px;
      padding: 3px 7px;
      border-radius: 3px;
      letter-spacing: 0.5px;
    }
    .brand-title {
      font-weight: 800;
      font-size: 13px;
      color: #fff;
      letter-spacing: 1.2px;
    }
    .brand-subtitle {
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .case-badge {
      background: #0f1622;
      border: 1px solid var(--border-dim);
      color: var(--slb-cyan);
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 3px;
    }

    /* WORKSPACE TABS */
    .tabs-deck {
      display: flex;
      gap: 3px;
      height: 100%;
      align-items: flex-end;
    }
    .tab-btn {
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      color: var(--text-dim);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.6px;
      padding: 12px 14px 10px 14px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
      text-transform: uppercase;
    }
    .tab-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.03);
    }
    .tab-btn.active {
      color: var(--slb-cyan);
      border-bottom-color: var(--slb-cyan);
      background: rgba(0, 192, 240, 0.06);
    }

    /* SIMULATION CONTROL DECK */
    .control-deck {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      background: #172433;
      border: 1px solid #23374d;
      color: #7dd3fc;
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--color-good);
      box-shadow: 0 0 8px var(--color-good);
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }

    .clock-display {
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 700;
      color: #e2e8f0;
      background: #090e16;
      border: 1px solid var(--border-dim);
      padding: 4px 8px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }
    .speed-btn-group {
      display: flex;
      border: 1px solid var(--border-dim);
      border-radius: 4px;
      overflow: hidden;
    }
    .speed-btn {
      background: #111823;
      border: none;
      color: var(--text-dim);
      font-size: 10px;
      font-weight: 700;
      padding: 5px 8px;
      cursor: pointer;
    }
    .speed-btn.active {
      background: var(--slb-blue);
      color: #fff;
    }
    .ribbon-btn {
      background: #192332;
      border: 1px solid var(--border-bright);
      color: #e2e8f0;
      border-radius: 4px;
      padding: 5px 10px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: background 0.15s;
    }
    .ribbon-btn:hover { background: #223044; }
    .ribbon-btn.btn-run-good { background: #065f46; border-color: #059669; color: #a7f3d0; }
    .ribbon-btn.btn-run-good:hover { background: #047857; }
    .ribbon-btn.btn-run-paused { background: #881337; border-color: #be123c; color: #fecdd3; }
    .ribbon-btn.btn-run-paused:hover { background: #9f1239; }

    /* WORKSPACE SCREENS CONTAINER */
    #workspace-container {
      flex: 1;
      height: calc(100vh - 48px);
      overflow: hidden;
      position: relative;
    }
    .workspace-screen {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      display: none;
      height: 100%;
      overflow: hidden;
    }
    .workspace-screen.active {
      display: flex;
    }

    /* SCREEN 1: LANDSCAPE 4-COLUMN DRILLING & HYDRAULICS SUITE */
    #screen-hydraulics {
      display: none;
      grid-template-columns: 210px 240px 1fr 340px;
      gap: 8px;
      padding: 8px;
      height: 100%;
      box-sizing: border-box;
    }
    #screen-hydraulics.active {
      display: grid;
    }

    /* PANELS & CARDS */
    .cad-panel {
      background: var(--bg-panel);
      border: 1px solid var(--border-dim);
      border-radius: 5px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .cad-header {
      height: 28px;
      min-height: 28px;
      background: #101621;
      border-bottom: 1px solid var(--border-dim);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px;
      font-size: 10.5px;
      font-weight: 800;
      letter-spacing: 0.8px;
      color: #94a3b8;
      text-transform: uppercase;
    }
    .cad-badge {
      font-family: var(--font-mono);
      font-size: 9.5px;
      padding: 1px 6px;
      border-radius: 3px;
      background: #172433;
      color: var(--slb-cyan);
      border: 1px solid #23374d;
    }
    .cad-body {
      flex: 1;
      position: relative;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* COLUMN 1: WELLBORE CAD CANVAS */
    #wellboreCanvas {
      width: 100%;
      height: 100%;
      background: #090e15;
      display: block;
    }
    .fluid-legend {
      height: 26px;
      background: #0e141f;
      border-top: 1px solid var(--border-dim);
      display: flex;
      align-items: center;
      justify-content: space-around;
      font-size: 9.5px;
      font-family: var(--font-mono);
      color: var(--text-dim);
    }
    .legend-item { display: flex; align-items: center; gap: 4px; }
    .legend-box { width: 8px; height: 8px; border-radius: 1px; }

    /* COLUMN 2: DRILLING WINDOW CANVAS */
    #windowCanvas {
      width: 100%;
      height: 100%;
      background: #090e15;
      display: block;
    }
    .window-metrics {
      height: 38px;
      background: #0e141f;
      border-top: 1px solid var(--border-dim);
      display: grid;
      grid-template-columns: 1fr 1fr;
      padding: 4px 8px;
      gap: 6px;
      font-family: var(--font-mono);
    }
    .wm-card {
      background: #131c28;
      border: 1px solid #1f2b3c;
      border-radius: 3px;
      padding: 2px 6px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .wm-label { font-size: 8px; color: var(--text-muted); text-transform: uppercase; }
    .wm-val { font-size: 11px; font-weight: 700; color: #38bdf8; }

    /* COLUMN 3: EDR STRIP CHART & LIVE READOUTS */
    .edr-chart-container {
      flex: 1;
      position: relative;
      background: #080c13;
      overflow: hidden;
    }
    #stripChart {
      width: 100%;
      height: 100%;
      display: block;
    }
    .chart-cursor-hud {
      position: absolute;
      top: 6px;
      right: 8px;
      background: rgba(14, 20, 31, 0.9);
      border: 1px solid var(--border-dim);
      border-radius: 3px;
      padding: 3px 8px;
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-dim);
      display: flex;
      gap: 12px;
      pointer-events: none;
    }

    /* HUD TELEMETRY STRIP (Bottom of Col 3) */
    .edr-telemetry-hud {
      height: 64px;
      min-height: 64px;
      background: #0e141e;
      border-top: 1px solid var(--border-dim);
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 6px;
      padding: 6px;
    }
    .hud-tile {
      background: #141c28;
      border: 1px solid var(--border-dim);
      border-radius: 4px;
      padding: 4px 8px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      font-family: var(--font-mono);
    }
    .hud-label {
      font-size: 9px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .hud-num {
      font-size: 18px;
      font-weight: 800;
      line-height: 1.1;
      font-variant-numeric: tabular-nums;
    }
    .hud-unit {
      font-size: 8.5px;
      color: var(--text-muted);
      align-self: flex-end;
    }

    /* COLUMN 4: RIG CONSOLE & SWACO CHOKE PANEL */
    .console-stack {
      display: flex;
      flex-direction: column;
      gap: 8px;
      height: 100%;
      overflow-y: auto;
      padding: 4px;
    }
    .dial-card {
      background: #101722;
      border: 1px solid var(--border-dim);
      border-radius: 4px;
      padding: 6px;
      display: flex;
      justify-content: space-around;
      align-items: center;
    }
    #dialsCanvas {
      width: 310px;
      height: 130px;
      display: block;
    }

    /* SWACO HYDRAULIC CHOKE CARD */
    .choke-box {
      background: #101722;
      border: 1px solid var(--border-dim);
      border-radius: 4px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .choke-row-main {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .choke-readout {
      font-family: var(--font-mono);
      font-size: 22px;
      font-weight: 900;
      color: var(--color-choke);
      letter-spacing: 0.5px;
      font-variant-numeric: tabular-nums;
    }
    .choke-subreadout {
      font-size: 10px;
      color: var(--text-dim);
      font-family: var(--font-mono);
      text-align: right;
    }
    .choke-bar-track {
      height: 7px;
      background: #090e15;
      border-radius: 3px;
      overflow: hidden;
      border: 1px solid #1f2b3e;
    }
    .choke-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #ffd54f, #f43f5e);
      width: 100%;
      transition: width 0.15s ease;
    }
    .jog-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 5px;
    }
    .jog-btn {
      background: #182232;
      border: 1px solid var(--border-bright);
      color: #e2e8f0;
      padding: 7px 0;
      font-size: 10.5px;
      font-weight: 800;
      font-family: var(--font-mono);
      border-radius: 3px;
      cursor: pointer;
      text-align: center;
      transition: background 0.1s;
    }
    .jog-btn:hover { background: #25354c; }
    .jog-btn:active { background: #0077c8; color: #fff; }

    /* AUTO-CHOKE ROW */
    .autochoke-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #0c121b;
      border: 1px solid var(--border-dim);
      border-radius: 3px;
      padding: 5px 8px;
      font-size: 10.5px;
    }
    .toggle-pill {
      background: #182433;
      border: 1px solid var(--border-bright);
      color: var(--text-dim);
      font-size: 10px;
      font-weight: 800;
      padding: 4px 8px;
      border-radius: 3px;
      cursor: pointer;
    }
    .toggle-pill.active {
      background: var(--slb-blue);
      border-color: #38bdf8;
      color: #fff;
    }

    /* SWITCHGEAR CONTROLS */
    .switch-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .switch-btn {
      background: #141d2a;
      border: 1px solid var(--border-bright);
      color: #cbd5e1;
      padding: 8px 6px;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.5px;
      border-radius: 4px;
      cursor: pointer;
      text-align: center;
      text-transform: uppercase;
      transition: all 0.15s;
    }
    .switch-btn:hover { background: #1c293c; color: #fff; }
    .switch-btn.active-green {
      background: #065f46;
      border-color: #10b981;
      color: #a7f3d0;
    }
    .switch-btn.bop-danger {
      background: #450a0a;
      border: 1px solid #ef4444;
      color: #fca5a5;
    }
    .switch-btn.bop-danger:hover { background: #7f1d1d; }
    .switch-btn.bop-sealed {
      background: #15803d;
      border-color: #22c55e;
      color: #dcfce7;
    }

    /* PUMP THROTTLE SLIDER */
    .throttle-card {
      background: #101722;
      border: 1px solid var(--border-dim);
      border-radius: 4px;
      padding: 7px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .slider-header {
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }
    input[type=range] {
      width: 100%;
      height: 5px;
      background: #0a0f16;
      border-radius: 2px;
      accent-color: var(--slb-cyan);
      cursor: pointer;
    }

    /* ALARM BANNER */
    #alarm-banner {
      position: absolute;
      top: 54px;
      left: 50%;
      transform: translateX(-50%);
      background: #991b1b;
      color: #fff;
      border: 1px solid #ef4444;
      box-shadow: 0 4px 16px rgba(239, 68, 68, 0.4);
      padding: 6px 20px;
      border-radius: 4px;
      font-size: 11.5px;
      font-weight: 800;
      letter-spacing: 0.8px;
      display: none;
      z-index: 200;
      animation: alertPulse 1s infinite alternate;
    }
    @keyframes alertPulse { from { opacity: 0.85; } to { opacity: 1; } }

    /* SCREEN 2: IADC WELLSHARP WORKBENCH (Landscape 2-Column) */
    #screen-iadc {
      display: none;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 12px;
      height: 100%;
      box-sizing: border-box;
    }
    #screen-iadc.active { display: grid; }
    .iadc-scroller {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px;
    }
    .lesson-item {
      background: #0e141f;
      border-left: 3px solid #334155;
      border-radius: 0 4px 4px 0;
      padding: 10px;
      font-size: 11px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .lesson-item.passed {
      border-left-color: var(--color-good);
      background: rgba(16, 185, 129, 0.05);
    }
    .lesson-step-head { font-weight: 700; color: #f8fafc; }
    .lesson-step-inst { color: #cbd5e1; font-size: 11.5px; }
    .lesson-step-hint { color: var(--text-muted); font-size: 10px; font-style: italic; }

    /* KILL SHEET FORM TABLE */
    .ks-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      margin-top: 6px;
    }
    .ks-table td {
      padding: 5px 8px;
      border: 1px solid var(--border-dim);
    }
    .ks-label {
      background: #101622;
      color: var(--text-dim);
      font-weight: 600;
      width: 50%;
    }
    .ks-input {
      width: 100%;
      background: #090e15;
      border: 1px solid var(--border-bright);
      color: #fff;
      padding: 4px 8px;
      font-family: var(--font-mono);
      font-size: 12px;
      border-radius: 3px;
    }

    /* SCREEN 3: 3D SPATIAL & CASING INTEGRITY (Landscape 3-Column) */
    #screen-3d-integrity {
      display: none;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      padding: 10px;
      height: 100%;
      box-sizing: border-box;
    }
    #screen-3d-integrity.active { display: grid; }
    #canvas3d {
      width: 100%;
      height: 100%;
      background: #070a0f;
      display: block;
      cursor: grab;
    }

    /* SCREEN 4: STRATEGY COMPARISON */
    #screen-strategy {
      display: none;
      padding: 12px;
      height: 100%;
      box-sizing: border-box;
      flex-direction: column;
    }
    #screen-strategy.active { display: flex; }
    .strategy-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      height: 100%;
    }
    .strategy-card {
      background: var(--bg-panel);
      border: 1px solid var(--border-dim);
      border-radius: 6px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    /* SCREEN 5: CASE STUDIO */
    #screen-case-studio {
      display: none;
      padding: 12px;
      height: 100%;
      box-sizing: border-box;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    #screen-case-studio.active { display: grid; }
    textarea.code-yaml {
      width: 100%;
      height: 100%;
      background: #090e15;
      border: 1px solid var(--border-dim);
      color: #7dd3fc;
      font-family: var(--font-mono);
      font-size: 12px;
      padding: 10px;
      resize: none;
      border-radius: 4px;
    }
  </style>
</head>
<body>

  <!-- TOP COMMAND & STATUS RIBBON -->
  <header id="ribbon">
    <div class="brand-cluster">
      <span class="slb-badge">SLB</span>
      <span class="brand-title">DRILLBENCH</span>
      <span class="brand-subtitle">v2026.1 // Dynamic Well Control</span>
      <span class="case-badge">CASE: DW-EXPLORATION-01</span>
    </div>

    <!-- WORKSPACE TABS -->
    <nav class="tabs-deck">
      <button class="tab-btn active" onclick="switchTab('screen-hydraulics', this)">
        <span>📊</span> Hydraulics &amp; Wellbore
      </button>
      <button class="tab-btn" onclick="switchTab('screen-iadc', this)">
        <span>📋</span> IADC WellSharp &amp; Kill Sheet
      </button>
      <button class="tab-btn" onclick="switchTab('screen-3d-integrity', this)">
        <span>🧭</span> 3D Spatial &amp; Integrity
      </button>
      <button class="tab-btn" onclick="switchTab('screen-strategy', this)">
        <span>⚖️</span> Kill Comparison
      </button>
      <button class="tab-btn" onclick="switchTab('screen-case-studio', this)">
        <span>⚙️</span> Case Studio
      </button>
    </nav>

    <!-- SIMULATION CONTROL DECK -->
    <div class="control-deck">
      <div id="sim-state-pill" class="status-pill">
        <span class="status-dot"></span>
        <span id="rig-badge">DRILLING</span>
      </div>

      <div class="clock-display">
        T = <span id="sim-time">0.0</span>s
      </div>

      <div class="speed-btn-group">
        <button class="speed-btn active" onclick="setSpeed(1, this)">1x</button>
        <button class="speed-btn" onclick="setSpeed(2, this)">2x</button>
        <button class="speed-btn" onclick="setSpeed(5, this)">5x</button>
      </div>

      <button id="btn-run" class="ribbon-btn btn-run-good" onclick="toggleRun()">
        <span id="run-icon">⏸</span> RUNNING
      </button>

      <button class="ribbon-btn" onclick="stepOnce()" title="Step 1 second forward">
        <span>⏭</span> STEP
      </button>

      <button id="btn-audio" class="ribbon-btn" onclick="toggleAudio()">
        <span id="audio-icon">🔔</span> AUDIO
      </button>

      <a href="/api/witsml/trajectory" download="trajectory.witsml.xml" style="text-decoration:none;">
        <button class="ribbon-btn">
          <span>📥</span> WITSML
        </button>
      </a>
    </div>
  </header>

  <!-- ALARM BANNER -->
  <div id="alarm-banner"></div>

  <!-- WORKSPACE SCREENS CONTAINER -->
  <main id="workspace-container">

    <!-- ======================================================== -->
    <!-- SCREEN 1: DRILLING & DYNAMIC HYDRAULICS (LANDSCAPE 4-COL) -->
    <!-- ======================================================== -->
    <div id="screen-hydraulics" class="workspace-screen active">

      <!-- COL 1: WELLBORE CAD ELEVATION -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>WELLBORE ELEVATION</span>
          <span class="cad-badge">TD 12,500'</span>
        </div>
        <div class="cad-body">
          <canvas id="wellboreCanvas" width="210" height="420"></canvas>
        </div>
        <div class="fluid-legend">
          <div class="legend-item"><span class="legend-box" style="background:#1b5e20;"></span> OMW</div>
          <div class="legend-item"><span class="legend-box" style="background:#7b1fa2;"></span> KMW</div>
          <div class="legend-item"><span class="legend-box" style="background:#ef4444;"></span> INFLUX</div>
        </div>
      </section>

      <!-- COL 2: DOWNHOLE DRILLING WINDOW -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>DRILLING OPERATING WINDOW</span>
          <span class="cad-badge">TVD vs EMW</span>
        </div>
        <div class="cad-body">
          <canvas id="windowCanvas" width="240" height="420"></canvas>
        </div>
        <div class="window-metrics">
          <div class="wm-card">
            <span class="wm-label">Csg Shoe MAASP</span>
            <span id="val-maasp" class="wm-val">1248 PSI</span>
          </div>
          <div class="wm-card">
            <span class="wm-label">Shoe Pressure</span>
            <span id="val-shoe-p" class="wm-val">0 PSI</span>
          </div>
        </div>
      </section>

      <!-- COL 3: HIGH-RES EDR STRIP CHART -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>REAL-TIME EDR DRILLING RECORDER</span>
          <div style="display:flex; gap:10px; font-size:9.5px; font-family:var(--font-mono);">
            <span style="color:var(--color-spp);">■ SPP</span>
            <span style="color:var(--color-casing);">■ CICP</span>
            <span style="color:var(--color-pit);">■ GAIN (x50)</span>
            <span style="color:var(--color-target);">--- AUTO CHOKE</span>
          </div>
        </div>
        <div class="edr-chart-container">
          <canvas id="stripChart" width="800" height="280"></canvas>
          <div id="chart-hud" class="chart-cursor-hud">
            <span>SPP: <b id="hud-spp">--</b> psi</span>
            <span>CICP: <b id="hud-cicp">--</b> psi</span>
            <span>GAIN: <b id="hud-gain">--</b> bbl</span>
          </div>
        </div>
        <!-- EDR Live Telemetry Strip -->
        <div class="edr-telemetry-hud">
          <div class="hud-tile">
            <span class="hud-label">Standpipe (SPP)</span>
            <span id="val-spp" class="hud-num" style="color:var(--color-spp);">0</span>
            <span class="hud-unit">PSI</span>
          </div>
          <div class="hud-tile">
            <span class="hud-label">Casing (CICP)</span>
            <span id="val-casing" class="hud-num" style="color:var(--color-casing);">0</span>
            <span class="hud-unit">PSI</span>
          </div>
          <div class="hud-tile">
            <span class="hud-label">Bottomhole (BHP)</span>
            <span id="val-bhp" class="hud-num" style="color:#38bdf8;">0</span>
            <span class="hud-unit">PSI</span>
          </div>
          <div class="hud-tile">
            <span class="hud-label">Active Pit Gain</span>
            <span id="val-gain" class="hud-num" style="color:var(--color-pit);">+0.00</span>
            <span class="hud-unit">BBL</span>
          </div>
          <div class="hud-tile">
            <span class="hud-label">Flow Out Return</span>
            <span id="val-flow" class="hud-num" style="color:var(--color-flow);">0.0</span>
            <span class="hud-unit">% PADDLE</span>
          </div>
          <div class="hud-tile">
            <span class="hud-label">Pit Volume</span>
            <span id="val-pit" class="hud-num" style="color:#e2e8f0;">500.0</span>
            <span class="hud-unit">BBL TOTAL</span>
          </div>
        </div>
      </section>

      <!-- COL 4: RIG CONSOLE & SWACO CHOKE PANEL -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>SWACO CHOKE &amp; RIG CONSOLE</span>
          <span class="cad-badge">HYDRAULIC</span>
        </div>
        <div class="console-stack">
          <!-- Dual Dial Gauges (SPP & Casing) -->
          <div class="dial-card">
            <canvas id="dialsCanvas" width="310" height="130"></canvas>
          </div>

          <!-- SWACO Hydraulic Choke Console -->
          <div class="choke-box">
            <div class="choke-row-main">
              <div>
                <span style="font-size:9px; color:var(--text-muted); font-weight:700; text-transform:uppercase;">CHOKE OPENING</span>
                <div id="val-choke-pct" class="choke-readout">100%</div>
              </div>
              <div class="choke-subreadout">
                <div>VALVE: CAMERON 3"</div>
                <div style="color:var(--color-good);">HYDRAULIC READY</div>
              </div>
            </div>

            <div class="choke-bar-track">
              <div id="choke-bar" class="choke-bar-fill"></div>
            </div>

            <div class="jog-grid">
              <button class="jog-btn" onclick="jogChoke(-10)">◀◀ -10%</button>
              <button class="jog-btn" onclick="jogChoke(-2)">◀ -2%</button>
              <button class="jog-btn" onclick="jogChoke(2)">+2% ▶</button>
              <button class="jog-btn" onclick="jogChoke(10)">+10% ▶▶</button>
            </div>

            <!-- Auto-Choke Subpanel -->
            <div class="autochoke-row">
              <button id="btn-autochoke" class="toggle-pill" onclick="toggleAutoChoke()">AUTO-CHOKE: OFF</button>
              <div style="display:flex; align-items:center; gap:6px;">
                <label style="color:var(--text-muted); font-size:10px;">Target:</label>
                <input type="number" id="txt-choke-target" value="750" style="width:65px; background:#090e15; border:1px solid var(--border-bright); color:#fff; padding:3px 6px; font-family:var(--font-mono); font-size:11px; border-radius:3px;" onchange="updateChokeTarget(this.value)">
                <span style="font-size:9.5px; color:var(--text-muted);">psi</span>
              </div>
            </div>
          </div>

          <!-- Well Control Switchgear -->
          <div class="switch-grid">
            <button id="btn-space" class="switch-btn" onclick="toggleSpaceOut()">SPACE OUT JOINT</button>
            <button id="btn-bop" class="switch-btn bop-danger" onclick="toggleBOP()">CLOSE BOP (RAMS)</button>
            <button id="btn-hcr" class="switch-btn" onclick="toggleHCR()">OPEN HCR VALVE</button>
            <button id="btn-killmud" class="switch-btn" onclick="toggleKillMud()">ACTIVATE KILL MUD</button>
          </div>

          <!-- Mud Pump Throttle -->
          <div class="throttle-card">
            <div class="slider-header">
              <span>TRIPLEX MUD PUMPS 1 &amp; 2</span>
              <span><b id="lbl-spm">100</b> SPM (<span id="lbl-gpm">420</span> GPM)</span>
            </div>
            <input type="range" id="rng-spm" min="0" max="120" value="100" oninput="onPumpChange(this.value)">
          </div>

          <!-- Manual Choke Fallback Slider -->
          <div class="throttle-card">
            <div class="slider-header">
              <span>MANUAL CHOKE SLIDER</span>
              <span><b id="lbl-choke">100</b>% OPEN</span>
            </div>
            <input type="range" id="rng-choke" min="0" max="100" value="100" oninput="onChokeChange(this.value)">
          </div>
        </div>
      </section>
    </div>

    <!-- ======================================================== -->
    <!-- SCREEN 2: IADC WELLSHARP & KILL SHEET (LANDSCAPE 2-COL)   -->
    <!-- ======================================================== -->
    <div id="screen-iadc" class="workspace-screen">
      <!-- Left: Lesson Checklist -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>IADC WELLSHARP CURRICULUM SCENARIO</span>
          <select id="sel-lesson" onchange="changeLesson(this.value)" style="background:#090e15; color:#fff; border:1px solid var(--border-bright); border-radius:3px; padding:2px 8px; font-size:10.5px;">
            <option value="lesson_1_kick_detection">Lesson 1: Kick Detection</option>
            <option value="lesson_2_shut_in">Lesson 2: Shut-In Protocols</option>
            <option value="lesson_3_kill_sheet">Lesson 3: Kill Sheet Math</option>
            <option value="lesson_4_well_kill">Lesson 4: Well Kill Circulation</option>
          </select>
        </div>
        <div style="padding:10px 14px; background:#101722; border-bottom:1px solid var(--border-dim); display:flex; align-items:center; gap:12px;">
          <div style="flex:1;">
            <div style="display:flex; justify-content:space-between; font-size:10.5px; margin-bottom:4px;">
              <span id="lesson-summary" style="color:var(--text-dim);">Kick Detection Drill</span>
              <span id="prog-pct" style="font-family:var(--font-mono); font-weight:700; color:var(--slb-cyan);">0%</span>
            </div>
            <div style="background:#090e15; height:6px; border-radius:3px; overflow:hidden;">
              <div id="prog-bar" style="background:var(--slb-cyan); width:0%; height:100%; transition:width 0.2s;"></div>
            </div>
          </div>
        </div>
        <div id="lesson-steps" class="iadc-scroller"></div>
      </section>

      <!-- Right: Official IADC Kill Sheet -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>OFFICIAL IADC WELL CONTROL KILL SHEET</span>
          <div style="display:flex; gap:6px;">
            <button class="ribbon-btn" onclick="printKillSheetForm()" style="padding:2px 8px; font-size:10px;">PRINT FORM</button>
            <button class="ribbon-btn" onclick="exportKillSheetReport()" style="padding:2px 8px; font-size:10px;">EXPORT MD</button>
          </div>
        </div>
        <div style="flex:1; overflow-y:auto; padding:12px; display:flex; flex-direction:column; gap:12px;">
          <table class="ks-table">
            <tr>
              <td class="ks-label">Original Mud Weight (ppg):</td>
              <td><input type="number" id="ks-kmw" step="0.1" class="ks-input" placeholder="e.g. 11.0"></td>
            </tr>
            <tr>
              <td class="ks-label">Initial Circulating Pressure (ICP, psi):</td>
              <td><input type="number" id="ks-icp" step="1" class="ks-input" placeholder="e.g. 1120"></td>
            </tr>
            <tr>
              <td class="ks-label">Final Circulating Pressure (FCP, psi):</td>
              <td><input type="number" id="ks-fcp" step="1" class="ks-input" placeholder="e.g. 660"></td>
            </tr>
            <tr>
              <td class="ks-label">Initial MAASP (psi):</td>
              <td><input type="number" id="ks-maasp" step="1" class="ks-input" placeholder="e.g. 1248"></td>
            </tr>
          </table>

          <button onclick="submitKillSheet()" class="ribbon-btn btn-run-good" style="justify-content:center; padding:9px; font-size:12px;">
            GRADE IADC KILL SHEET
          </button>
          <div id="ks-feedback"></div>

          <!-- Step Down Pressure Canvas -->
          <div style="margin-top:auto; background:#090e15; border:1px solid var(--border-dim); border-radius:4px; padding:8px;">
            <span style="font-size:10px; color:var(--text-muted); font-weight:700; text-transform:uppercase;">DRILL PIPE PRESSURE STEP-DOWN SCHEDULE (ICP ➔ FCP)</span>
            <canvas id="stepDownCanvas" width="460" height="130" style="width:100%; height:130px; display:block; margin-top:6px;"></canvas>
          </div>
        </div>
      </section>
    </div>

    <!-- ======================================================== -->
    <!-- SCREEN 3: 3D SPATIAL & CASING INTEGRITY (LANDSCAPE 3-COL) -->
    <!-- ======================================================== -->
    <div id="screen-3d-integrity" class="workspace-screen">
      <!-- Col 1: 3D Orbit -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>3D WELLBORE TRAJECTORY</span>
          <button class="ribbon-btn" onclick="reset3DCamera()" style="padding:2px 6px; font-size:9.5px;">RESET VIEW</button>
        </div>
        <div class="cad-body">
          <canvas id="canvas3d" width="340" height="380"></canvas>
        </div>
        <div class="fluid-legend">
          <span style="color:#10b981;">■ DLS &lt; 2°</span>
          <span style="color:#ffd54f;">■ DLS 2-4°</span>
          <span style="color:#ef4444;">■ DLS &gt; 4°/100ft</span>
        </div>
      </section>

      <!-- Col 2: API 5CT Structural Casing Integrity -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>API SPEC 5CT CASING INTEGRITY</span>
          <span id="csg-status" class="cad-badge" style="color:#10b981;">SOUND</span>
        </div>
        <div style="padding:14px; display:flex; flex-direction:column; gap:14px; font-family:var(--font-mono);">
          <div style="background:#0e141f; border:1px solid var(--border-dim); border-radius:4px; padding:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
              <span style="color:var(--text-dim); font-size:11px;">MIN BURST SAFETY FACTOR:</span>
              <span id="csg-burst-sf" style="font-size:16px; font-weight:800; color:#10b981;">--</span>
            </div>
            <div style="background:#070a0f; height:6px; border-radius:3px; overflow:hidden;">
              <div id="burst-bar" style="background:#10b981; width:80%; height:100%;"></div>
            </div>
            <small style="color:var(--text-muted); font-size:9.5px; display:block; margin-top:4px;">Threshold &gt; 1.10 (API RP 90 / 5C3)</small>
          </div>

          <div style="background:#0e141f; border:1px solid var(--border-dim); border-radius:4px; padding:12px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
              <span style="color:var(--text-dim); font-size:11px;">MIN COLLAPSE SAFETY FACTOR:</span>
              <span id="csg-collapse-sf" style="font-size:16px; font-weight:800; color:#10b981;">--</span>
            </div>
            <div style="background:#070a0f; height:6px; border-radius:3px; overflow:hidden;">
              <div id="collapse-bar" style="background:#10b981; width:75%; height:100%;"></div>
            </div>
            <small style="color:var(--text-muted); font-size:9.5px; display:block; margin-top:4px;">Threshold &gt; 1.00 (External Hydrostatic)</small>
          </div>
        </div>
      </section>

      <!-- Col 3: AI State Estimation & Ballooning -->
      <section class="cad-panel">
        <div class="cad-header">
          <span>ADVANCED DIGITAL TWIN &amp; UKF</span>
          <span id="ukf-kick-badge" class="cad-badge" style="color:#10b981;">CONVERGED</span>
        </div>
        <div style="padding:14px; display:flex; flex-direction:column; gap:12px; font-family:var(--font-mono);">
          <!-- Ballooning -->
          <div style="background:#0e141f; border:1px solid var(--border-dim); border-radius:4px; padding:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
              <span style="font-size:10.5px; font-weight:700; color:var(--slb-cyan);">WELLBORE BALLOONING</span>
              <span id="ballooning-badge" class="status-pill" style="font-size:9px;">ANALYZING</span>
            </div>
            <div id="ballooning-details" style="font-size:10px; color:var(--text-dim); margin-top:4px;">Monitoring flowback decay...</div>
          </div>

          <!-- UKF State -->
          <div style="background:#0e141f; border:1px solid var(--border-dim); border-radius:4px; padding:10px; display:flex; flex-direction:column; gap:6px;">
            <span style="font-size:10.5px; font-weight:700; color:var(--slb-cyan);">UNSCENTED KALMAN FILTER</span>
            <div style="display:flex; justify-content:space-between; font-size:11px;">
              <span style="color:var(--text-muted);">Inferred Pore Pressure:</span>
              <span id="ukf-pore" style="font-weight:700; color:#38bdf8;">-- psi</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px;">
              <span style="color:var(--text-muted);">Downhole Influx Vol:</span>
              <span id="ukf-kick-vol" style="font-weight:700; color:#ffd54f;">0.00 bbl</span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:11px;">
              <span style="color:var(--text-muted);">Annular Friction Mult:</span>
              <span id="ukf-fric" style="color:#94a3b8;">1.000</span>
            </div>
          </div>
        </div>
      </section>
    </div>

    <!-- ======================================================== -->
    <!-- SCREEN 4: WELL CONTROL STRATEGY COMPARISON               -->
    <!-- ======================================================== -->
    <div id="screen-strategy" class="workspace-screen">
      <div class="strategy-grid">
        <div class="strategy-card">
          <div style="font-weight:800; color:var(--slb-cyan); font-size:13px; border-bottom:1px solid var(--border-dim); padding-bottom:6px;">DRILLER'S METHOD</div>
          <p style="font-size:11px; color:var(--text-dim);">Two full circulations. 1st cycle: circulate influx with original mud. 2nd cycle: circulate kill mud.</p>
          <div style="margin-top:auto; font-family:var(--font-mono); font-size:11px; display:flex; flex-direction:column; gap:4px;">
            <div>Max Shoe Stress: <b style="color:#ef4444;">High (Longest Exposure)</b></div>
            <div>Complexity: <b style="color:#10b981;">Lowest (Immediate Start)</b></div>
            <div>Circulations: <b>2.0 Cycles</b></div>
          </div>
        </div>

        <div class="strategy-card">
          <div style="font-weight:800; color:#10b981; font-size:13px; border-bottom:1px solid var(--border-dim); padding-bottom:6px;">WAIT &amp; WEIGHT</div>
          <p style="font-size:11px; color:var(--text-dim);">Engineer's single-circulation kill. Mix kill mud weight while shut in, then circulate kill mud directly down string.</p>
          <div style="margin-top:auto; font-family:var(--font-mono); font-size:11px; display:flex; flex-direction:column; gap:4px;">
            <div>Max Shoe Stress: <b style="color:#10b981;">Lowest (Preferred)</b></div>
            <div>Complexity: <b style="color:#ffd54f;">Moderate (Step-Down Chart)</b></div>
            <div>Circulations: <b>1.0 Cycle</b></div>
          </div>
        </div>

        <div class="strategy-card">
          <div style="font-weight:800; color:#f59e0b; font-size:13px; border-bottom:1px solid var(--border-dim); padding-bottom:6px;">VOLUMETRIC METHOD</div>
          <p style="font-size:11px; color:var(--text-dim);">Non-circulating kill. Used when drillstring is plugged or off-bottom. Gas migrates and expands; bleed mud proportionally.</p>
          <div style="margin-top:auto; font-family:var(--font-mono); font-size:11px; display:flex; flex-direction:column; gap:4px;">
            <div>Max Shoe Stress: <b style="color:#f59e0b;">Moderate to High</b></div>
            <div>Complexity: <b style="color:#ef4444;">High (Manual Bleed Cycle)</b></div>
            <div>Circulations: <b>0 (Static Migration)</b></div>
          </div>
        </div>

        <div class="strategy-card">
          <div style="font-weight:800; color:#ef4444; font-size:13px; border-bottom:1px solid var(--border-dim); padding-bottom:6px;">BULLHEADING</div>
          <p style="font-size:11px; color:var(--text-dim);">Pumping heavy mud down the well at high rates to force gas influx directly back into the permeable formation.</p>
          <div style="margin-top:auto; font-family:var(--font-mono); font-size:11px; display:flex; flex-direction:column; gap:4px;">
            <div>Max Shoe Stress: <b style="color:#ef4444;">Extreme (Formation Frac)</b></div>
            <div>Complexity: <b style="color:#ffd54f;">High Risk of Pack-off</b></div>
            <div>Application: <b>H2S or Emergency</b></div>
          </div>
        </div>
      </div>
    </div>

    <!-- ======================================================== -->
    <!-- SCREEN 5: CASE STUDIO & WELL DESIGNER                    -->
    <!-- ======================================================== -->
    <div id="screen-case-studio" class="workspace-screen">
      <section class="cad-panel">
        <div class="cad-header">
          <span>PARAMETRIC WELL SPECIFICATION</span>
          <button class="ribbon-btn btn-run-good" onclick="exportCustomCaseYaml()" style="padding:2px 8px; font-size:10px;">EXPORT CASE YAML</button>
        </div>
        <div style="padding:14px; display:flex; flex-direction:column; gap:10px; font-size:11px;">
          <div>
            <label style="color:var(--text-dim); display:block; margin-bottom:4px;">Case Name:</label>
            <input type="text" id="cs-name" value="deepwater_exploration_01" class="ks-input">
          </div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div>
              <label style="color:var(--text-dim); display:block; margin-bottom:4px;">Total Depth MD (ft):</label>
              <input type="number" id="cs-td" value="12500" class="ks-input">
            </div>
            <div>
              <label style="color:var(--text-dim); display:block; margin-bottom:4px;">Casing Shoe TVD (ft):</label>
              <input type="number" id="cs-shoe" value="10500" class="ks-input">
            </div>
            <div>
              <label style="color:var(--text-dim); display:block; margin-bottom:4px;">Reservoir Pres (psia):</label>
              <input type="number" id="cs-pres" value="6800" class="ks-input">
            </div>
            <div>
              <label style="color:var(--text-dim); display:block; margin-bottom:4px;">Permeability (mD):</label>
              <input type="number" id="cs-perm" value="120" class="ks-input">
            </div>
          </div>
          <div id="cs-status" style="margin-top:10px; color:#10b981; font-family:var(--font-mono); font-size:11px;">Ready to generate production simulation case.</div>
        </div>
      </section>

      <section class="cad-panel">
        <div class="cad-header"><span>GENERATED YAML CASE PREVIEW</span></div>
        <textarea id="yaml-preview" class="code-yaml" readonly></textarea>
      </section>
    </div>

  </main>

  <script>
    let isRunning = true;
    let audioEnabled = false;
    let audioCtx = null;
    let simSpeed = 1;
    let controls = {
      pump_spm: 100,
      choke_pct: 100,
      bop_closed: false,
      hcr_open: false,
      spaced_out: false,
      kill_mud_active: false,
      auto_choke: false,
      auto_choke_target_psi: 750
    };

    // Strip Chart History
    const maxPoints = 80;
    const history = { spp: [], casing: [], gain: [], target: [] };

    // TAB SWITCHER
    function switchTab(screenId, btn) {
      document.querySelectorAll('.workspace-screen').forEach(s => s.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      const scr = document.getElementById(screenId);
      if (scr) scr.classList.add('active');
      if (btn) btn.classList.add('active');
      if (screenId === 'screen-case-studio') updateYamlPreview();
    }

    // AUDIO
    function initAudio() {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    function toggleAudio() {
      initAudio();
      audioEnabled = !audioEnabled;
      const btn = document.getElementById('btn-audio');
      btn.innerHTML = audioEnabled ? '<span>🔔</span> AUDIO' : '<span>🔕</span> MUTED';
      btn.style.color = audioEnabled ? '#10b981' : '#94a3b8';
    }
    function playBeep(freq, duration) {
      if (!audioEnabled || !audioCtx) return;
      try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.frequency.value = freq;
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        gain.gain.value = 0.12;
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
      } catch (e) {}
    }

    // SIM SPEED & CONTROLS
    function setSpeed(spd, btn) {
      simSpeed = spd;
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      if (btn) btn.classList.add('active');
    }

    function toggleRun() {
      isRunning = !isRunning;
      const btn = document.getElementById('btn-run');
      btn.className = isRunning ? 'ribbon-btn btn-run-good' : 'ribbon-btn btn-run-paused';
      btn.innerHTML = isRunning ? '<span>⏸</span> RUNNING' : '<span>▶</span> PAUSED';
      const dot = document.querySelector('.status-dot');
      if (dot) dot.style.background = isRunning ? 'var(--color-good)' : 'var(--color-danger)';
    }

    async function stepOnce() {
      await tick();
    }

    async function tick() {
      if (!isRunning) return;
      try {
        const dtVal = 1.0 * simSpeed;
        const resp = await fetch('/api/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dt: dtVal, controls })
        });
        const st = await resp.json();
        updateUI(st);
        recordHistory(st);
        renderChart();
        drawDials(st.spp_psi, st.casing_pressure_psi);
      } catch (e) {
        console.error("Tick error:", e);
      }
    }

    setInterval(tick, 1000);

    function recordHistory(st) {
      history.spp.push(st.spp_psi);
      history.casing.push(st.casing_pressure_psi);
      history.gain.push(st.pit_gain_bbl * 50);
      history.target.push(controls.auto_choke_target_psi);
      if (history.spp.length > maxPoints) {
        history.spp.shift();
        history.casing.shift();
        history.gain.shift();
        history.target.shift();
      }
    }

    // DRAW EDR STRIP CHART
    function renderChart() {
      const cvs = document.getElementById('stripChart');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;
      ctx.clearRect(0, 0, w, h);

      // Oscilloscope background grid
      ctx.strokeStyle = '#111824';
      ctx.lineWidth = 1;
      for (let y = 0; y < h; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y); ctx.lineTo(w, y);
        ctx.stroke();
      }
      for (let x = 0; x < w; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x, h);
        ctx.stroke();
      }

      const pMax = 5000.0;
      const stepX = w / (maxPoints - 1);

      function drawLine(arr, color, isDashed = false) {
        if (arr.length < 2) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        if (isDashed) ctx.setLineDash([4, 4]);
        else ctx.setLineDash([]);
        for (let i = 0; i < arr.length; i++) {
          const x = i * stepX;
          const y = h - (arr[i] / pMax) * h;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      drawLine(history.target, '#f59e0b', true);
      drawLine(history.gain, '#10b981');
      drawLine(history.casing, '#ff9800');
      drawLine(history.spp, '#00c0f0');

      // Update Cursor HUD with latest values
      if (history.spp.length > 0) {
        document.getElementById('hud-spp').innerText = Math.round(history.spp[history.spp.length - 1]);
        document.getElementById('hud-cicp').innerText = Math.round(history.casing[history.casing.length - 1]);
        document.getElementById('hud-gain').innerText = (history.gain[history.gain.length - 1] / 50).toFixed(2);
      }
    }

    // DRAW SWACO DIALS
    function drawDials(spp, casing) {
      const cvs = document.getElementById('dialsCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      ctx.clearRect(0, 0, cvs.width, cvs.height);

      function drawGauge(cx, cy, r, val, maxVal, title, unit, col) {
        // Outer dark bezel
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, 2 * Math.PI);
        ctx.fillStyle = '#0b1017';
        ctx.fill();
        ctx.strokeStyle = '#223043';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Calibrated scale arc (from 135 deg to 405 deg)
        const startAng = 0.75 * Math.PI;
        const endAng = 2.25 * Math.PI;
        ctx.beginPath();
        ctx.arc(cx, cy, r - 6, startAng, endAng);
        ctx.strokeStyle = '#182436';
        ctx.lineWidth = 4;
        ctx.stroke();

        // Value arc
        const pct = Math.max(0, Math.min(1.0, val / maxVal));
        const valAng = startAng + pct * (endAng - startAng);
        ctx.beginPath();
        ctx.arc(cx, cy, r - 6, startAng, valAng);
        ctx.strokeStyle = col;
        ctx.lineWidth = 4;
        ctx.stroke();

        // Needle
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        const nx = cx + (r - 12) * Math.cos(valAng);
        const ny = cy + (r - 12) * Math.sin(valAng);
        ctx.lineTo(nx, ny);
        ctx.strokeStyle = '#f8fafc';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Center hub
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
        ctx.fillStyle = col;
        ctx.fill();

        // Label & value
        ctx.fillStyle = '#94a3b8';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(title, cx, cy - 14);

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(Math.round(val), cx, cy + 18);
        ctx.fillStyle = '#64748b';
        ctx.font = '7.5px monospace';
        ctx.fillText(unit, cx, cy + 28);
      }

      drawGauge(75, 65, 55, spp, 5000, "STANDPIPE", "PSI", "#00c0f0");
      drawGauge(235, 65, 55, casing, 5000, "CASING", "PSI", "#ff9800");
    }

    // UPDATE UI TELEMETRY
    function updateUI(st) {
      document.getElementById('sim-time').innerText = st.time_s.toFixed(1);
      document.getElementById('val-spp').innerText = Math.round(st.spp_psi);
      document.getElementById('val-casing').innerText = Math.round(st.casing_pressure_psi);
      document.getElementById('val-bhp').innerText = Math.round(st.bhp_psi);
      document.getElementById('val-pit').innerText = st.pit_volume_bbl.toFixed(1);
      document.getElementById('val-gain').innerText = (st.pit_gain_bbl >= 0 ? '+' : '') + st.pit_gain_bbl.toFixed(2);
      document.getElementById('val-flow').innerText = st.flow_out_pct.toFixed(1);
      document.getElementById('val-choke-pct').innerText = Math.round(st.choke_pct) + '%';
      document.getElementById('rng-choke').value = Math.round(st.choke_pct);
      document.getElementById('lbl-choke').innerText = Math.round(st.choke_pct);
      document.getElementById('choke-bar').style.width = Math.round(st.choke_pct) + '%';
      document.getElementById('val-shoe-p').innerText = Math.round(st.shoe_pressure_psi) + ' PSI';
      document.getElementById('val-maasp').innerText = Math.round(st.maasp_psi) + ' PSI';

      // Badge
      const badge = document.getElementById('rig-badge');
      badge.innerText = st.rig_state.replace('_', ' ');

      // Alarms & Audio
      const alarmBanner = document.getElementById('alarm-banner');
      if (st.active_alarms && st.active_alarms.length > 0) {
        alarmBanner.style.display = 'block';
        alarmBanner.innerText = '⚠ ' + st.active_alarms.join(' | ');
        playBeep(880, 0.2);
      } else {
        alarmBanner.style.display = 'none';
      }

      // Lesson
      if (st.lesson) {
        document.getElementById('prog-bar').style.width = st.lesson.progress_pct + '%';
        document.getElementById('prog-pct').innerText = Math.round(st.lesson.progress_pct) + '%';
        let stepsHtml = '';
        st.lesson.steps.forEach(s => {
          stepsHtml += `<div class="lesson-item ${s.passed ? 'passed' : ''}">
            <div class="lesson-step-head">${s.passed ? '✓' : '○'} Step ${s.step_number}: ${s.title}</div>
            <div class="lesson-step-inst">${s.instruction}</div>
            <div class="lesson-step-hint">Hint: ${s.hint}</div>
          </div>`;
        });
        document.getElementById('lesson-steps').innerHTML = stepsHtml;
      }

      drawWellboreSchematic(st);
      drawPressureWindow(st);
    }

    // DRAW WELLBORE SCHEMATIC
    function drawWellboreSchematic(st) {
      const cvs = document.getElementById('wellboreCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const shoeY = h * 0.62;
      const tdY = h * 0.94;

      // Depth ticks on left
      ctx.fillStyle = '#64748b';
      ctx.font = '8px monospace';
      ctx.textAlign = 'left';
      ctx.fillText("0'", 4, 15);
      ctx.fillText("5,000'", 4, h * 0.35);
      ctx.fillText("10,500'", 4, shoeY);
      ctx.fillText("12,500'", 4, tdY);

      // Casing string
      ctx.fillStyle = '#162130';
      ctx.fillRect(cx - 38, 12, 76, shoeY - 12);
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(cx - 38, 12, 76, shoeY - 12);

      // Open hole section
      ctx.fillStyle = '#111722';
      ctx.fillRect(cx - 32, shoeY, 64, tdY - shoeY);
      ctx.strokeStyle = '#78350f';
      ctx.lineWidth = 1;
      ctx.strokeRect(cx - 32, shoeY, 64, tdY - shoeY);

      // Annulus Base Mud (OMW)
      ctx.fillStyle = '#1b5e20';
      ctx.fillRect(cx - 36, 12, 72, shoeY - 12);
      ctx.fillRect(cx - 30, shoeY, 60, tdY - shoeY);

      // Kill mud displacement front
      if (st.kill_mud_in_annulus_bbl > 0) {
        const kmwRatio = Math.min(1.0, st.kill_mud_in_annulus_bbl / 850.0);
        const kmwY = tdY - (tdY - 12) * kmwRatio;
        ctx.fillStyle = '#7b1fa2';
        ctx.fillRect(cx - 30, kmwY, 60, tdY - kmwY);
      }

      // Gas influx bubble
      if (st.influx_volume_bbl > 0) {
        const yTop = 12 + (st.influx_top_md / 12500.0) * (tdY - 12);
        const bubbleH = Math.max(14, (st.influx_volume_bbl / 20.0) * 40);
        ctx.fillStyle = '#ef4444';
        ctx.fillRect(cx - 35, Math.min(tdY - bubbleH, yTop), 70, bubbleH);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 8.5px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('GAS INFLUX', cx, Math.min(tdY - 4, yTop + 10));
      }

      // Drillstring
      ctx.fillStyle = '#94a3b8';
      ctx.fillRect(cx - 6, 12, 12, tdY - 12);
      // Bit
      ctx.fillStyle = '#ffd54f';
      ctx.beginPath();
      ctx.moveTo(cx - 12, tdY - 6);
      ctx.lineTo(cx + 12, tdY - 6);
      ctx.lineTo(cx, tdY + 2);
      ctx.closePath();
      ctx.fill();
    }

    // DRAW DRILLING WINDOW
    function drawPressureWindow(st) {
      const cvs = document.getElementById('windowCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;
      ctx.clearRect(0, 0, w, h);

      const mLeft = 24, mRight = 14, mTop = 15, mBottom = 22;
      const pW = w - mLeft - mRight;
      const pH = h - mTop - mBottom;

      // EMW Grid Lines
      ctx.strokeStyle = '#182434';
      ctx.lineWidth = 1;
      ctx.fillStyle = '#64748b';
      ctx.font = '8px monospace';
      ctx.textAlign = 'center';
      for (let emw = 8; emw <= 16; emw += 2) {
        const x = mLeft + ((emw - 8) / 8.0) * pW;
        ctx.beginPath();
        ctx.moveTo(x, mTop); ctx.lineTo(x, h - mBottom);
        ctx.stroke();
        ctx.fillText(emw + 'p', x, h - 8);
      }

      // Pore Pressure (Cyan: 10.5 ppg)
      const xPore = mLeft + ((10.5 - 8) / 8.0) * pW;
      ctx.strokeStyle = '#00c0f0';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xPore, mTop); ctx.lineTo(xPore, h - mBottom);
      ctx.stroke();

      // Fracture Limit (Red: 14.5 ppg)
      const xFrac = mLeft + ((14.5 - 8) / 8.0) * pW;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(xFrac, mTop); ctx.lineTo(xFrac, h - mBottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Dynamic ECD Curve
      const ecd = Math.max(8.0, Math.min(16.0, st.bhp_psi / (0.052 * 12500.0) || 11.2));
      const xEcd = mLeft + ((ecd - 8) / 8.0) * pW;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(xEcd, mTop); ctx.lineTo(xEcd, h - mBottom);
      ctx.stroke();

      // Current ECD dot indicator
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(xEcd, h - mBottom, 4, 0, 2 * Math.PI);
      ctx.fill();
    }

    // CONTROL ACTIONS
    function onPumpChange(v) {
      controls.pump_spm = parseFloat(v);
      document.getElementById('lbl-spm').innerText = v;
      document.getElementById('lbl-gpm').innerText = Math.round(v * 4.2);
    }
    function onChokeChange(v) {
      controls.choke_pct = parseFloat(v);
      document.getElementById('lbl-choke').innerText = v;
    }
    function jogChoke(delta) {
      let cur = controls.choke_pct + delta;
      cur = Math.max(0, Math.min(100, cur));
      controls.choke_pct = cur;
      document.getElementById('rng-choke').value = cur;
      document.getElementById('lbl-choke').innerText = cur;
    }
    function toggleAutoChoke() {
      controls.auto_choke = !controls.auto_choke;
      const btn = document.getElementById('btn-autochoke');
      btn.className = controls.auto_choke ? 'toggle-pill active' : 'toggle-pill';
      btn.innerText = controls.auto_choke ? 'AUTO-CHOKE: ACTIVE' : 'AUTO-CHOKE: OFF';
    }
    function updateChokeTarget(v) {
      controls.auto_choke_target_psi = parseFloat(v) || 750.0;
    }
    function toggleSpaceOut() {
      controls.spaced_out = !controls.spaced_out;
      const btn = document.getElementById('btn-space');
      btn.className = controls.spaced_out ? 'switch-btn active-green' : 'switch-btn';
    }
    function toggleBOP() {
      controls.bop_closed = !controls.bop_closed;
      const btn = document.getElementById('btn-bop');
      btn.className = controls.bop_closed ? 'switch-btn bop-sealed' : 'switch-btn bop-danger';
      btn.innerText = controls.bop_closed ? 'BOP CLOSED (SEALED)' : 'CLOSE BOP (RAMS)';
    }
    function toggleHCR() {
      controls.hcr_open = !controls.hcr_open;
      const btn = document.getElementById('btn-hcr');
      btn.className = controls.hcr_open ? 'switch-btn active-green' : 'switch-btn';
    }
    function toggleKillMud() {
      controls.kill_mud_active = !controls.kill_mud_active;
      const btn = document.getElementById('btn-killmud');
      btn.className = controls.kill_mud_active ? 'switch-btn active-green' : 'switch-btn';
    }

    async function changeLesson(id) {
      await fetch('/api/lesson/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lesson_id: id })
      });
    }

    async function submitKillSheet() {
      const payload = {
        kmw: parseFloat(document.getElementById('ks-kmw').value || 0),
        icp: parseFloat(document.getElementById('ks-icp').value || 0),
        fcp: parseFloat(document.getElementById('ks-fcp').value || 0),
        maasp: parseFloat(document.getElementById('ks-maasp').value || 0)
      };
      const resp = await fetch('/api/killsheet/grade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const res = await resp.json();
      const fbBox = document.getElementById('ks-feedback');
      let fbHtml = `<div style="font-weight:bold; font-size:11.5px; color:${res.passed ? 'var(--color-good)' : 'var(--color-danger)'};">
        ${res.passed ? 'PASSED (GRADE: ' + res.score_pct + '%)' : 'REVISION REQUIRED (SCORE: ' + res.score_pct + '%)'}
      </div><ul style="margin-left:14px; margin-top:4px; font-size:10.5px; color:var(--text-dim); line-height:1.4;">`;
      res.feedback.forEach(f => { fbHtml += `<li>${f}</li>`; });
      fbHtml += '</ul>';
      fbBox.innerHTML = fbHtml;
      drawStepDownChart(payload.icp, payload.fcp);
    }

    function drawStepDownChart(icp, fcp) {
      const cvs = document.getElementById('stepDownCanvas');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width, h = cvs.height;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = '#182434'; ctx.lineWidth = 1;
      ctx.strokeRect(20, 10, w - 30, h - 30);

      // Line from ICP to FCP
      const yIcp = 15;
      const yFcp = h - 35;
      ctx.strokeStyle = '#00c0f0';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(25, yIcp);
      ctx.lineTo(w - 20, yFcp);
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '8.5px monospace';
      ctx.fillText('ICP ' + (icp || 1120) + ' psi', 30, yIcp + 12);
      ctx.fillText('FCP ' + (fcp || 660) + ' psi', w - 90, yFcp - 6);
    }

    function exportKillSheetReport() { window.open('/api/killsheet/export', '_blank'); }
    function printKillSheetForm() { window.open('/api/killsheet/print', '_blank'); }

    // CASE STUDIO YAML
    function updateYamlPreview() {
      const name = document.getElementById('cs-name').value || 'case';
      const td = document.getElementById('cs-td').value || 12500;
      const shoe = document.getElementById('cs-shoe').value || 10500;
      const pres = document.getElementById('cs-pres').value || 6800;
      const perm = document.getElementById('cs-perm').value || 120;
      const yaml = `name: ${name}
version: 1
reservoir:
  pressure: ${pres} psia
  temperature: 240 degF
  permeability: ${perm} mD
  depth_md: ${td} ft
fluid:
  type: dry_gas
  gas_sg: 0.65
well:
  total_depth_md: ${td} ft
  hole:
    - {top_md: 0, bottom_md: ${shoe}, id: 8.681 in, kind: casing}
    - {top_md: ${shoe}, bottom_md: ${td}, id: 8.5 in, kind: open_hole}
`;
      document.getElementById('yaml-preview').value = yaml;
    }

    async function exportCustomCaseYaml() {
      updateYamlPreview();
      const yamlText = document.getElementById('yaml-preview').value;
      const name = document.getElementById('cs-name').value || 'case';
      const blob = new Blob([yamlText], { type: 'text/yaml' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name + '.yaml';
      a.click();
      document.getElementById('cs-status').innerText = '✓ Exported ' + name + '.yaml successfully!';
    }

    // 3D SPATIAL ENGINE
    let trajData = [];
    let camYaw = 0.8, camPitch = 0.4;
    let isDragging3D = false, lastMouseX = 0, lastMouseY = 0;

    async function init3DTrajectory() {
      try {
        const resp = await fetch('/api/trajectory');
        trajData = await resp.json();
        renderTrajectory3D();
      } catch (e) {}
    }
    function reset3DCamera() {
      camYaw = 0.8; camPitch = 0.4;
      renderTrajectory3D();
    }

    const canvas3d = document.getElementById('canvas3d');
    if (canvas3d) {
      canvas3d.addEventListener('mousedown', (e) => {
        isDragging3D = true; lastMouseX = e.clientX; lastMouseY = e.clientY;
      });
      window.addEventListener('mouseup', () => { isDragging3D = false; });
      window.addEventListener('mousemove', (e) => {
        if (!isDragging3D) return;
        camYaw += (e.clientX - lastMouseX) * 0.015;
        camPitch = Math.max(-1.4, Math.min(1.4, camPitch + (e.clientY - lastMouseY) * 0.015));
        lastMouseX = e.clientX; lastMouseY = e.clientY;
        renderTrajectory3D();
      });
    }

    function renderTrajectory3D() {
      const cv = document.getElementById('canvas3d');
      if (!cv || !trajData || trajData.length === 0) return;
      const ctx = cv.getContext('2d');
      const w = cv.width, h = cv.height;
      ctx.clearRect(0, 0, w, h);
      const cx = w / 2, cy = h / 2 - 10, scale = 0.020;

      function project(n, e, tvd) {
        let x = e, y = n, z = tvd - 6000;
        let x1 = x * Math.cos(camYaw) - y * Math.sin(camYaw);
        let y1 = x * Math.sin(camYaw) + y * Math.cos(camYaw);
        let z1 = z;
        let y2 = y1 * Math.cos(camPitch) - z1 * Math.sin(camPitch);
        let z2 = y1 * Math.sin(camPitch) + z1 * Math.cos(camPitch);
        return { x: cx + x1 * scale, y: cy + z2 * scale };
      }

      for (let i = 0; i < trajData.length - 1; i++) {
        const p1 = project(trajData[i].north_ft, trajData[i].east_ft, trajData[i].tvd_ft);
        const p2 = project(trajData[i + 1].north_ft, trajData[i + 1].east_ft, trajData[i + 1].tvd_ft);
        const dls = trajData[i + 1].dls_deg_100ft || 0.0;
        ctx.strokeStyle = dls > 4.0 ? '#ef4444' : (dls > 2.0 ? '#ffd54f' : '#10b981');
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }

    async function updateIntegrityMonitors() {
      try {
        const [bResp, cResp, kResp] = await Promise.all([
          fetch('/api/ballooning').then(r => r.json()).catch(() => null),
          fetch('/api/casing_integrity').then(r => r.json()).catch(() => null),
          fetch('/api/kalman').then(r => r.json()).catch(() => null)
        ]);

        if (bResp) {
          const badge = document.getElementById('ballooning-badge');
          if (badge) badge.innerText = bResp.diagnosis;
          const det = document.getElementById('ballooning-details');
          if (det) det.innerText = `dq/dt: ${bResp.flow_decay_rate_gpm_per_min} gpm/min | Asymptote: ${bResp.asymptotic_volume_bbl} bbl`;
        }
        if (cResp) {
          document.getElementById('csg-burst-sf').innerText = cResp.min_burst_sf;
          document.getElementById('csg-collapse-sf').innerText = cResp.min_collapse_sf;
        }
        if (kResp) {
          document.getElementById('ukf-pore').innerText = Math.round(kResp.pore_pressure_psi) + ' psi';
          document.getElementById('ukf-kick-vol').innerText = kResp.influx_volume_bbl.toFixed(2) + ' bbl';
          document.getElementById('ukf-fric').innerText = kResp.friction_multiplier.toFixed(3);
        }
      } catch (e) {}
    }

    init3DTrajectory();
    setInterval(updateIntegrityMonitors, 3000);
    updateIntegrityMonitors();
    drawStepDownChart(1120, 660);
  </script>
</body>
</html>
"""


class SimulatorRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        elif parsed.path == "/api/state":
            st_data = SESSION.sim.state.as_dict()
            self._send_json(st_data)
        elif parsed.path == "/api/wits":
            from wellsim.core.wits import WITSRecord01, encode_wits0_packet
            st = SESSION.sim.state
            rec = WITSRecord01(
                spp_psi=st.spp_psi,
                pump_spm=st.controls.pump_spm,
                flow_out_pct=st.flow_out_pct,
                pit_gain_bbl=st.pit_gain_bbl,
                casing_pressure_psi=st.casing_pressure_psi,
                mw_in_ppg=SESSION.sim.kmw_ppg if st.controls.kill_mud_active else SESSION.sim.omw_ppg,
                mw_out_ppg=SESSION.sim.omw_ppg,
                ecd_ppg=round(st.bhp_psi / (0.052 * max(1.0, SESSION.sim.geom.tvd_well_ft)), 2),
                bhp_psi=st.bhp_psi,
            )
            packet = encode_wits0_packet(rec)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=ascii")
            self.end_headers()
            self.wfile.write(packet.encode("ascii"))
        elif parsed.path == "/api/surge_swab":
            from wellsim.core.surge_swab import calculate_trip_envelope, PipeCondition
            sim = SESSION.sim
            max_rih, max_pooh, profile = calculate_trip_envelope(
                tvd_well_ft=sim.geom.tvd_well_ft,
                shoe_tvd_ft=sim.geom.tvd_shoe_ft,
                hole_id_in=sim.geom.hole_id_in,
                pipe_od_in=sim.geom.pipe_od_in,
                pipe_id_in=sim.geom.pipe_id_in,
                rho_ppg=sim.omw_ppg,
                pv_cp=25.0,
                yp_lbf_100ft2=15.0,
                pore_pressure_ppg=sim.formation_pressure_psi / (0.052 * sim.geom.tvd_well_ft),
                frac_gradient_ppg=sim.lot_emw_ppg,
                pipe_condition=PipeCondition.CLOSED_ENDED,
            )
            data = {
                "max_safe_rih_speed_ft_min": round(max_rih, 1),
                "max_safe_pooh_speed_ft_min": round(max_pooh, 1),
                "stations": [
                    {
                        "depth_ft": round(p.depth_ft, 0),
                        "surge_psi": round(p.surge_pressure_psi, 1),
                        "swab_psi": round(p.swab_pressure_psi, 1),
                        "ecd_surge_ppg": round(p.ecd_surge_ppg, 2),
                        "ecd_swab_ppg": round(p.ecd_swab_ppg, 2),
                        "margin_to_frac_psi": round(p.margin_to_frac_psi, 1),
                        "margin_to_pore_psi": round(p.margin_to_pore_psi, 1),
                    }
                    for p in profile
                ],
            }
            self._send_json(data)
        elif parsed.path == "/api/kill_comparison":
            from wellsim.core.kill_comparison import compare_kill_methods
            sim = SESSION.sim
            inputs = KillSheetInputs(
                omw_ppg=sim.omw_ppg,
                tvd_well_ft=sim.geom.tvd_well_ft,
                tvd_shoe_ft=sim.geom.tvd_shoe_ft,
                lot_emw_ppg=sim.lot_emw_ppg,
                sidpp_psi=520.0,
                sicp_psi=750.0,
                pit_gain_bbl=sim.state.pit_gain_bbl or 20.0,
                scrp_psi=sim.scrp_psi,
                drillstring_volume_bbl=sim.geom.drillstring_volume_bbl,
                annular_volume_bbl=sim.geom.annular_volume_bbl,
                annular_capacity_around_bha_bbl_ft=sim.geom.annular_capacity_bbl_ft,
                pump_displacement_bbl_per_stk=sim.geom.pump_displacement_bbl_per_stk,
            )
            res = compare_kill_methods(inputs)
            data = {
                k: {
                    "method_name": v.method_name,
                    "circulations_required": v.circulations_required,
                    "peak_casing_pressure_psi": v.peak_casing_pressure_psi,
                    "max_shoe_pressure_psi": v.max_shoe_pressure_psi,
                    "shoe_fractured": v.shoe_fractured,
                    "total_kill_time_hr": v.total_kill_time_hr,
                    "total_volume_pumped_bbl": v.total_volume_pumped_bbl,
                    "operational_complexity": v.operational_complexity,
                    "key_advantage": v.key_advantage,
                    "key_risk": v.key_risk,
                }
                for k, v in res.items()
            }
            self._send_json(data)
        elif parsed.path == "/api/hole_cleaning":
            from wellsim.core.cuttings_transport import simulate_hole_cleaning_profile
            sim = SESSION.sim
            traj = [
                (sim.geom.tvd_well_ft * 0.25, 10.0),
                (sim.geom.tvd_well_ft * 0.50, 45.0),
                (sim.geom.tvd_well_ft * 0.75, 65.0),
                (sim.geom.tvd_well_ft, 85.0),
            ]
            prof = simulate_hole_cleaning_profile(
                md_total_ft=sim.geom.tvd_well_ft,
                inclinations=traj,
                hole_diameter_in=sim.geom.hole_id_in,
                pipe_od_in=sim.geom.pipe_od_in,
                flow_rate_gpm=max(150.0, sim.state.pump_rate_gpm or 350.0),
                rop_ft_hr=45.0,
                mud_weight_ppg=sim.omw_ppg,
                pv_cp=24.0,
                yp_lbf_100ft2=15.0,
            )
            data = [
                {
                    "md_ft": st.md_ft,
                    "inclination_deg": st.inclination_deg,
                    "annular_velocity_fps": st.annular_velocity_fps,
                    "critical_velocity_fps": st.critical_velocity_fps,
                    "bed_height_fraction": st.bed_height_fraction,
                    "cuttings_concentration_pct": st.cuttings_concentration_pct,
                    "effective_diameter_in": st.effective_diameter_in,
                    "additional_friction_loss_psi": st.additional_friction_loss_psi,
                    "packoff_risk_index": st.packoff_risk_index,
                }
                for st in prof
            ]
            self._send_json(data)
        elif parsed.path == "/api/trajectory":
            from wellsim.core.survey import SurveyStation, calculate_minimum_curvature
            sim = SESSION.sim
            raw = [
                SurveyStation(md_ft=0.0, inclination_deg=0.0, azimuth_deg=45.0),
                SurveyStation(md_ft=sim.geom.tvd_well_ft * 0.3, inclination_deg=0.0, azimuth_deg=45.0),
                SurveyStation(md_ft=sim.geom.tvd_well_ft * 0.6, inclination_deg=40.0, azimuth_deg=45.0),
                SurveyStation(md_ft=sim.geom.tvd_well_ft, inclination_deg=85.0, azimuth_deg=45.0),
            ]
            traj = calculate_minimum_curvature(raw, target_azimuth_deg=45.0)
            self._send_json(traj.to_3d_coordinates())
        elif parsed.path == "/api/witsml/trajectory":
            from wellsim.core.survey import SurveyStation, calculate_minimum_curvature
            from wellsim.core.witsml import generate_trajectory_witsml
            sim = SESSION.sim
            raw = [
                SurveyStation(md_ft=0.0, inclination_deg=0.0, azimuth_deg=45.0),
                SurveyStation(md_ft=sim.geom.tvd_well_ft * 0.3, inclination_deg=0.0, azimuth_deg=45.0),
                SurveyStation(md_ft=sim.geom.tvd_well_ft * 0.6, inclination_deg=40.0, azimuth_deg=45.0),
                SurveyStation(md_ft=sim.geom.tvd_well_ft, inclination_deg=85.0, azimuth_deg=45.0),
            ]
            traj = calculate_minimum_curvature(raw, target_azimuth_deg=45.0)
            xml_data = generate_trajectory_witsml(
                traj.stations, well_name="Rig-01", wellbore_name="Horizontal Lateral"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/xml; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="trajectory.witsml.xml"')
            self.end_headers()
            self.wfile.write(xml_data.encode("utf-8"))
        elif parsed.path == "/api/ballooning":
            from wellsim.core.ballooning import diagnose_flowback_event
            sim = SESSION.sim
            history = getattr(sim, "history", [])
            if len(history) >= 2:
                times = [h.time_s for h in history[-10:]]
                flows = [max(0.0, h.return_flow_pct * 5.0) for h in history[-10:]]
                gains = [h.pit_gain_bbl for h in history[-10:]]
                press = [h.casing_pressure_psi for h in history[-10:]]
                diag = diagnose_flowback_event(times, flows, gains, shut_in_pressures_psi=press)
                data = {
                    "diagnosis": diag.diagnosis.value,
                    "confidence_pct": diag.confidence_pct,
                    "flow_decay_rate_gpm_per_min": diag.flow_decay_rate_gpm_per_min,
                    "asymptotic_volume_bbl": diag.asymptotic_volume_bbl,
                    "sustained_flow_gpm": diag.sustained_flow_gpm,
                    "pressure_rebuild_rate_psi_per_min": diag.pressure_rebuild_rate_psi_per_min,
                    "recommendation": diag.recommendation,
                    "explanation": diag.explanation,
                }
            else:
                data = {
                    "diagnosis": "STABLE",
                    "confidence_pct": 100.0,
                    "flow_decay_rate_gpm_per_min": 0.0,
                    "asymptotic_volume_bbl": 0.0,
                    "sustained_flow_gpm": 0.0,
                    "pressure_rebuild_rate_psi_per_min": 0.0,
                    "recommendation": "Well is stable and under control.",
                    "explanation": "No influx or breathing detected.",
                }
            self._send_json(data)
        elif parsed.path == "/api/casing_integrity":
            from wellsim.core.casing_integrity import (
                CasingGrade,
                CasingSection,
                calculate_casing_integrity,
            )
            sim = SESSION.sim
            csg = CasingSection(
                top_md_ft=0.0,
                bottom_md_ft=sim.geom.tvd_shoe_ft,
                od_in=9.625,
                weight_lbf_ft=47.0,
                wall_thickness_in=0.472,
                grade_yield_psi=CasingGrade.L80,
            )
            pore_ppg = sim.formation_pressure_psi / (0.052 * max(1.0, sim.geom.tvd_well_ft))
            report = calculate_casing_integrity(
                casing=csg,
                surface_casing_pressure_psi=sim.state.casing_pressure_psi or 0.0,
                internal_mud_ppg=sim.omw_ppg,
                external_pore_ppg=pore_ppg,
            )
            data = {
                "min_burst_sf": report.min_burst_sf,
                "min_collapse_sf": report.min_collapse_sf,
                "critical_burst_depth_ft": report.critical_burst_depth_ft,
                "critical_collapse_depth_ft": report.critical_collapse_depth_ft,
                "is_structurally_sound": report.is_structurally_sound,
                "alarms": report.alarms,
            }
            self._send_json(data)
        elif parsed.path == "/api/kalman":
            from wellsim.core.kalman import DrillingControlInput, DrillingObservation, WellboreUKF
            sim = SESSION.sim
            ukf = getattr(SESSION, "_ukf", None)
            if ukf is None:
                ukf = WellboreUKF(initial_pore_pressure_psi=sim.formation_pressure_psi)
                SESSION._ukf = ukf

            ctrl = DrillingControlInput(
                pump_spm=sim.state.controls.pump_spm,
                choke_pct=sim.state.controls.choke_pct,
                well_tvd_ft=sim.geom.tvd_well_ft,
                mud_weight_ppg=sim.omw_ppg,
            )
            obs = DrillingObservation(
                spp_psi=sim.state.spp_psi,
                casing_pressure_psi=sim.state.casing_pressure_psi,
                pit_gain_bbl=sim.state.pit_gain_bbl,
                flow_out_pct=sim.state.flow_out_pct,
            )
            est = ukf.filter_step(ctrl, obs, dt_s=1.0)
            data = {
                "pore_pressure_psi": est.pore_pressure_psi,
                "pore_pressure_std_psi": est.pore_pressure_std_psi,
                "influx_volume_bbl": est.influx_volume_bbl,
                "influx_volume_std_bbl": est.influx_volume_std_bbl,
                "friction_multiplier": est.friction_multiplier,
                "friction_std": est.friction_std,
                "is_kick_detected": est.is_kick_detected,
            }
            self._send_json(data)
        elif parsed.path == "/api/dgd":
            from wellsim.core.dual_gradient import (
                DGDSystemConfig,
                compare_dgd_vs_conventional_profiles,
                evaluate_mlp_operating_state,
            )
            sim = SESSION.sim
            wd = min(sim.geom.tvd_well_ft * 0.4, 6000.0)
            config = DGDSystemConfig(
                water_depth_ft=wd,
                total_depth_ft=sim.geom.tvd_well_ft,
                seawater_density_ppg=8.55,
                mud_density_ppg=sim.omw_ppg,
            )
            mlp = evaluate_mlp_operating_state(
                config, flow_rate_gpm=max(100.0, sim.state.pump_rate_gpm or 500.0)
            )
            pore_curve = [(wd, 8.55), (sim.geom.tvd_shoe_ft, 10.5), (sim.geom.tvd_well_ft, 12.5)]
            frac_curve = [(wd, 10.2), (sim.geom.tvd_shoe_ft, 13.5), (sim.geom.tvd_well_ft, 15.0)]
            profile = compare_dgd_vs_conventional_profiles(config, pore_curve, frac_curve)
            data = {
                "mudline_suction_psi": mlp.suction_pressure_psi,
                "mlp_discharge_psi": mlp.discharge_pressure_psi,
                "mlp_differential_psi": mlp.pump_differential_pressure_psi,
                "mlp_hhp": mlp.hydraulic_horsepower_hhp,
                "stations": [
                    {
                        "depth_ft": s.depth_ft,
                        "p_conv_psi": s.p_conventional_psi,
                        "p_dgd_psi": s.p_dgd_psi,
                        "ecd_conv_ppg": s.ecd_conventional_ppg,
                        "ecd_dgd_ppg": s.ecd_dgd_ppg,
                        "conv_in_window": s.conventional_in_window,
                        "dgd_in_window": s.dgd_in_window,
                    }
                    for s in profile
                ],
            }
            self._send_json(data)
        elif parsed.path == "/api/riser_gas":
            from wellsim.core.riser_gas import MarineRiserGeometry, simulate_riser_gas_unloading
            sim = SESSION.sim
            riser = MarineRiserGeometry(water_depth_ft=min(5000.0, sim.geom.tvd_well_ft * 0.5))
            res = simulate_riser_gas_unloading(
                riser=riser,
                influx_volume_at_bop_bbl=sim.state.pit_gain_bbl or 10.0,
                mud_weight_ppg=sim.omw_ppg,
                dt_s=5.0,
            )
            data = {
                "peak_liquid_unloading_gpm": res.peak_liquid_unloading_gpm,
                "peak_gas_rate_mmscfd": res.peak_gas_rate_mmscfd,
                "peak_diverter_backpressure_psi": res.peak_diverter_backpressure_psi,
                "min_riser_base_pressure_psi": res.min_riser_base_pressure_psi,
                "mgs_blown_through": res.mgs_blown_through,
                "diverter_rating_exceeded": res.diverter_rating_exceeded,
                "summary": res.summary_messages,
            }
            self._send_json(data)
        elif parsed.path == "/api/cementing":
            from wellsim.core.cementing import CementJobConfig, evaluate_api_gas_migration_risk
            sim = SESSION.sim
            job = CementJobConfig(
                shoe_md_ft=sim.geom.tvd_shoe_ft,
                mud_density_ppg=sim.omw_ppg,
                pore_pressure_ppg=sim.formation_pressure_psi / (0.052 * max(1.0, sim.geom.tvd_well_ft)),
            )
            rep = evaluate_api_gas_migration_risk(job)
            data = {
                "transition_time_min": rep.transition_time_min,
                "initial_overbalance_psi": rep.initial_overbalance_psi,
                "pressure_loss_at_500_sgs_psi": rep.pressure_loss_at_500_sgs_psi,
                "net_effective_pressure_at_500_sgs_psi": rep.net_effective_pressure_at_500_sgs_psi,
                "underbalanced_duration_min": rep.underbalanced_duration_min,
                "risk_level": rep.risk_level.value,
                "recommendations": rep.recommendations,
            }
            self._send_json(data)
        elif parsed.path == "/api/killsheet/export":
            inputs = KillSheetInputs(
                omw_ppg=SESSION.sim.omw_ppg,
                tvd_well_ft=SESSION.sim.geom.tvd_well_ft,
                tvd_shoe_ft=SESSION.sim.geom.tvd_shoe_ft,
                lot_emw_ppg=SESSION.sim.lot_emw_ppg,
                sidpp_psi=520.0,
                sicp_psi=750.0,
                pit_gain_bbl=SESSION.sim.state.pit_gain_bbl or 20.0,
                scrp_psi=SESSION.sim.scrp_psi,
                drillstring_volume_bbl=SESSION.sim.geom.drillstring_volume_bbl,
                annular_volume_bbl=SESSION.sim.geom.annular_volume_bbl,
                annular_capacity_around_bha_bbl_ft=SESSION.sim.geom.annular_capacity_bbl_ft,
                pump_displacement_bbl_per_stk=SESSION.sim.geom.pump_displacement_bbl_per_stk,
            )
            res = calculate_kill_sheet(inputs)
            md_report = (
                f"# IADC WellSharp Kill Sheet Report\n\n"
                f"**Generated Time:** {SESSION.sim.state.time_s:.1f} s | **Status:** {SESSION.sim.state.rig_state.value}\n\n"
                f"### Well & Geometry Data\n"
                f"- Well TVD: {inputs.tvd_well_ft:.0f} ft | Casing Shoe TVD: {inputs.tvd_shoe_ft:.0f} ft\n"
                f"- Original Mud Weight: {inputs.omw_ppg:.1f} ppg | LOT EMW: {inputs.lot_emw_ppg:.1f} ppg\n"
                f"- Slow Circulating Rate Pressure: {inputs.scrp_psi:.0f} psi\n\n"
                f"### Shut-In Influx Data\n"
                f"- SIDPP: {inputs.sidpp_psi:.0f} psi | SICP: {inputs.sicp_psi:.0f} psi | Pit Gain: {inputs.pit_gain_bbl:.1f} bbl\n"
                f"- Influx Height: {res.influx_height_ft:.1f} ft | Gradient: {res.influx_gradient_psi_ft:.4f} psi/ft ({res.influx_type.upper()})\n\n"
                f"### Calculated Kill Values\n"
                f"- **Kill Mud Weight (KMW):** {res.kmw_rounded_ppg:.1f} ppg\n"
                f"- **Initial Circulating Pressure (ICP):** {res.icp_psi:.0f} psi\n"
                f"- **Final Circulating Pressure (FCP):** {res.fcp_psi:.0f} psi\n"
                f"- **Initial MAASP:** {res.maasp_initial_psi:.0f} psi | **MAASP with KMW:** {res.maasp_with_kmw_psi:.0f} psi\n"
                f"- Strokes to Bit: {res.strokes_surface_to_bit} | Total Kill Strokes: {res.total_kill_strokes}\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="iadc_kill_sheet_report.md"')
            self.end_headers()
            self.wfile.write(md_report.encode("utf-8"))
        elif parsed.path == "/api/killsheet/print":
            inputs = KillSheetInputs(
                omw_ppg=SESSION.sim.omw_ppg,
                tvd_well_ft=SESSION.sim.geom.tvd_well_ft,
                tvd_shoe_ft=SESSION.sim.geom.tvd_shoe_ft,
                lot_emw_ppg=SESSION.sim.lot_emw_ppg,
                sidpp_psi=520.0,
                sicp_psi=750.0,
                pit_gain_bbl=SESSION.sim.state.pit_gain_bbl or 20.0,
                scrp_psi=SESSION.sim.scrp_psi,
                drillstring_volume_bbl=SESSION.sim.geom.drillstring_volume_bbl,
                annular_volume_bbl=SESSION.sim.geom.annular_volume_bbl,
                annular_capacity_around_bha_bbl_ft=SESSION.sim.geom.annular_capacity_bbl_ft,
                pump_displacement_bbl_per_stk=SESSION.sim.geom.pump_displacement_bbl_per_stk,
            )
            res = calculate_kill_sheet(inputs)
            step_rows = "".join(
                f"<tr><td>{p.strokes}</td><td>{p.dpp_psi:.0f} psi</td></tr>"
                for p in res.step_down.points
            )
            html_print = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>IADC WellSharp Official Kill Sheet</title>
  <style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 30px; color: #111; font-size: 13px; }}
    h1 {{ text-align: center; font-size: 20px; margin-bottom: 5px; text-transform: uppercase; }}
    .sub {{ text-align: center; font-size: 11px; color: #555; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .box {{ border: 1.5px solid #222; border-radius: 4px; padding: 12px; margin-bottom: 15px; }}
    .box-title {{ font-weight: bold; background: #e0e0e0; margin: -12px -12px 10px -12px; padding: 6px 12px; border-bottom: 1.5px solid #222; font-size: 12px; text-transform: uppercase; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    td, th {{ border: 1px solid #ccc; padding: 5px 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .hl {{ font-weight: bold; color: #b71c1c; font-size: 14px; }}
    @media print {{
      .no-print {{ display: none; }}
      body {{ margin: 0; }}
    }}
  </style>
</head>
<body>
  <div class="no-print" style="margin-bottom: 15px; text-align: right;">
    <button onclick="window.print()" style="padding: 8px 16px; background: #1976d2; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">PRINT TO PDF / PAPER</button>
  </div>
  <h1>IADC WellSharp Surface / Subsea Well Control Kill Sheet</h1>
  <div class="sub">Generated by wellsim Autonomous Wild Well Engineering Platform // Certified Standards Compliance</div>
  
  <div class="grid">
    <div>
      <div class="box">
        <div class="box-title">1. Pre-Recorded Well & Geometry Data</div>
        <table>
          <tr><td>Well Total Depth (TVD / MD)</td><td><b>{inputs.tvd_well_ft:.0f} ft</b> / {SESSION.sim.geom.md_well_ft:.0f} ft</td></tr>
          <tr><td>Casing Shoe Depth (TVD)</td><td><b>{inputs.tvd_shoe_ft:.0f} ft</b></td></tr>
          <tr><td>Original Mud Weight (OMW)</td><td><b>{inputs.omw_ppg:.1f} ppg</b></td></tr>
          <tr><td>Leak-Off Test EMW (LOT)</td><td><b>{inputs.lot_emw_ppg:.1f} ppg</b></td></tr>
          <tr><td>Initial MAASP with OMW</td><td class="hl">{res.maasp_initial_psi:.0f} psi</td></tr>
          <tr><td>Slow Circulating Rate (SCR)</td><td><b>{inputs.scrp_psi:.0f} psi</b> at 30 SPM</td></tr>
        </table>
      </div>

      <div class="box">
        <div class="box-title">2. Shut-In Kick & Influx Data</div>
        <table>
          <tr><td>Shut-In Drillpipe Pressure (SIDPP)</td><td><b>{inputs.sidpp_psi:.0f} psi</b></td></tr>
          <tr><td>Shut-In Casing Pressure (SICP)</td><td><b>{inputs.sicp_psi:.0f} psi</b></td></tr>
          <tr><td>Pit Volume Gain</td><td><b>{inputs.pit_gain_bbl:.1f} bbl</b></td></tr>
          <tr><td>Estimated Influx Height</td><td><b>{res.influx_height_ft:.1f} ft</b></td></tr>
          <tr><td>Influx Gradient / Type</td><td><b>{res.influx_gradient_psi_ft:.4f} psi/ft ({res.influx_type.upper()})</b></td></tr>
        </table>
      </div>
    </div>

    <div>
      <div class="box">
        <div class="box-title">3. Calculated Well Control Parameters</div>
        <table>
          <tr><td><b>Kill Mud Weight (KMW)</b></td><td class="hl">{res.kmw_rounded_ppg:.1f} ppg</td></tr>
          <tr><td><b>Initial Circulating Pressure (ICP)</b></td><td class="hl">{res.icp_psi:.0f} psi</td></tr>
          <tr><td><b>Final Circulating Pressure (FCP)</b></td><td class="hl">{res.fcp_psi:.0f} psi</td></tr>
          <tr><td>MAASP with Kill Mud Weight</td><td><b>{res.maasp_with_kmw_psi:.0f} psi</b></td></tr>
          <tr><td>Strokes Surface-to-Bit</td><td><b>{res.strokes_surface_to_bit} strokes</b></td></tr>
          <tr><td>Strokes Bit-to-Surface</td><td><b>{res.strokes_bit_to_surface} strokes</b></td></tr>
          <tr><td>Total Kill Circulating Strokes</td><td><b>{res.total_kill_strokes} strokes</b></td></tr>
        </table>
      </div>

      <div class="box">
        <div class="box-title">4. Drillpipe Step-Down Schedule</div>
        <table>
          <tr><th>Pump Strokes</th><th>Required Drillpipe Pressure</th></tr>
          {step_rows}
        </table>
      </div>
    </div>
  </div>

  <div style="margin-top: 25px; border-top: 1px solid #777; padding-top: 10px; display: flex; justify-content: space-between; font-size: 11px;">
    <div><b>Supervisor Signature:</b> ___________________________</div>
    <div><b>Date & Time:</b> ___________________________</div>
    <div><b>Rig Name / Well ID:</b> wellsim Digital Twin Unit #1</div>
  </div>
</body>
</html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_print.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            body = json.loads(post_body)
        except Exception:
            body = {}

        if parsed.path == "/api/step":
            dt = float(body.get("dt", 1.0))
            ctrls_data = body.get("controls", {})
            ctrls = UserControls(
                pump_spm=float(ctrls_data.get("pump_spm", SESSION.sim.state.controls.pump_spm)),
                choke_pct=float(ctrls_data.get("choke_pct", SESSION.sim.state.controls.choke_pct)),
                bop_closed=bool(ctrls_data.get("bop_closed", SESSION.sim.state.controls.bop_closed)),
                hcr_open=bool(ctrls_data.get("hcr_open", SESSION.sim.state.controls.hcr_open)),
                spaced_out=bool(ctrls_data.get("spaced_out", SESSION.sim.state.controls.spaced_out)),
                kill_mud_active=bool(ctrls_data.get("kill_mud_active", SESSION.sim.state.controls.kill_mud_active)),
                auto_choke=bool(ctrls_data.get("auto_choke", SESSION.sim.state.controls.auto_choke)),
                auto_choke_target_psi=float(ctrls_data.get("auto_choke_target_psi", SESSION.sim.state.controls.auto_choke_target_psi)),
            )
            SESSION.sim.state.controls = ctrls
            st_data = SESSION.step(dt=dt)
            self._send_json(st_data)

        elif parsed.path == "/api/lesson/select":
            lid = body.get("lesson_id", LessonID.KICK_DETECTION.value)
            SESSION.set_lesson(lid)
            self._send_json({"status": "ok", "lesson_id": lid})

        elif parsed.path == "/api/killsheet/grade":
            inputs = KillSheetInputs(
                omw_ppg=10.0,
                tvd_well_ft=10000.0,
                tvd_shoe_ft=6000.0,
                lot_emw_ppg=14.0,
                sidpp_psi=520.0,
                sicp_psi=750.0,
                pit_gain_bbl=20.0,
                scrp_psi=600.0,
                drillstring_volume_bbl=150.0,
                annular_volume_bbl=850.0,
                annular_capacity_around_bha_bbl_ft=0.040,
                pump_displacement_bbl_per_stk=0.10,
            )
            res = grade_kill_sheet_submission(
                inputs=inputs,
                user_kmw=float(body.get("kmw", 0.0)),
                user_icp=float(body.get("icp", 0.0)),
                user_fcp=float(body.get("fcp", 0.0)),
                user_maasp=float(body.get("maasp", 0.0)),
            )
            self._send_json({
                "passed": res.passed,
                "score_pct": res.score_pct,
                "feedback": res.feedback,
                "details": res.details,
            })
        elif parsed.path == "/api/case/export":
            c_name = str(body.get("name", "custom_case"))
            td = float(body.get("td_md_ft", 12500.0))
            shoe = float(body.get("shoe_md_ft", 10500.0))
            pres = float(body.get("pres_psia", 6800.0))
            perm = float(body.get("perm_md", 120.0))
            yaml_content = f"""# Generated by wellsim Visual Case Studio
name: {c_name}
description: Interactive visual well design generated by wellsim
reservoir:
  pressure: {pres:.1f} psia
  temperature: 180.0 degF
  depth_md: {td - 200.0:.1f} ft
  net_pay: 50.0 ft
  permeability: {perm:.1f} md
  wellbore_radius: 0.354 ft
  drainage_radius: 2000.0 ft
  skin: 1.5
fluid:
  type: dry_gas
  gas_sg: 0.65
  co2: 1 %
  n2: 1 %
well:
  total_depth_md: {td:.1f} ft
  hole:
    - {{top_md: 0, bottom_md: {shoe:.1f}, id: 8.681 in, kind: casing}}
    - {{top_md: {shoe:.1f}, bottom_md: {td:.1f}, id: 8.5 in, kind: open_hole}}
flow_path: no_string
exit:
  ambient_pressure: 14.7 psia
"""
            self.send_response(200)
            self.send_header("Content-Type", "text/yaml; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{c_name}.yaml"')
            self.end_headers()
            self.wfile.write(yaml_content.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, data: Any) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        pass


def run_server(port: int = 8080) -> None:
    server_address = ("", port)
    httpd = HTTPServer(server_address, SimulatorRequestHandler)
    print(f"============================================================")
    print(f"  wellsim Drilling Simulator & Well Control Console Online   ")
    print(f"  Access URL: http://localhost:{port}                       ")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down simulator server.")
        httpd.server_close()
