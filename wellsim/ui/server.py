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
  <title>wellsim - Drilling Simulator & Well Control Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --bg: #0f141c;
      --card-bg: #1a2230;
      --accent: #00bcd4;
      --accent-warn: #ff9800;
      --accent-danger: #f44336;
      --accent-good: #4caf50;
      --text: #e0e6ed;
      --text-muted: #8c9ba5;
      --border: #2c384a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background-color: var(--bg); color: var(--text); padding: 15px; }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid var(--border); padding-bottom: 12px; margin-bottom: 15px; }
    h1 { font-size: 1.4rem; color: var(--accent); letter-spacing: 1px; }
    .badge { padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; text-transform: uppercase; }
    .badge-drilling { background: #1565c0; color: #fff; }
    .badge-flow-check { background: #e65100; color: #fff; }
    .badge-shut-in { background: #c62828; color: #fff; }
    .badge-circulating-kill { background: #2e7d32; color: #fff; }
    
    .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 15px; }
    @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
    
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 15px; margin-bottom: 15px; }
    .card-title { font-size: 1.05rem; font-weight: 600; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 8px; margin-bottom: 12px; display: flex; justify-content: space-between; }
    
    .gauge-cluster { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
    .gauge-box { background: #121824; border: 1px solid var(--border); border-radius: 6px; padding: 10px; text-align: center; }
    .gauge-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: bold; margin-bottom: 4px; }
    .gauge-value { font-size: 1.5rem; font-weight: bold; color: var(--accent); }
    .gauge-unit { font-size: 0.75rem; color: var(--text-muted); }
    
    .choke-console { display: flex; flex-direction: column; gap: 12px; }
    .choke-controls { display: flex; gap: 10px; align-items: center; justify-content: center; }
    button { background: #2a3b50; color: #fff; border: 1px solid #3d516b; border-radius: 4px; padding: 8px 14px; cursor: pointer; font-weight: 600; transition: background 0.2s; }
    button:hover { background: #3d516b; }
    button.btn-danger { background: #b71c1c; border-color: #e53935; }
    button.btn-danger:hover { background: #c62828; }
    button.btn-good { background: #1b5e20; border-color: #2e7d32; }
    button.btn-good:hover { background: #2e7d32; }
    
    .slider-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; }
    .slider-row label { flex: 1; font-size: 0.85rem; color: var(--text-muted); }
    .slider-row input[type=range] { flex: 2; }
    
    .alarm-banner { background: #b71c1c; color: white; padding: 8px 12px; border-radius: 4px; margin-bottom: 10px; font-weight: bold; animation: pulse 1s infinite; display: none; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    
    .lesson-step { padding: 8px 10px; border-left: 3px solid #555; background: #151b26; margin-bottom: 6px; border-radius: 0 4px 4px 0; font-size: 0.85rem; }
    .lesson-step.passed { border-left-color: var(--accent-good); background: #132418; }
    .lesson-step-title { font-weight: bold; }
  </style>
</head>
<body>

  <header>
    <div>
      <h1>WELLSIM // CYBER DRILLING & WELL CONTROL CONSOLE</h1>
      <small style="color: var(--text-muted);">Industrial Steady-State & Transient Simulation Platform</small>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span id="rig-badge" class="badge badge-drilling">DRILLING</span>
      <button onclick="toggleAudio()" id="btn-audio">AUDIO: MUTED</button>
      <button onclick="toggleRun()" id="btn-run" class="btn-good">SIMULATION RUNNING</button>
    </div>
  </header>

  <div id="alarm-box" class="alarm-banner"></div>

  <div class="grid">
    <!-- Left Column: Gauges, Strip Chart & Controls -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>MAIN GAUGE CLUSTER</span>
          <span style="font-size: 0.8rem; color: var(--text-muted);">Elapsed: <span id="sim-time">0.0</span>s</span>
        </div>
        <div class="gauge-cluster">
          <div class="gauge-box">
            <div class="gauge-label">Standpipe (SPP)</div>
            <div id="val-spp" class="gauge-value" style="color: #29b6f6;">0</div>
            <div class="gauge-unit">PSI</div>
          </div>
          <div class="gauge-box">
            <div class="gauge-label">Casing Pressure</div>
            <div id="val-casing" class="gauge-value" style="color: var(--accent-warn);">0</div>
            <div class="gauge-unit">PSI</div>
          </div>
          <div class="gauge-box">
            <div class="gauge-label">Bottomhole (BHP)</div>
            <div id="val-bhp" class="gauge-value">0</div>
            <div class="gauge-unit">PSI</div>
          </div>
          <div class="gauge-box">
            <div class="gauge-label">Active Pit Volume</div>
            <div id="val-pit" class="gauge-value">500.0</div>
            <div class="gauge-unit">BBL</div>
          </div>
          <div class="gauge-box">
            <div class="gauge-label">Cumulative Pit Gain</div>
            <div id="val-gain" class="gauge-value" style="color: var(--accent-good);">+0.00</div>
            <div class="gauge-unit">BBL</div>
          </div>
          <div class="gauge-box">
            <div class="gauge-label">Flow Out Return</div>
            <div id="val-flow" class="gauge-value">0.0</div>
            <div class="gauge-unit">% PADDLE</div>
          </div>
        </div>
      </div>

      <!-- Real-Time Strip Chart -->
      <div class="card">
        <div class="card-title">REAL-TIME PRESSURE & PIT GAIN STRIP CHART</div>
        <canvas id="stripChart" width="700" height="180" style="width: 100%; height: 180px; background: #0c1017; border-radius: 4px; border: 1px solid var(--border);"></canvas>
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 6px; font-size: 0.75rem;">
          <span style="color: #29b6f6;">■ Standpipe Pressure (SPP)</span>
          <span style="color: #ff9800;">■ Casing Pressure (CICP)</span>
          <span style="color: #4caf50;">■ Pit Gain (+bbl x50)</span>
          <span style="color: #ffd54f;">--- Auto-Choke Target</span>
        </div>
      </div>

      <!-- Wellbore Dynamic Schematic & Drilling Window -->
      <div class="card">
        <div class="card-title">DYNAMIC WELLBORE SCHEMATIC & DRILLING WINDOW</div>
        <div style="display: flex; gap: 15px; justify-content: space-around; align-items: flex-start; background: #121824; padding: 10px; border-radius: 6px;">
          <div style="text-align: center;">
            <small style="color: var(--text-muted); display: block; margin-bottom: 4px;">Dynamic Multi-Fluid Tracking</small>
            <canvas id="wellboreCanvas" width="180" height="240" style="background: #0d121a; border: 1px solid var(--border); border-radius: 4px;"></canvas>
            <div style="display: flex; gap: 6px; font-size: 0.7rem; justify-content: center; margin-top: 4px;">
              <span style="color: #2e7d32;">■ OMW</span>
              <span style="color: #7b1fa2;">■ KMW</span>
              <span style="color: #f44336;">■ Gas Influx</span>
            </div>
          </div>
          <div style="text-align: center; flex-grow: 1;">
            <small style="color: var(--text-muted); display: block; margin-bottom: 4px;">Pore Pressure vs. Dynamic ECD vs. Fracture Limit</small>
            <canvas id="windowCanvas" width="280" height="240" style="background: #0d121a; border: 1px solid var(--border); border-radius: 4px; width: 100%;"></canvas>
            <div style="display: flex; gap: 10px; font-size: 0.7rem; justify-content: center; margin-top: 4px;">
              <span style="color: #29b6f6;">■ Pore Pressure</span>
              <span style="color: #4caf50;">■ Dynamic ECD</span>
              <span style="color: #f44336;">--- Fracture Limit</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 3D Wellbore Trajectory & Geomechanical Integrity Card -->
      <div class="card">
        <div class="card-title">
          <span>3D SPATIAL TRAJECTORY & REAL-TIME INTEGRITY</span>
          <div style="display: flex; gap: 4px;">
            <a href="/api/witsml/trajectory" download="trajectory.witsml.xml" style="text-decoration: none;">
              <button style="padding: 2px 8px; font-size: 0.75rem; background: #00897b;">EXPORT WITSML 1.4</button>
            </a>
            <button onclick="reset3DCamera()" style="padding: 2px 8px; font-size: 0.75rem;">RESET VIEW</button>
          </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: flex-start; background: #121824; padding: 10px; border-radius: 6px;">
          <div style="text-align: center;">
            <canvas id="canvas3d" width="340" height="230" style="background: #080c14; border: 1px solid var(--border); border-radius: 4px; cursor: grab;"></canvas>
            <div style="display: flex; gap: 8px; font-size: 0.7rem; justify-content: center; margin-top: 4px;">
              <span style="color: #4caf50;">■ DLS &lt; 2°</span>
              <span style="color: #ffd54f;">■ DLS 2-4°</span>
              <span style="color: #f44336;">■ DLS &gt; 4°/100ft</span>
              <span style="color: #90a4ae;">(Drag to rotate 3D)</span>
            </div>
          </div>
          <div style="flex-grow: 1; font-size: 0.8rem; display: flex; flex-direction: column; gap: 8px;">
            <!-- Breathing / Kick Discriminator -->
            <div style="background: #1a2230; padding: 8px; border-radius: 4px; border: 1px solid var(--border);">
              <div style="font-weight: bold; color: var(--accent); margin-bottom: 4px;">BREATHING / KICK DISCRIMINATOR</div>
              <div id="ballooning-badge" class="badge badge-drilling" style="display: inline-block;">ANALYZING...</div>
              <div id="ballooning-details" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 4px;">Monitoring flowback decay &amp; asymptotic gain...</div>
            </div>
            <!-- API 5CT Casing Structural Integrity -->
            <div style="background: #1a2230; padding: 8px; border-radius: 4px; border: 1px solid var(--border);">
              <div style="font-weight: bold; color: var(--accent); margin-bottom: 4px;">API 5CT CASING INTEGRITY ENVELOPE</div>
              <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span>Min Burst Safety Factor:</span>
                <span id="csg-burst-sf" style="font-weight: bold; color: #4caf50;">--</span>
              </div>
              <div style="display: flex; justify-content: space-between;">
                <span>Min Collapse Safety Factor:</span>
                <span id="csg-collapse-sf" style="font-weight: bold; color: #4caf50;">--</span>
              </div>
              <div id="csg-status" style="font-size: 0.75rem; color: #4caf50; margin-top: 4px;">STRUCTURAL RATING: SOUND</div>
            </div>
            <!-- UKF Real-Time Digital Twin State Estimator -->
            <div style="background: #1a2230; padding: 8px; border-radius: 4px; border: 1px solid var(--border);">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: bold; color: var(--accent);">UNSCENTED KALMAN FILTER (UKF)</span>
                <span id="ukf-kick-badge" class="badge" style="background: #2e7d32; font-size: 0.65rem;">FILTER CONVERGED</span>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                <span>Inferred Pore Pressure:</span>
                <span id="ukf-pore" style="font-weight: bold; color: #29b6f6;">-- psi</span>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                <span>Downhole Kick Volume:</span>
                <span id="ukf-kick-vol" style="font-weight: bold; color: #ffd54f;">0.00 bbl</span>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                <span>Friction Multiplier:</span>
                <span id="ukf-fric" style="color: #90a4ae;">1.000</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Remote Choke & BOP Console -->
      <div class="card">
        <div class="card-title">REMOTE HYDRAULIC CHOKE CONSOLE</div>
        <div class="choke-console">
          <div style="display: flex; justify-content: space-around; align-items: center; background: #121824; padding: 10px; border-radius: 6px;">
            <div>
              <div class="gauge-label">CHOKE OPENING</div>
              <div id="val-choke-pct" style="font-size: 1.8rem; font-weight: bold; color: var(--accent);">100%</div>
            </div>
            <div>
              <div class="gauge-label">CASING SHOE INTEGRITY</div>
              <div id="val-shoe-p" style="font-size: 1.2rem; font-weight: bold;">0 PSI</div>
              <small style="color: var(--text-muted);">MAASP: <span id="val-maasp">1248</span> PSI</small>
            </div>
          </div>

          <!-- Auto-Choke Subpanel -->
          <div style="display: flex; gap: 10px; align-items: center; background: #141b27; padding: 8px; border-radius: 4px; border: 1px solid var(--border);">
            <button id="btn-autochoke" onclick="toggleAutoChoke()">AUTO-CHOKE: OFF</button>
            <label style="font-size: 0.8rem; color: var(--text-muted);">Target Setpoint (psi):</label>
            <input type="number" id="txt-choke-target" value="750" style="width: 80px; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;" onchange="updateChokeTarget(this.value)">
            <span style="font-size: 0.75rem; color: var(--text-muted);">(PID automated choke adjustment)</span>
          </div>

          <div class="choke-controls">
            <button onclick="jogChoke(-10)">◀◀ CLOSE 10%</button>
            <button onclick="jogChoke(-2)">◀ CLOSE 2%</button>
            <button onclick="jogChoke(2)">OPEN 2% ▶</button>
            <button onclick="jogChoke(10)">OPEN 10% ▶▶</button>
          </div>

          <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
            <button id="btn-space" onclick="toggleSpaceOut()">SPACE OUT TOOL JOINT</button>
            <button id="btn-bop" onclick="toggleBOP()" class="btn-danger">CLOSE BOP (RAMS)</button>
            <button id="btn-hcr" onclick="toggleHCR()">OPEN HCR VALVE</button>
            <button id="btn-killmud" onclick="toggleKillMud()">ACTIVATE KILL MUD</button>
          </div>
        </div>
      </div>

      <!-- Driller Console Inputs -->
      <div class="card">
        <div class="card-title">DRILLER OPERATIONAL THROTTLE</div>
        <div class="slider-row">
          <label>Mud Pump 1 & 2 Speed: <b id="lbl-spm">100</b> SPM (<span id="lbl-gpm">420</span> GPM)</label>
          <input type="range" id="rng-spm" min="0" max="120" value="100" oninput="onPumpChange(this.value)">
        </div>
        <div class="slider-row">
          <label>Choke Position: <b id="lbl-choke">100</b>% Open</label>
          <input type="range" id="rng-choke" min="0" max="100" value="100" oninput="onChokeChange(this.value)">
        </div>
      </div>
    </div>

    <!-- Right Column: Interactive Lesson & Kill Sheet -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>IADC WELLSHARP LESSONS</span>
          <select id="sel-lesson" onchange="changeLesson(this.value)" style="background: #141b27; color: #fff; border: 1px solid var(--border); border-radius: 4px; padding: 2px 6px;">
            <option value="lesson_1_kick_detection">Lesson 1: Kick Detection</option>
            <option value="lesson_2_shut_in">Lesson 2: Shut-In Protocols</option>
            <option value="lesson_3_kill_sheet">Lesson 3: Kill Sheet Math</option>
            <option value="lesson_4_well_kill">Lesson 4: Well Kill Circulation</option>
          </select>
        </div>
        <p id="lesson-summary" style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 10px;"></p>
        <div style="margin-bottom: 10px;">
          <small style="color: var(--text-muted);">Lesson Progress:</small>
          <div style="background: #141b27; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 4px;">
            <div id="prog-bar" style="background: var(--accent-good); width: 0%; height: 100%;"></div>
          </div>
        </div>
        <div id="lesson-steps"></div>
      </div>

      <!-- Kill Sheet Calculator Form -->
      <div class="card">
        <div class="card-title">
          <span>IADC KILL SHEET SUBMISSION</span>
          <div style="display: flex; gap: 4px;">
            <button onclick="printKillSheetForm()" style="padding: 2px 8px; font-size: 0.75rem; background: #3949ab;">PRINT IADC FORM</button>
            <button onclick="exportKillSheetReport()" style="padding: 2px 8px; font-size: 0.75rem;">EXPORT REPORT</button>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85rem;">
          <div>
            <label style="color: var(--text-muted);">Kill Mud Wt (ppg):</label>
            <input type="number" id="ks-kmw" step="0.1" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;" placeholder="e.g. 11.0">
          </div>
          <div>
            <label style="color: var(--text-muted);">Initial ICP (psi):</label>
            <input type="number" id="ks-icp" step="1" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;" placeholder="e.g. 1120">
          </div>
          <div>
            <label style="color: var(--text-muted);">Final FCP (psi):</label>
            <input type="number" id="ks-fcp" step="1" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;" placeholder="e.g. 660">
          </div>
          <div>
            <label style="color: var(--text-muted);">Initial MAASP (psi):</label>
            <input type="number" id="ks-maasp" step="1" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;" placeholder="e.g. 1248">
          </div>
        </div>
        <button onclick="submitKillSheet()" style="width: 100%; margin-top: 10px;" class="btn-good">GRADE KILL SHEET</button>
        <div id="ks-feedback" style="margin-top: 8px; font-size: 0.8rem;"></div>
      </div>

      <!-- Visual Case Studio & Well Designer Card -->
      <div class="card">
        <div class="card-title">
          <span>VISUAL CASE STUDIO &amp; WELL DESIGNER</span>
          <button onclick="exportCustomCaseYaml()" style="padding: 2px 8px; font-size: 0.75rem; background: #00897b;">EXPORT CASE YAML</button>
        </div>
        <div style="font-size: 0.8rem; display: flex; flex-direction: column; gap: 8px;">
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
            <div>
              <label style="color: var(--text-muted); font-size: 0.75rem;">Case Name:</label>
              <input type="text" id="cs-name" value="deepwater_exploration_01" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;">
            </div>
            <div>
              <label style="color: var(--text-muted); font-size: 0.75rem;">Total Depth MD (ft):</label>
              <input type="number" id="cs-td" value="12500" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;">
            </div>
            <div>
              <label style="color: var(--text-muted); font-size: 0.75rem;">Reservoir Pres (psia):</label>
              <input type="number" id="cs-pres" value="6800" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;">
            </div>
            <div>
              <label style="color: var(--text-muted); font-size: 0.75rem;">Permeability (mD):</label>
              <input type="number" id="cs-perm" value="120" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;">
            </div>
          </div>
          <div>
            <label style="color: var(--text-muted); font-size: 0.75rem;">Casing Shoe TVD (ft):</label>
            <input type="number" id="cs-shoe" value="10500" style="width: 100%; background: #121824; border: 1px solid var(--border); color: #fff; padding: 4px; border-radius: 3px;">
          </div>
          <div id="cs-status" style="font-size: 0.75rem; color: #4caf50;">Ready to validate and generate production YAML spec.</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let isRunning = true;
    let audioEnabled = false;
    let audioCtx = null;
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

    function initAudio() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
    }

    function toggleAudio() {
      initAudio();
      audioEnabled = !audioEnabled;
      const btn = document.getElementById('btn-audio');
      btn.innerText = audioEnabled ? 'AUDIO: ENABLED' : 'AUDIO: MUTED';
      btn.className = audioEnabled ? 'btn-good' : '';
    }

    function playBeep(freq, duration) {
      if (!audioEnabled || !audioCtx) return;
      try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.frequency.value = freq;
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        gain.gain.value = 0.15;
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
      } catch (e) {}
    }

    async function tick() {
      if (!isRunning) return;
      try {
        const resp = await fetch('/api/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dt: 1.0, controls })
        });
        const st = await resp.json();
        updateUI(st);
        recordHistory(st);
        renderChart();
      } catch (e) {
        console.error("Tick error:", e);
      }
    }

    setInterval(tick, 1000);

    function recordHistory(st) {
      history.spp.push(st.spp_psi);
      history.casing.push(st.casing_pressure_psi);
      history.gain.push(st.pit_gain_bbl * 50); // Scale for visibility on 0-5000 scale
      history.target.push(controls.auto_choke_target_psi);
      if (history.spp.length > maxPoints) {
        history.spp.shift();
        history.casing.shift();
        history.gain.shift();
        history.target.shift();
      }
    }

    function renderChart() {
      const cvs = document.getElementById('stripChart');
      if (!cvs) return;
      const ctx = cvs.getContext('2d');
      const w = cvs.width;
      const h = cvs.height;
      ctx.clearRect(0, 0, w, h);

      // Grid
      ctx.strokeStyle = '#1b2535';
      ctx.lineWidth = 1;
      for (let y = 0; y < h; y += 45) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
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

      drawLine(history.target, '#ffd54f', true);
      drawLine(history.gain, '#4caf50');
      drawLine(history.spp, '#29b6f6');
      drawLine(history.casing, '#ff9800');
    }

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
      document.getElementById('val-shoe-p').innerText = Math.round(st.shoe_pressure_psi) + ' PSI';
      document.getElementById('val-maasp').innerText = Math.round(st.maasp_psi);

      // Badge
      const badge = document.getElementById('rig-badge');
      badge.innerText = st.rig_state;
      badge.className = 'badge badge-' + st.rig_state.toLowerCase().replace('_', '-');

      // Alarms & Audio
      const alarmBox = document.getElementById('alarm-box');
      if (st.active_alarms && st.active_alarms.length > 0) {
        alarmBox.style.display = 'block';
        alarmBox.innerText = '⚠ ' + st.active_alarms.join(' | ');
        playBeep(880, 0.2); // Alert tone
      } else {
        alarmBox.style.display = 'none';
      }

      // Lesson
      if (st.lesson) {
        document.getElementById('prog-bar').style.width = st.lesson.progress_pct + '%';
        let stepsHtml = '';
        st.lesson.steps.forEach(s => {
          stepsHtml += `<div class="lesson-step ${s.passed ? 'passed' : ''}">
            <div class="lesson-step-title">${s.passed ? '✓' : '○'} Step ${s.step_number}: ${s.title}</div>
            <div>${s.instruction}</div>
            <div style="color: var(--text-muted); font-size: 0.75rem;">Hint: ${s.hint}</div>
          </div>`;
        });
        document.getElementById('lesson-steps').innerHTML = stepsHtml;
      }

      drawWellboreSchematic(st);
      drawPressureWindow(st);
    }

    function drawWellboreSchematic(st) {
      const canvas = document.getElementById('wellboreCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const shoeY = h * 0.60;
      const tdY = h * 0.95;

      // Casing section
      ctx.fillStyle = '#223044';
      ctx.fillRect(cx - 45, 10, 90, shoeY - 10);
      ctx.strokeStyle = '#546e7a';
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - 45, 10, 90, shoeY - 10);

      // Open hole section
      ctx.fillStyle = '#1a1f29';
      ctx.fillRect(cx - 38, shoeY, 76, tdY - shoeY);
      ctx.strokeStyle = '#8d6e63';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(cx - 38, shoeY, 76, tdY - shoeY);

      // Mud fluid in annulus (OMW)
      ctx.fillStyle = '#1b5e20';
      ctx.fillRect(cx - 43, 10, 86, shoeY - 10);
      ctx.fillRect(cx - 36, shoeY, 72, tdY - shoeY);

      // Kill mud displacement front
      if (st.kill_mud_in_annulus_bbl > 0) {
        const kmwRatio = Math.min(1.0, st.kill_mud_in_annulus_bbl / 850.0);
        const kmwY = tdY - (tdY - 10) * kmwRatio;
        ctx.fillStyle = '#7b1fa2';
        ctx.fillRect(cx - 36, kmwY, 72, tdY - kmwY);
      }

      // Gas influx bubble
      if (st.influx_volume_bbl > 0) {
        const yTop = 10 + (st.influx_top_md / 10000.0) * (tdY - 10);
        const bubbleH = Math.max(12, (st.influx_volume_bbl / 20.0) * 35);
        ctx.fillStyle = '#f44336';
        ctx.fillRect(cx - 42, Math.min(tdY - bubbleH, yTop), 84, bubbleH);
        ctx.fillStyle = '#ffffff';
        ctx.font = '9px monospace';
        ctx.fillText('GAS INFLUX', cx - 26, Math.min(tdY - 4, yTop + 10));
      }

      // Drillstring
      ctx.fillStyle = '#90a4ae';
      ctx.fillRect(cx - 8, 10, 16, tdY - 10);
      // Bit
      ctx.fillStyle = '#ffd54f';
      ctx.beginPath();
      ctx.moveTo(cx - 16, tdY - 6);
      ctx.lineTo(cx + 16, tdY - 6);
      ctx.lineTo(cx, tdY + 2);
      ctx.closePath();
      ctx.fill();

      // Labels
      ctx.fillStyle = '#9e9e9e';
      ctx.font = '8px monospace';
      ctx.fillText('0 ft', 4, 18);
      ctx.fillText('Shoe 6k', 4, shoeY);
      ctx.fillText('TD 10k', 4, tdY);
    }

    function drawPressureWindow(st) {
      const canvas = document.getElementById('windowCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const mLeft = 25;
      const mRight = 15;
      const mTop = 15;
      const mBottom = 25;
      const pWidth = w - mLeft - mRight;
      const pHeight = h - mTop - mBottom;

      // Grid
      ctx.strokeStyle = '#1e2838';
      ctx.lineWidth = 1;
      for (let emw = 8; emw <= 16; emw += 2) {
        const x = mLeft + ((emw - 8) / 8.0) * pWidth;
        ctx.beginPath();
        ctx.moveTo(x, mTop);
        ctx.lineTo(x, h - mBottom);
        ctx.stroke();
        ctx.fillStyle = '#607d8b';
        ctx.font = '8px monospace';
        ctx.fillText(emw + ' ppg', x - 12, h - 10);
      }

      // Pore Pressure (blue: 10.5 ppg)
      const xPore = mLeft + ((10.5 - 8) / 8.0) * pWidth;
      ctx.strokeStyle = '#29b6f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xPore, mTop);
      ctx.lineTo(xPore, h - mBottom);
      ctx.stroke();

      // Fracture Limit (red dashed: 14.5 ppg)
      const xFrac = mLeft + ((14.5 - 8) / 8.0) * pWidth;
      ctx.strokeStyle = '#f44336';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(xFrac, mTop);
      ctx.lineTo(xFrac, h - mBottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Dynamic ECD
      const ecd = Math.max(8.0, Math.min(16.0, st.bhp_psi / (0.052 * 10000.0) || 12.0));
      const xEcd = mLeft + ((ecd - 8) / 8.0) * pWidth;
      ctx.strokeStyle = '#4caf50';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(xEcd, mTop);
      ctx.lineTo(xEcd, h - mBottom);
      ctx.stroke();

      // Draw current point indicator
      ctx.fillStyle = '#4caf50';
      ctx.beginPath();
      ctx.arc(xEcd, h - mBottom, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = '9px monospace';
      ctx.fillText('ECD: ' + ecd.toFixed(2) + ' ppg', Math.max(mLeft, Math.min(w - 90, xEcd - 30)), mTop + 14);
    }

    function toggleRun() {
      isRunning = !isRunning;
      const btn = document.getElementById('btn-run');
      btn.innerText = isRunning ? 'SIMULATION RUNNING' : 'SIMULATION PAUSED';
      btn.className = isRunning ? 'btn-good' : 'btn-danger';
    }

    function onPumpChange(v) {
      controls.pump_spm = parseFloat(v);
      document.getElementById('lbl-spm').innerText = v;
      document.getElementById('lbl-gpm').innerText = Math.round(v * 0.1 * 42);
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
      btn.className = controls.auto_choke ? 'btn-good' : '';
      btn.innerText = controls.auto_choke ? 'AUTO-CHOKE: ACTIVE' : 'AUTO-CHOKE: OFF';
    }

    function updateChokeTarget(v) {
      controls.auto_choke_target_psi = parseFloat(v) || 750.0;
    }

    function toggleSpaceOut() {
      controls.spaced_out = !controls.spaced_out;
      const btn = document.getElementById('btn-space');
      btn.className = controls.spaced_out ? 'btn-good' : '';
      btn.innerText = controls.spaced_out ? 'SPACED OUT: YES' : 'SPACE OUT TOOL JOINT';
    }

    function toggleBOP() {
      controls.bop_closed = !controls.bop_closed;
      const btn = document.getElementById('btn-bop');
      btn.className = controls.bop_closed ? 'btn-good' : 'btn-danger';
      btn.innerText = controls.bop_closed ? 'BOP CLOSED (SEALED)' : 'CLOSE BOP (RAMS)';
    }

    function toggleHCR() {
      controls.hcr_open = !controls.hcr_open;
      const btn = document.getElementById('btn-hcr');
      btn.className = controls.hcr_open ? 'btn-good' : '';
      btn.innerText = controls.hcr_open ? 'HCR OPEN' : 'OPEN HCR VALVE';
    }

    function toggleKillMud() {
      controls.kill_mud_active = !controls.kill_mud_active;
      const btn = document.getElementById('btn-killmud');
      btn.className = controls.kill_mud_active ? 'btn-good' : '';
      btn.innerText = controls.kill_mud_active ? 'KILL MUD: ACTIVE' : 'ACTIVATE KILL MUD';
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
      let fbHtml = `<div style="font-weight: bold; color: ${res.passed ? 'var(--accent-good)' : 'var(--accent-danger)'};">
        ${res.passed ? 'CONGRATULATIONS: PASSED (SCORE ' + res.score_pct + '%)' : 'NEEDS REVISION (SCORE ' + res.score_pct + '%)'}
      </div><ul style="margin-left: 15px; margin-top: 5px;">`;
      res.feedback.forEach(f => { fbHtml += `<li>${f}</li>`; });
      fbHtml += '</ul>';
      fbBox.innerHTML = fbHtml;
    }

    function exportKillSheetReport() {
      window.open('/api/killsheet/export', '_blank');
    }

    function printKillSheetForm() {
      window.open('/api/killsheet/print', '_blank');
    }

    async function exportCustomCaseYaml() {
      const payload = {
        name: document.getElementById('cs-name').value || 'custom_case',
        td_md_ft: parseFloat(document.getElementById('cs-td').value) || 12500.0,
        shoe_md_ft: parseFloat(document.getElementById('cs-shoe').value) || 10500.0,
        pres_psia: parseFloat(document.getElementById('cs-pres').value) || 6800.0,
        perm_md: parseFloat(document.getElementById('cs-perm').value) || 120.0,
      };
      try {
        const resp = await fetch('/api/case/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const yamlText = await resp.text();
        const blob = new Blob([yamlText], { type: 'text/yaml' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = payload.name + '.yaml';
        a.click();
        document.getElementById('cs-status').innerText = '✓ Exported ' + payload.name + '.yaml successfully!';
      } catch (e) {
        document.getElementById('cs-status').innerText = 'Export failed: ' + e;
      }
    }

    // 3D Spatial Trajectory Engine & Mouse Orbit Controls
    let trajData = [];
    let camYaw = 0.8;
    let camPitch = 0.4;
    let isDragging3D = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    async function init3DTrajectory() {
      try {
        const resp = await fetch('/api/trajectory');
        trajData = await resp.json();
        renderTrajectory3D();
      } catch (e) {
        console.error("Failed to load trajectory:", e);
      }
    }

    function reset3DCamera() {
      camYaw = 0.8;
      camPitch = 0.4;
      renderTrajectory3D();
    }

    const canvas3d = document.getElementById('canvas3d');
    if (canvas3d) {
      canvas3d.addEventListener('mousedown', (e) => {
        isDragging3D = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      });
      window.addEventListener('mouseup', () => { isDragging3D = false; });
      window.addEventListener('mousemove', (e) => {
        if (!isDragging3D) return;
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        camYaw += dx * 0.015;
        camPitch = Math.max(-1.4, Math.min(1.4, camPitch + dy * 0.015));
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        renderTrajectory3D();
      });
    }

    function renderTrajectory3D() {
      const cv = document.getElementById('canvas3d');
      if (!cv || !trajData || trajData.length === 0) return;
      const ctx = cv.getContext('2d');
      const w = cv.width;
      const h = cv.height;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2 - 10;
      const scale = 0.018;

      const midNorth = trajData[trajData.length - 1].north_ft * 0.5;
      const midEast = trajData[trajData.length - 1].east_ft * 0.5;
      const midTvd = trajData[trajData.length - 1].tvd_ft * 0.5;

      function project(n, e, tvd) {
        let x = (e - midEast);
        let y = (n - midNorth);
        let z = (tvd - midTvd);

        let x1 = x * Math.cos(camYaw) - y * Math.sin(camYaw);
        let y1 = x * Math.sin(camYaw) + y * Math.cos(camYaw);
        let z1 = z;

        let y2 = y1 * Math.cos(camPitch) - z1 * Math.sin(camPitch);
        let z2 = y1 * Math.sin(camPitch) + z1 * Math.cos(camPitch);

        return { x: cx + x1 * scale, y: cy + z2 * scale };
      }

      // Draw Grid / Ground surface reference
      ctx.strokeStyle = '#1a2636';
      ctx.lineWidth = 1;
      const surf0 = project(0, 0, 0);
      const surfN = project(2000, 0, 0);
      const surfE = project(0, 2000, 0);
      ctx.beginPath();
      ctx.moveTo(surf0.x, surf0.y); ctx.lineTo(surfN.x, surfN.y);
      ctx.moveTo(surf0.x, surf0.y); ctx.lineTo(surfE.x, surfE.y);
      ctx.stroke();

      ctx.fillStyle = '#f44336'; ctx.font = '9px monospace'; ctx.fillText('N', surfN.x + 3, surfN.y);
      ctx.fillStyle = '#4caf50'; ctx.fillText('E', surfE.x + 3, surfE.y);

      // Draw Trajectory line segments
      for (let i = 0; i < trajData.length - 1; i++) {
        const p1 = project(trajData[i].north_ft, trajData[i].east_ft, trajData[i].tvd_ft);
        const p2 = project(trajData[i + 1].north_ft, trajData[i + 1].east_ft, trajData[i + 1].tvd_ft);

        const dls = trajData[i + 1].dls_deg_100ft || 0.0;
        let col = '#4caf50';
        if (dls > 4.0) col = '#f44336';
        else if (dls > 2.0) col = '#ffd54f';

        ctx.strokeStyle = col;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }

      // Wellhead marker
      const wh = project(0, 0, 0);
      ctx.fillStyle = '#00bcd4';
      ctx.beginPath(); ctx.arc(wh.x, wh.y, 4, 0, 2 * Math.PI); ctx.fill();
      ctx.fillText('WELLHEAD', wh.x + 6, wh.y + 3);

      // Bit at TD
      const last = trajData[trajData.length - 1];
      const bit = project(last.north_ft, last.east_ft, last.tvd_ft);
      ctx.fillStyle = '#ffd54f';
      ctx.beginPath(); ctx.arc(bit.x, bit.y, 5, 0, 2 * Math.PI); ctx.fill();
      ctx.fillText('BIT: MD ' + Math.round(last.md_ft) + "'", bit.x + 7, bit.y + 3);

      // Overlay HUD stats
      ctx.fillStyle = '#90a4ae';
      ctx.font = '9px monospace';
      ctx.fillText('YAW: ' + (camYaw * 57.3 % 360).toFixed(0) + '° | PITCH: ' + (camPitch * 57.3).toFixed(0) + '°', 8, 14);
    }

    async function updateIntegrityMonitors() {
      try {
        const bResp = await fetch('/api/ballooning');
        const bData = await bResp.json();
        const badge = document.getElementById('ballooning-badge');
        const det = document.getElementById('ballooning-details');
        if (badge && bData) {
          badge.innerText = bData.diagnosis;
          if (bData.diagnosis === 'BALLOONING') {
            badge.className = 'badge';
            badge.style.background = '#e65100';
          } else if (bData.diagnosis.indexOf('KICK') !== -1) {
            badge.className = 'badge';
            badge.style.background = '#f44336';
          } else {
            badge.className = 'badge badge-drilling';
            badge.style.background = '#2e7d32';
          }
          det.innerText = 'dq/dt: ' + bData.flow_decay_rate_gpm_per_min + ' gpm/min | Asymptote: ' + bData.asymptotic_volume_bbl + ' bbl | Conf: ' + bData.confidence_pct + '%';
        }

        const cResp = await fetch('/api/casing_integrity');
        const cData = await cResp.json();
        if (cData) {
          const elBurst = document.getElementById('csg-burst-sf');
          const elColl = document.getElementById('csg-collapse-sf');
          const elStat = document.getElementById('csg-status');
          if (elBurst) {
            elBurst.innerText = cData.min_burst_sf;
            elBurst.style.color = cData.min_burst_sf >= 1.1 ? '#4caf50' : '#f44336';
          }
          if (elColl) {
            elColl.innerText = cData.min_collapse_sf;
            elColl.style.color = cData.min_collapse_sf >= 1.0 ? '#4caf50' : '#f44336';
          }
          if (elStat) {
            elStat.innerText = cData.is_structurally_sound ? 'STRUCTURAL RATING: SOUND' : 'WARNING: CASING STRESS LIMIT EXCEEDED';
            elStat.style.color = cData.is_structurally_sound ? '#4caf50' : '#f44336';
          }
        }

        // 3. UKF Real-Time Digital Twin State Estimator
        const kResp = await fetch('/api/kalman');
        const kData = await kResp.json();
        if (kData) {
          const elPore = document.getElementById('ukf-pore');
          const elKick = document.getElementById('ukf-kick-vol');
          const elFric = document.getElementById('ukf-fric');
          const elBadge = document.getElementById('ukf-kick-badge');
          if (elPore) elPore.innerText = Math.round(kData.pore_pressure_psi) + ' ± ' + Math.round(2 * kData.pore_pressure_std_psi) + ' psi';
          if (elKick) {
            elKick.innerText = kData.influx_volume_bbl.toFixed(2) + ' bbl';
            elKick.style.color = kData.influx_volume_bbl > 1.0 ? '#f44336' : '#ffd54f';
          }
          if (elFric) elFric.innerText = kData.friction_multiplier.toFixed(3);
          if (elBadge) {
            elBadge.innerText = kData.is_kick_detected ? 'KICK DETECTED (UKF)' : 'FILTER CONVERGED';
            elBadge.style.background = kData.is_kick_detected ? '#f44336' : '#2e7d32';
          }
        }
      } catch (e) {
        console.error("Failed to update integrity monitors:", e);
      }
    }

    init3DTrajectory();
    setInterval(updateIntegrityMonitors, 3000);
    updateIntegrityMonitors();
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
