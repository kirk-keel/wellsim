"""IADC WellSharp standard teaching scenarios and automated grading engine.

Provides structured, interactive well control lessons:
- Lesson 1: Kick Detection and Flow Check
- Lesson 2: Shut-In Protocols (Hard vs. Soft Shut-In)
- Lesson 3: Kill Sheet Formulation and Verification
- Lesson 4: Well Kill Execution (Driller's Method and Wait & Weight)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from wellsim.core.transient import RigState, SimulationState, TransientSimulator, UserControls
from wellsim.core.well_control import (
    KillSheetInputs,
    KillSheetResult,
    calculate_kill_sheet,
)

__all__ = [
    "LessonID",
    "LessonStep",
    "LessonScenario",
    "AssessmentResult",
    "get_all_lessons",
    "get_lesson",
    "grade_kill_sheet_submission",
]


class LessonID(str, Enum):
    KICK_DETECTION = "lesson_1_kick_detection"
    SHUT_IN_PROCEDURES = "lesson_2_shut_in"
    KILL_SHEET = "lesson_3_kill_sheet"
    WELL_KILL = "lesson_4_well_kill"


@dataclass
class LessonStep:
    step_number: int
    title: str
    instruction: str
    hint: str
    validator: Callable[[SimulationState], bool]
    passed: bool = False


@dataclass
class LessonScenario:
    lesson_id: LessonID
    title: str
    target_role: str  # e.g. "Driller", "Toolpusher / Supervisor"
    summary: str
    initial_simulator_setup: Callable[[], TransientSimulator]
    steps: list[LessonStep]
    passing_score_pct: float = 80.0

    def evaluate_step(self, step_idx: int, state: SimulationState) -> bool:
        if 0 <= step_idx < len(self.steps):
            step = self.steps[step_idx]
            if not step.passed and step.validator(state):
                step.passed = True
                return True
        return False

    @property
    def progress_pct(self) -> float:
        if not self.steps:
            return 100.0
        passed_count = sum(1 for s in self.steps if s.passed)
        return (passed_count / len(self.steps)) * 100.0


@dataclass
class AssessmentResult:
    lesson_id: str
    passed: bool
    score_pct: float
    feedback: list[str]
    details: dict[str, Any] = field(default_factory=dict)


def _setup_lesson_1() -> TransientSimulator:
    # 10.0 ppg mud, 11.0 ppg pore pressure -> underbalanced kick
    return TransientSimulator(omw_ppg=10.0, formation_pressure_psi=5720.0)


def _setup_lesson_2() -> TransientSimulator:
    sim = TransientSimulator(omw_ppg=10.0, formation_pressure_psi=5720.0)
    # Fast-forward 15 seconds of drilling to establish a 10 bbl influx
    for _ in range(15):
        sim.step(dt=1.0)
    return sim


def _setup_lesson_3() -> TransientSimulator:
    return TransientSimulator(omw_ppg=10.0, formation_pressure_psi=5720.0)


def _setup_lesson_4() -> TransientSimulator:
    sim = TransientSimulator(omw_ppg=10.0, formation_pressure_psi=5720.0)
    # Establish a shut-in well
    for _ in range(15):
        sim.step(dt=1.0)
    sim.step(dt=1.0, controls=UserControls(pump_spm=0.0, choke_pct=0.0, bop_closed=True, spaced_out=True))
    for _ in range(5):
        sim.step(dt=1.0)
    return sim


def get_all_lessons() -> list[LessonScenario]:
    """Retrieve all standard IADC WellSharp training lesson scenarios."""
    lesson_1 = LessonScenario(
        lesson_id=LessonID.KICK_DETECTION,
        title="Lesson 1: Kick Detection & Flow Check",
        target_role="Driller",
        summary=(
            "Monitor drilling parameters (ROP, flow paddle return, pit volume). "
            "Identify an influx from pit gain alert, initiate flow check protocol, "
            "and confirm positive flow."
        ),
        initial_simulator_setup=_setup_lesson_1,
        steps=[
            LessonStep(
                step_number=1,
                title="Identify Pit Gain",
                instruction="Monitor pit volume while drilling until a pit gain exceeds 10 bbl.",
                hint="Observe the Pit Volume Totalizer (PVT) gauge in the main cluster.",
                validator=lambda st: st.pit_gain_bbl >= 10.0,
            ),
            LessonStep(
                step_number=2,
                title="Stop Mud Pumps",
                instruction="Throttle mud pump speed down to 0 SPM.",
                hint="Move the Mud Pump SPM slider to 0.",
                validator=lambda st: st.controls.pump_spm == 0.0,
            ),
            LessonStep(
                step_number=3,
                title="Space Out Tool Joint",
                instruction="Space out the drillstring so a tool joint is clear of the BOP rams.",
                hint="Toggle the 'Space Out Tool Joint' control.",
                validator=lambda st: st.controls.spaced_out is True,
            ),
            LessonStep(
                step_number=4,
                title="Perform Flow Check",
                instruction="Confirm whether the well continues flowing with pumps off.",
                hint="Observe Flow Out % gauge. If flow > 0%, the well is kicking.",
                validator=lambda st: st.rig_state == RigState.FLOW_CHECK and st.flow_out_pct > 0.0,
            ),
        ],
    )

    lesson_2 = LessonScenario(
        lesson_id=LessonID.SHUT_IN_PROCEDURES,
        title="Lesson 2: Well Shut-In Protocols (Hard Shut-In)",
        target_role="Driller & Wellsite Supervisor",
        summary=(
            "Execute standard IADC hard shut-in procedure: Space out, stop pumps, "
            "open HCR valve, close Annular/BOP, and record stabilized SIDPP & SICP."
        ),
        initial_simulator_setup=_setup_lesson_2,
        steps=[
            LessonStep(
                step_number=1,
                title="Space Out and Shut Down Pumps",
                instruction="Verify tool joint is spaced out and mud pump is at 0 SPM.",
                hint="Engage space out and verify pump throttle is 0.",
                validator=lambda st: st.controls.spaced_out and st.controls.pump_spm == 0.0,
            ),
            LessonStep(
                step_number=2,
                title="Close Annular / BOP Rams",
                instruction="Actuate BOP closure to isolate the wellbore annulus.",
                hint="Toggle 'Close BOP Rams' to True.",
                validator=lambda st: st.controls.bop_closed is True,
            ),
            LessonStep(
                step_number=3,
                title="Verify Zero Flow Out",
                instruction="Confirm flow paddle returns to 0% once the BOP is closed.",
                hint="Flow Out gauge should drop to 0.0%.",
                validator=lambda st: st.flow_out_pct == 0.0 and st.controls.bop_closed,
            ),
            LessonStep(
                step_number=4,
                title="Record Stabilized Shut-In Pressures",
                instruction="Allow pressures to stabilize and record SIDPP and SICP.",
                hint="Wait for casing and drillpipe pressures to level off.",
                validator=lambda st: st.rig_state == RigState.SHUT_IN and st.spp_psi > 450.0 and st.casing_pressure_psi > 450.0,
            ),
        ],
    )

    lesson_3 = LessonScenario(
        lesson_id=LessonID.KILL_SHEET,
        title="Lesson 3: Kill Sheet Formulation & Verification",
        target_role="Driller, Toolpusher & Drilling Engineer",
        summary=(
            "Using stabilized shut-in data (SIDPP, SICP, Pit Gain), calculate Kill Mud Weight (KMW), "
            "Initial Circulating Pressure (ICP), Final Circulating Pressure (FCP), and MAASP."
        ),
        initial_simulator_setup=_setup_lesson_3,
        steps=[
            LessonStep(
                step_number=1,
                title="Review Well Data",
                instruction="Inspect TVD (10,000 ft), Shoe TVD (6,000 ft), OMW (10.0 ppg), LOT (14.0 ppg), SCRP (600 psi).",
                hint="Reference data provided in the lesson overview.",
                validator=lambda st: True,
            ),
            LessonStep(
                step_number=2,
                title="Submit Kill Sheet Calculations",
                instruction="Submit values for KMW, ICP, FCP, and MAASP on the interactive form.",
                hint="Use the Kill Sheet verification tool below.",
                validator=lambda st: True,
            ),
        ],
    )

    lesson_4 = LessonScenario(
        lesson_id=LessonID.WELL_KILL,
        title="Lesson 4: Well Kill Circulation (Choke Manipulation)",
        target_role="Driller & Choke Operator",
        summary=(
            "Bring pumps up to slow circulating rate (30 SPM) while adjusting the remote choke "
            "to hold casing pressure constant, then monitor drillpipe pressure against the step-down schedule."
        ),
        initial_simulator_setup=_setup_lesson_4,
        steps=[
            LessonStep(
                step_number=1,
                title="Open HCR Valve",
                instruction="Open the hydraulic choke line (HCR) valve.",
                hint="Toggle HCR Valve to OPEN.",
                validator=lambda st: st.controls.hcr_open is True,
            ),
            LessonStep(
                step_number=2,
                title="Start Pumps to Kill Speed (30 SPM)",
                instruction="Bring mud pumps up to Slow Circulating Rate (30 SPM) while keeping choke cracked open.",
                hint="Adjust choke to ~10-20% and set pump SPM to 30.",
                validator=lambda st: st.controls.pump_spm >= 25.0 and st.controls.choke_pct > 0.0,
            ),
            LessonStep(
                step_number=3,
                title="Circulate Kill Mud",
                instruction="Engage kill mud and observe casing pressure containment below MAASP.",
                hint="Toggle 'Kill Mud Active' to True and maintain casing pressure under 1248 psi.",
                validator=lambda st: st.controls.kill_mud_active and st.casing_pressure_psi < st.maasp_psi,
            ),
        ],
    )

    return [lesson_1, lesson_2, lesson_3, lesson_4]


def get_lesson(lesson_id: str | LessonID) -> LessonScenario | None:
    for les in get_all_lessons():
        if les.lesson_id == lesson_id:
            return les
    return None


def grade_kill_sheet_submission(
    inputs: KillSheetInputs,
    user_kmw: float,
    user_icp: float,
    user_fcp: float,
    user_maasp: float,
) -> AssessmentResult:
    """Grade a trainee's kill sheet submission against exact IADC WellSharp formulas."""
    reference: KillSheetResult = calculate_kill_sheet(inputs)
    feedback: list[str] = []
    points_earned = 0
    total_points = 4

    # 1. KMW: tolerance +/- 0.1 ppg
    kmw_ref = reference.kmw_rounded_ppg
    if abs(user_kmw - kmw_ref) <= 0.15:
        points_earned += 1
        feedback.append(f"PASS: Kill Mud Weight ({user_kmw:.1f} ppg) is correct (Reference: {kmw_ref:.1f} ppg).")
    else:
        feedback.append(
            f"FAIL: Kill Mud Weight ({user_kmw:.1f} ppg) deviates from reference ({kmw_ref:.1f} ppg). "
            f"Formula: KMW = OMW + SIDPP / (0.052 * TVD)"
        )

    # 2. ICP: tolerance +/- 15 psi
    icp_ref = reference.icp_psi
    if abs(user_icp - icp_ref) <= 15.0:
        points_earned += 1
        feedback.append(f"PASS: Initial Circulating Pressure ({user_icp:.0f} psi) is correct (Reference: {icp_ref:.0f} psi).")
    else:
        feedback.append(
            f"FAIL: ICP ({user_icp:.0f} psi) deviates from reference ({icp_ref:.0f} psi). "
            f"Formula: ICP = SIDPP + SCRP"
        )

    # 3. FCP: tolerance +/- 15 psi
    fcp_ref = reference.fcp_psi
    if abs(user_fcp - fcp_ref) <= 15.0:
        points_earned += 1
        feedback.append(f"PASS: Final Circulating Pressure ({user_fcp:.0f} psi) is correct (Reference: {fcp_ref:.0f} psi).")
    else:
        feedback.append(
            f"FAIL: FCP ({user_fcp:.0f} psi) deviates from reference ({fcp_ref:.0f} psi). "
            f"Formula: FCP = SCRP * (KMW / OMW)"
        )

    # 4. MAASP: tolerance +/- 20 psi
    maasp_ref = reference.maasp_initial_psi
    if abs(user_maasp - maasp_ref) <= 20.0:
        points_earned += 1
        feedback.append(f"PASS: Initial MAASP ({user_maasp:.0f} psi) is correct (Reference: {maasp_ref:.0f} psi).")
    else:
        feedback.append(
            f"FAIL: Initial MAASP ({user_maasp:.0f} psi) deviates from reference ({maasp_ref:.0f} psi). "
            f"Formula: MAASP = (LOT - OMW) * 0.052 * TVD_shoe"
        )

    score_pct = (points_earned / total_points) * 100.0
    passed = score_pct >= 75.0  # IADC WellSharp 75% passing standard

    return AssessmentResult(
        lesson_id=LessonID.KILL_SHEET.value,
        passed=passed,
        score_pct=score_pct,
        feedback=feedback,
        details={
            "user_submission": {
                "kmw_ppg": user_kmw,
                "icp_psi": user_icp,
                "fcp_psi": user_fcp,
                "maasp_psi": user_maasp,
            },
            "reference": {
                "kmw_ppg": kmw_ref,
                "icp_psi": icp_ref,
                "fcp_psi": fcp_ref,
                "maasp_psi": maasp_ref,
                "influx_type": reference.influx_type,
            },
        },
    )
