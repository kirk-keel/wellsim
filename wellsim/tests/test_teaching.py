"""Unit tests for the IADC WellSharp teaching module (teaching/scenarios.py)."""

import pytest

from wellsim.core.transient import RigState, UserControls
from wellsim.core.well_control import KillSheetInputs
from wellsim.teaching.scenarios import (
    LessonID,
    get_all_lessons,
    get_lesson,
    grade_kill_sheet_submission,
)


def test_lessons_registered():
    lessons = get_all_lessons()
    assert len(lessons) == 4

    l1 = get_lesson(LessonID.KICK_DETECTION)
    assert l1 is not None
    assert len(l1.steps) == 4
    assert l1.progress_pct == 0.0


def test_lesson_1_step_progression():
    l1 = get_lesson(LessonID.KICK_DETECTION)
    sim = l1.initial_simulator_setup()

    # Step 1: establish 10 bbl pit gain
    for _ in range(35):
        st = sim.step(dt=1.0)
    assert l1.evaluate_step(0, st) is True
    assert l1.steps[0].passed is True

    # Step 2: stop pumps
    st = sim.step(dt=1.0, controls=UserControls(pump_spm=0.0))
    assert l1.evaluate_step(1, st) is True

    # Step 3: space out
    st = sim.step(dt=1.0, controls=UserControls(pump_spm=0.0, spaced_out=True))
    assert l1.evaluate_step(2, st) is True

    # Step 4: flow check
    assert l1.evaluate_step(3, st) is True
    assert l1.progress_pct == 100.0


def test_kill_sheet_submission_grading():
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

    # Perfect submission: KMW=11.0, ICP=1120, FCP=660, MAASP=1248
    res_pass = grade_kill_sheet_submission(
        inputs=inputs,
        user_kmw=11.0,
        user_icp=1120.0,
        user_fcp=660.0,
        user_maasp=1248.0,
    )
    assert res_pass.passed is True
    assert res_pass.score_pct == 100.0

    # Failing submission: wrong values
    res_fail = grade_kill_sheet_submission(
        inputs=inputs,
        user_kmw=12.5,
        user_icp=1500.0,
        user_fcp=800.0,
        user_maasp=1000.0,
    )
    assert res_fail.passed is False
    assert res_fail.score_pct == 0.0
    assert any("FAIL" in f for f in res_fail.feedback)
