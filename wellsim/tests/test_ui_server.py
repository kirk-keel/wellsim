"""Unit tests for the UI web server and API endpoints (ui/server.py)."""

import json
from unittest.mock import MagicMock

import pytest

from wellsim.ui.server import SESSION, SimulatorRequestHandler


def test_ui_session_state_and_step():
    st = SESSION.step(dt=1.0)
    assert "spp_psi" in st
    assert "casing_pressure_psi" in st
    assert "pit_volume_bbl" in st
    assert "lesson" in st
    assert st["lesson"]["id"] == "lesson_1_kick_detection"


def test_ui_session_switch_lesson():
    SESSION.set_lesson("lesson_2_shut_in")
    st = SESSION.step(dt=1.0)
    assert st["lesson"]["id"] == "lesson_2_shut_in"
    assert st["lesson"]["progress_pct"] >= 0.0

    # Switch back
    SESSION.set_lesson("lesson_1_kick_detection")


def test_ui_wits_and_surge_swab_endpoints():
    handler = SimulatorRequestHandler.__new__(SimulatorRequestHandler)
    handler.wfile = MagicMock()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    # Test /api/wits
    handler.path = "/api/wits"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/surge_swab
    handler.path = "/api/surge_swab"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/kill_comparison
    handler.path = "/api/kill_comparison"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/hole_cleaning
    handler.path = "/api/hole_cleaning"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/trajectory
    handler.path = "/api/trajectory"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/killsheet/print
    handler.path = "/api/killsheet/print"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/witsml/trajectory
    handler.path = "/api/witsml/trajectory"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/ballooning
    handler.path = "/api/ballooning"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/casing_integrity
    handler.path = "/api/casing_integrity"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/kalman
    handler.path = "/api/kalman"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/dgd
    handler.path = "/api/dgd"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/riser_gas
    handler.path = "/api/riser_gas"
    handler.do_GET()
    handler.send_response.assert_called_with(200)

    # Test /api/cementing
    handler.path = "/api/cementing"
    handler.do_GET()
    handler.send_response.assert_called_with(200)


def test_ui_case_export_post():
    handler = SimulatorRequestHandler.__new__(SimulatorRequestHandler)
    handler.wfile = MagicMock()
    handler.rfile = MagicMock()
    handler.headers = {"Content-Length": "120"}
    body_json = json.dumps({
        "name": "gulf_of_mexico_deep",
        "td_md_ft": 15000.0,
        "shoe_md_ft": 11000.0,
        "pres_psia": 8500.0,
        "perm_md": 75.0,
    })
    handler.rfile.read.return_value = body_json.encode("utf-8")
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    handler.path = "/api/case/export"
    handler.do_POST()
    handler.send_response.assert_called_with(200)
    handler.send_header.assert_any_call("Content-Type", "text/yaml; charset=utf-8")
