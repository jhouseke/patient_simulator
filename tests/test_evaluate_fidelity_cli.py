import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_fidelity.py"


@pytest.mark.parametrize("value, status, outcome", [(13, 0, "pass"), (20, 1, "fail"), (None, 2, "not_evaluable")])
def test_cli_evaluates_export_and_writes_json(tmp_path, value, status, outcome):
    challenge = {
        "name": "illustration", "status": "draft", "reviewer": None, "review_date": None,
        "provenance": "invented; requires clinical approval", "metrics": [
            {"name": "response", "kind": "response", "channel": "signal",
             "window": {"start": 1, "end": 1}, "baseline": {"start": 0, "end": 0},
             "lower": 2, "upper": 5, "direction": "increase"},
            {"name": "paired", "kind": "paired", "channel": "signal",
             "window": {"start": 1, "end": 1}, "lower": 2, "upper": 5},
        ],
    }
    trajectory = {"truth": [{"time_s": 0, "values": {"signal": 10}},
                            {"time_s": 1, "values": {} if value is None else {"signal": value}}],
                  "observations": []}
    baseline = {"truth": [{"time_s": 1, "values": {"signal": 10}}], "observations": []}
    for name, data in (("challenge", challenge), ("trajectory", trajectory), ("baseline", baseline)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    output = tmp_path / "result.json"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--challenge", str(tmp_path / "challenge.json"),
                           "--trajectory", str(tmp_path / "trajectory.json"), "--intervention-time", "0",
                           "--baseline", str(tmp_path / "baseline.json"), "--output", str(output)],
                          capture_output=True, text=True)
    assert proc.returncode == status, proc.stderr
    result = json.loads(output.read_text())
    assert result["outcome"] == outcome
    assert result["clinical_approval"] is False
    assert len(result["metrics"]) == 2


def test_cli_approved_date_and_invalid_input(tmp_path):
    challenge = {"name": "reviewed", "status": "approved", "reviewer": "Clinician",
                 "review_date": "2026-01-02", "provenance": "review", "metrics": [
                     {"name": "change", "kind": "response", "channel": "x", "lower": 1, "upper": 3,
                      "window": {"start": 1, "end": 1}, "baseline": {"start": 0, "end": 0}}]}
    trajectory = {"truth": [{"time_s": 0, "values": {"x": 1}}, {"time_s": 1, "values": {"x": 3}}],
                  "observations": []}
    (tmp_path / "challenge.json").write_text(json.dumps(challenge))
    (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))
    command = [sys.executable, str(SCRIPT), "--challenge", str(tmp_path / "challenge.json"),
               "--trajectory", str(tmp_path / "trajectory.json"), "--intervention-time", "0"]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["clinical_approval"] is True
    trajectory["truth"].reverse()
    (tmp_path / "trajectory.json").write_text(json.dumps(trajectory))
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "timestamps" in proc.stderr
