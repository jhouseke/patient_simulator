"""Review export integrates example patients, monitors and channel timestamps."""
import importlib.util
import json
from pathlib import Path
import sys


def test_cli_exports_truth_observations_and_waveform_clocks(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_single_sim.py"
    spec = importlib.util.spec_from_file_location("single_sim_export", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    class Adapter:
        def __init__(self, seed, patient_cfg, waveform_channels):
            self.waveform_channels = waveform_channels
            self.last_waveform_start_s = {"ART": .008}
            self.cfg = patient_cfg
            self.t = 0
            self.closed = False
        def initialize(self): pass
        def close(self): self.closed = True
        def advance(self, duration): self.t += duration
        def sample_scalars(self): return {"hr": 70 + self.t}
        def sample_waveforms(self, duration): return {"ART": [1.0] * round(duration * 125)}
        def save_state(self): return b""
    monkeypatch.setattr(cli, "PulseAdapter", Adapter)
    output = tmp_path / "trajectory.json"
    monitor = tmp_path / "monitor.json"
    monitor.write_text('{"hr": {"lag_s": 5}}')
    monkeypatch.setattr(sys, "argv", [str(path), "--hours", str(10/3600), "--quiet",
        "--phenotype", "older_adult", "--observation-config", str(monitor),
        "--waveforms", "ART", "--output", str(output)])
    cli.main()
    result = json.loads(output.read_text())
    assert result["truth"][0]["values"]["hr"] == 75
    assert result["observations"][0]["values"]["hr"] == 70
    assert result["phenotype"]["review_status"] == "clinician-review-pending"
    assert result["waveforms"][0]["channel_start_time_s"]["ART"] == .008
    assert result["clinical_validation"] == "pending"
