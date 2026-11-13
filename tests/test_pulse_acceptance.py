"""Opt-in real-engine checks for the compiled Pulse testing machine.

PSIM_RUN_PULSE_TESTS=1 PYTHONPATH=src python -m pytest -m slow tests/test_pulse_acceptance.py
"""
import os

import numpy as np
import pytest

from engine.physiology.pulse_adapter import PatientConfig, PulseAdapter

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    os.environ.get("PSIM_RUN_PULSE_TESTS") != "1", reason="requires an installed compiled Pulse engine")]


def test_real_pulse_scalar_and_waveform_replay():
    adapter = PulseAdapter(123, PatientConfig("MALE", 45, 175, 80),
                           waveform_channels=("ART", "PLETH", "ECG_II"))
    try:
        adapter.initialize()
        adapter.advance(1)
        values = adapter.sample_scalars()
        assert {"hr", "map", "sbp", "dbp", "spo2"} <= values.keys()
        assert all(np.isfinite(v) for v in values.values())
        adapter.sample_waveforms(1)
        snapshot = adapter.save_state()
        adapter.advance(1)
        expected_vitals = adapter.sample_scalars()
        expected_waves = adapter.sample_waveforms(1)
        adapter.load_state(snapshot)
        adapter.advance(1)
        assert adapter.sample_scalars() == pytest.approx(expected_vitals, rel=1e-10, abs=1e-10)
        actual_waves = adapter.sample_waveforms(1)
        for channel in expected_waves:
            np.testing.assert_allclose(actual_waves[channel], expected_waves[channel], rtol=1e-10, atol=1e-10)
    finally:
        adapter.close()
