"""Tests for the Pulse adapter layer.

The adapter itself (real Pulse calls) is gated behind `-m slow` because it
requires the compiled engine. Everything here is pure-Python and must stay in
the fast suite.
"""

import pytest


def test_scalar_requests_cover_core_metrics():
    from engine.physiology.data_requests import SCALAR_REQUESTS

    codes = {r.metric_code for r in SCALAR_REQUESTS}
    assert {"hr", "map", "rr", "spo2", "temp_c", "co", "cvp", "etco2"} <= codes


def test_scalar_requests_are_unique():
    from engine.physiology.data_requests import SCALAR_REQUESTS

    codes = [r.metric_code for r in SCALAR_REQUESTS]
    assert len(codes) == len(set(codes)), "duplicate metric_code in SCALAR_REQUESTS"


def test_oxygen_saturation_converts_to_percent():
    """Pulse reports OxygenSaturation unitless 0..1; our canonical unit is %."""
    from engine.physiology.data_requests import SCALAR_REQUESTS

    spo2 = next(r for r in SCALAR_REQUESTS if r.metric_code == "spo2")
    assert spo2.pulse_unit == "unitless"
    assert spo2.convert(0.98) == pytest.approx(98.0)


def test_cardiac_output_needs_no_conversion():
    """CardiacOutput is requested directly in L/min, so convert is identity."""
    from engine.physiology.data_requests import SCALAR_REQUESTS

    co = next(r for r in SCALAR_REQUESTS if r.metric_code == "co")
    assert co.pulse_unit == "L/min"
    assert co.convert(5.4) == pytest.approx(5.4)


def test_every_request_has_a_canonical_unit():
    """Each sampled metric must be renderable in snapshots."""
    from engine.pdes.snapshot import _metric_unit
    from engine.physiology.data_requests import SCALAR_REQUESTS

    for req in SCALAR_REQUESTS:
        assert _metric_unit(req.metric_code) != "", (
            f"{req.metric_code} missing from snapshot._metric_unit"
        )


def test_pulse_unit_keys_are_known():
    """pulse_unit strings must map to a real Pulse unit object at build time."""
    from engine.physiology.data_requests import PULSE_UNIT_KEYS, SCALAR_REQUESTS

    for req in SCALAR_REQUESTS:
        assert req.pulse_unit in PULSE_UNIT_KEYS, (
            f"{req.metric_code}: unknown pulse_unit {req.pulse_unit!r}"
        )


# ----------------------------------------------------------------------
# Adapter construction — importable and configurable without Pulse installed
# ----------------------------------------------------------------------


def test_adapter_module_imports_without_pulse():
    """`import pulse` must be deferred to initialize()."""
    from engine.physiology.pulse_adapter import PatientConfig, PulseAdapter

    cfg = PatientConfig(sex="MALE", age_yr=65, height_cm=175, weight_kg=80)
    adapter = PulseAdapter(seed=42, patient_cfg=cfg)
    assert adapter.seed == 42
    assert adapter.cfg.age_yr == 65


def test_adapter_rejects_unknown_waveform_channel():
    from engine.physiology.pulse_adapter import PatientConfig, PulseAdapter

    cfg = PatientConfig(sex="MALE", age_yr=65, height_cm=175, weight_kg=80)
    with pytest.raises(ValueError):
        PulseAdapter(seed=1, patient_cfg=cfg, waveform_channels=("NOPE",))


def test_adapter_requires_initialize_before_use():
    from engine.physiology.pulse_adapter import PatientConfig, PulseAdapter

    cfg = PatientConfig(sex="MALE", age_yr=65, height_cm=175, weight_kg=80)
    adapter = PulseAdapter(seed=1, patient_cfg=cfg)
    with pytest.raises(RuntimeError, match="initialize"):
        adapter.sample_scalars()


def test_advance_is_a_noop_for_nonpositive_durations():
    from engine.physiology.pulse_adapter import PatientConfig, PulseAdapter

    cfg = PatientConfig(sex="MALE", age_yr=65, height_cm=175, weight_kg=80)
    adapter = PulseAdapter(seed=1, patient_cfg=cfg)
    adapter.advance(0.0)  # must not require an engine
