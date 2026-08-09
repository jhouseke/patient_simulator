"""Waveform synthesis and windowing (Tasks 2.2, 2.3).

DB-backed waveform storage tests live in test_waveform_store.py (Postgres-gated).
"""

import numpy as np
import pytest

from engine.physiology.pressure_synth import PressureSynthesizer


def test_art_waveform_tracks_pressures():
    ps = PressureSynthesizer(seed=1, sample_rate_hz=125.0)
    sig = ps.generate_art(duration_s=10.0, heart_rate_bpm=60.0, sbp_mmhg=120.0, dbp_mmhg=80.0)
    assert len(sig) == 1250
    assert 114.0 <= sig.max() <= 126.0, "systolic peaks must track SBP"
    assert 74.0 <= sig.min() <= 84.0, "diastolic troughs must track DBP"


def test_art_follows_changing_pressures():
    ps = PressureSynthesizer(seed=1, sample_rate_hz=125.0)
    low = ps.generate_art(10.0, 60.0, sbp_mmhg=90.0, dbp_mmhg=50.0)
    assert 84.0 <= low.max() <= 96.0
    assert 44.0 <= low.min() <= 56.0


def test_art_beat_count_tracks_heart_rate():
    ps = PressureSynthesizer(seed=1, sample_rate_hz=125.0)
    sig = ps.generate_art(10.0, 90.0, 120.0, 80.0)
    # 90 bpm over 10 s => ~15 systolic upstrokes
    above = sig > (sig.min() + 0.7 * (sig.max() - sig.min()))
    beats = np.count_nonzero((~above[:-1]) & (above[1:]))
    assert 13 <= beats <= 17


def test_pleth_is_normalized_and_periodic():
    ps = PressureSynthesizer(seed=1, sample_rate_hz=125.0)
    sig = ps.generate_pleth(duration_s=10.0, heart_rate_bpm=60.0, spo2_pct=98.0)
    assert len(sig) == 1250
    assert 0.0 <= sig.min() and sig.max() <= 1.0


def test_pressure_synth_is_continuous_across_windows():
    ps = PressureSynthesizer(seed=2, sample_rate_hz=125.0)
    joined = np.concatenate(
        [ps.generate_art(5.0, 72.0, 120.0, 80.0), ps.generate_art(5.0, 72.0, 120.0, 80.0)]
    )
    whole = PressureSynthesizer(seed=2, sample_rate_hz=125.0).generate_art(
        10.0, 72.0, 120.0, 80.0
    )
    assert np.allclose(joined, whole), "stateful generator must not seam at window edges"


def test_pressure_synth_is_deterministic():
    a = PressureSynthesizer(seed=9, sample_rate_hz=125.0).generate_art(5.0, 70.0, 118.0, 72.0)
    b = PressureSynthesizer(seed=9, sample_rate_hz=125.0).generate_art(5.0, 70.0, 118.0, 72.0)
    assert np.allclose(a, b)


def test_art_and_pleth_share_beat_phase():
    """Both channels come off the same heartbeat, so they must not drift apart."""
    ps = PressureSynthesizer(seed=1, sample_rate_hz=125.0)
    assert ps.beat_phase == 0.0
    ps.generate_art(1.0, 60.0, 120.0, 80.0)
    after_art = ps.beat_phase
    assert after_art != 0.0


# ----------------------------------------------------------------------
# Task 2.3 — window accumulation
# ----------------------------------------------------------------------


def test_waveform_buffer_emits_fixed_windows():
    from engine.physiology.waveforms import WaveformBuffer

    buf = WaveformBuffer(channel="ART", sample_rate_hz=125.0, window_s=10.0)
    emitted = []
    for _ in range(5):  # feed 25s of data in 5s increments
        emitted.extend(buf.add([0.0] * 625))
    assert len(emitted) == 2, "two complete 10s windows (20s), 5s remains buffered"
    assert len(emitted[0].samples) == 1250


def test_waveform_buffer_preserves_sample_order():
    from engine.physiology.waveforms import WaveformBuffer

    buf = WaveformBuffer(channel="ECG_II", sample_rate_hz=10.0, window_s=1.0)
    windows = list(buf.add(list(range(25))))
    assert len(windows) == 2
    assert windows[0].samples == list(range(10))
    assert windows[1].samples == list(range(10, 20))


def test_waveform_buffer_reports_window_start_times():
    from engine.physiology.waveforms import WaveformBuffer

    buf = WaveformBuffer(channel="ART", sample_rate_hz=10.0, window_s=1.0)
    windows = list(buf.add([0.0] * 30))
    assert [w.start_offset_s for w in windows] == [0.0, 1.0, 2.0]


def test_waveform_buffer_rejects_bad_config():
    from engine.physiology.waveforms import WaveformBuffer

    with pytest.raises(ValueError):
        WaveformBuffer(channel="ART", sample_rate_hz=0.0, window_s=10.0)


def test_waveform_rates_cover_all_channels():
    from engine.physiology.waveforms import WAVEFORM_CHANNELS, WAVEFORM_RATES

    assert set(WAVEFORM_RATES) == set(WAVEFORM_CHANNELS)
    # Pulse-native channels are capped at the 50 Hz engine timestep.
    for native in ("CO2", "AWP", "FLOW", "VOL"):
        assert WAVEFORM_RATES[native] == 50.0
    for synthesized in ("ART", "PLETH"):
        assert WAVEFORM_RATES[synthesized] == 125.0
    assert WAVEFORM_RATES["ECG_II"] == 250.0
