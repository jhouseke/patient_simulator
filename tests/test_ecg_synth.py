"""Synthetic ECG morphology (Task 2.2).

Pulse emits only a 50 Hz Lead III; monitor-rate Lead II morphology is
synthesized (Arch Decision 9). These generators are seeded AND stateful so
successive waveform windows join without seams.
"""

import numpy as np

from engine.physiology.ecg_synth import ECGSynthesizer


def test_ecg_has_expected_rate():
    ecg = ECGSynthesizer(seed=1, sample_rate_hz=250.0)
    sig = ecg.generate(duration_s=10.0, heart_rate_bpm=60.0)
    assert len(sig) == 2500
    # ~1 R-peak per second at 60 bpm => ~10 peaks in 10s
    peaks = ecg.detect_r_peaks(sig)
    assert 8 <= len(peaks) <= 12, f"expected ~10 R-peaks, got {len(peaks)}"


def test_ecg_rate_tracks_heart_rate():
    ecg = ECGSynthesizer(seed=1, sample_rate_hz=250.0)
    fast = ecg.generate(duration_s=10.0, heart_rate_bpm=120.0)
    assert 18 <= len(ecg.detect_r_peaks(fast)) <= 22


def test_ecg_is_deterministic():
    a = ECGSynthesizer(seed=7, sample_rate_hz=250.0).generate(5.0, 75.0)
    b = ECGSynthesizer(seed=7, sample_rate_hz=250.0).generate(5.0, 75.0)
    assert np.allclose(a, b)


def test_different_seeds_differ():
    a = ECGSynthesizer(seed=1, sample_rate_hz=250.0).generate(5.0, 75.0)
    b = ECGSynthesizer(seed=2, sample_rate_hz=250.0).generate(5.0, 75.0)
    assert not np.allclose(a, b)


def test_ecg_is_continuous_across_windows():
    ecg = ECGSynthesizer(seed=3, sample_rate_hz=250.0)
    joined = np.concatenate([ecg.generate(10.0, 72.0), ecg.generate(10.0, 72.0)])
    whole = ECGSynthesizer(seed=3, sample_rate_hz=250.0).generate(20.0, 72.0)
    assert np.allclose(joined, whole), "stateful generator must not seam at window edges"


def test_ecg_accepts_instantaneous_hr_series():
    ecg = ECGSynthesizer(seed=5, sample_rate_hz=250.0)
    hr = np.linspace(60.0, 90.0, 2500)  # HR ramp over 10s => beat-to-beat variability
    sig = ecg.generate(10.0, hr)
    assert len(sig) == 2500
    peaks = ecg.detect_r_peaks(sig)
    # Accelerating rate => later RR intervals shorter than earlier ones
    rr = np.diff(peaks)
    assert rr[-1] < rr[0]


def test_ecg_amplitude_is_physiologic():
    sig = ECGSynthesizer(seed=1, sample_rate_hz=250.0).generate(10.0, 70.0)
    assert 0.5 <= sig.max() <= 2.0, "R wave should be ~1 mV"
    assert sig.min() >= -1.0
