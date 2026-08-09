"""Synthetic Lead-II ECG via a sum-of-Gaussians PQRST model.

Driven by an instantaneous heart rate (scalar, or a per-sample series from
Pulse). Pulse emits only a Lead III potential at its 50 Hz engine timestep, so
monitor-rate morphology is synthesized here (Arch Decision 9).

STATEFUL: beat phase, baseline-wander time, and the noise RNG all carry across
generate() calls so successive waveform windows are seamless. Seeded for
reproducibility. Calibrated against VitalDB ECG_II spectra in the Phase 2 gate.
"""

from __future__ import annotations

import numpy as np

# (center_angle_rad, gaussian_width, amplitude_mV) for P,Q,R,S,T
_WAVES = [
    (-np.pi / 3, 0.25, 0.08),    # P
    (-np.pi / 12, 0.10, -0.12),  # Q
    (0.0, 0.10, 1.10),           # R
    (np.pi / 12, 0.10, -0.25),   # S
    (np.pi / 2, 0.40, 0.30),     # T
]


class ECGSynthesizer:
    def __init__(
        self,
        seed: int,
        sample_rate_hz: float = 250.0,
        noise_mv: float = 0.02,
        wander_mv: float = 0.05,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.fs = sample_rate_hz
        self.rng = np.random.default_rng(seed)
        self.noise_mv = noise_mv
        self.wander_mv = wander_mv
        self._t = 0.0          # continuous time across calls (wander phase)
        self._phase = -np.pi   # beat phase, carried across calls (no seams)

    def generate(self, duration_s: float, heart_rate_bpm) -> np.ndarray:
        """Return `duration_s` of Lead-II signal in mV.

        heart_rate_bpm may be a scalar or a per-sample array — pass Pulse's
        instantaneous HR series to get beat-to-beat variability.
        """
        n = int(round(duration_s * self.fs))
        if n <= 0:
            return np.zeros(0)

        hr = np.broadcast_to(np.asarray(heart_rate_bpm, dtype=float), (n,))
        # Integrate angular velocity so a varying HR bends the beat phase.
        dphi = 2 * np.pi * np.maximum(hr, 1.0) / 60.0 / self.fs
        phase = self._phase + np.cumsum(dphi)
        self._phase = float(np.angle(np.exp(1j * phase[-1])))

        wrapped = np.angle(np.exp(1j * phase))
        sig = np.zeros(n)
        for center, width, amp in _WAVES:
            d = np.angle(np.exp(1j * (wrapped - center)))  # wrapped distance
            sig += amp * np.exp(-(d**2) / (2 * width**2))

        t = self._t + np.arange(n) / self.fs
        self._t = float(t[-1] + 1.0 / self.fs)
        wander = self.wander_mv * np.sin(2 * np.pi * 0.15 * t)
        noise = self.rng.normal(0.0, self.noise_mv, n)
        return sig + wander + noise

    def detect_r_peaks(self, sig: np.ndarray) -> np.ndarray:
        """Rising-edge threshold detector — used by tests and QC, not by Pulse."""
        if sig.size == 0:
            return np.zeros(0, dtype=int)
        thr = 0.6 * np.max(sig)
        above = sig > thr
        return np.where((~above[:-1]) & (above[1:]))[0]
