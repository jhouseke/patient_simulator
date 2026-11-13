"""Synthetic arterial-line and plethysmograph morphology.

Pulse advances at a fixed 0.02 s timestep, so 50 Hz is the highest rate it can
natively emit — monitor-rate ART (125 Hz) and PLETH (125 Hz) morphology is
generated here from Pulse's beat-level outputs (HR, SBP, DBP, SpO2) rather than
upsampled from a 50 Hz trace (Arch Decision 9).

STATEFUL and seeded, exactly like ECGSynthesizer: beat phase and the noise RNG
carry across generate_* calls so successive windows join without seams. ART and
PLETH share one beat phase because they come off the same heartbeat.

Beat templates are normalized to [0, 1] so systolic peaks land on SBP and
diastolic troughs on DBP by construction. Shapes get calibrated against VitalDB
SNUADC/ART and SNUADC/PLETH in the Phase 2 gate.
"""

from __future__ import annotations

import copy

import numpy as np

_TEMPLATE_N = 2048


def _arterial_template(n: int = _TEMPLATE_N) -> np.ndarray:
    """One cardiac cycle of arterial pressure, normalized to [0, 1].

    Piecewise: sharp systolic upstroke, systolic decline, dicrotic notch and
    bump, then exponential diastolic runoff back to the starting value.
    """
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    t = np.zeros(n)

    # Systolic upstroke: 0 -> 1
    m = u < 0.13
    t[m] = 0.5 * (1.0 - np.cos(np.pi * u[m] / 0.13))

    # Systolic decline: 1 -> 0.45
    m = (u >= 0.13) & (u < 0.38)
    x = (u[m] - 0.13) / 0.25
    t[m] = 0.45 + 0.55 * 0.5 * (1.0 + np.cos(np.pi * x))

    # Dicrotic notch (dip) then dicrotic wave (bump), centered on the valve close
    m = (u >= 0.38) & (u < 0.50)
    x = (u[m] - 0.38) / 0.12
    t[m] = 0.45 - 0.10 * np.sin(np.pi * x) + 0.06 * np.sin(2 * np.pi * x)

    # Diastolic runoff: 0.45 -> 0 (exactly, so beats join continuously)
    m = u >= 0.50
    x = (u[m] - 0.50) / 0.50
    decay = np.exp(-3.0 * x)
    t[m] = 0.45 * (decay - np.exp(-3.0)) / (1.0 - np.exp(-3.0))

    return t


def _pleth_template(n: int = _TEMPLATE_N) -> np.ndarray:
    """Photoplethysmograph contour, normalized to [0, 1].

    Same beat timing as the arterial pulse but smoother, with a later and more
    rounded dicrotic notch — it is a peripheral volume signal, not a pressure.
    """
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    t = np.zeros(n)

    m = u < 0.20
    t[m] = 0.5 * (1.0 - np.cos(np.pi * u[m] / 0.20))

    m = (u >= 0.20) & (u < 0.45)
    x = (u[m] - 0.20) / 0.25
    t[m] = 0.55 + 0.45 * 0.5 * (1.0 + np.cos(np.pi * x))

    m = (u >= 0.45) & (u < 0.60)
    x = (u[m] - 0.45) / 0.15
    t[m] = 0.55 - 0.08 * np.sin(np.pi * x)

    m = u >= 0.60
    x = (u[m] - 0.60) / 0.40
    decay = np.exp(-2.5 * x)
    t[m] = 0.55 * (decay - np.exp(-2.5)) / (1.0 - np.exp(-2.5))

    return t


_ART_TEMPLATE = _arterial_template()
_PLETH_TEMPLATE = _pleth_template()


class PressureSynthesizer:
    def __init__(
        self,
        seed: int,
        sample_rate_hz: float = 125.0,
        art_noise_mmhg: float = 0.3,
        pleth_noise: float = 0.004,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.fs = sample_rate_hz
        self.rng = np.random.default_rng(seed)
        self.art_noise_mmhg = art_noise_mmhg
        self.pleth_noise = pleth_noise
        # Beat phase in cycles [0, 1). Shared by ART and PLETH.
        self.beat_phase = 0.0

    def snapshot_state(self) -> dict:
        return {"beat_phase": self.beat_phase, "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def restore_state(self, state: dict) -> None:
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.beat_phase = float(state["beat_phase"])

    def _advance_phase(self, n: int, heart_rate_bpm) -> np.ndarray:
        """Return per-sample beat phase in cycles and carry the state forward."""
        hr = np.broadcast_to(np.asarray(heart_rate_bpm, dtype=float), (n,))
        dphi = np.maximum(hr, 1.0) / 60.0 / self.fs
        phase = self.beat_phase + np.cumsum(dphi)
        self.beat_phase = float(phase[-1] % 1.0)
        return phase % 1.0

    @staticmethod
    def _sample_template(template: np.ndarray, phase_cycles: np.ndarray) -> np.ndarray:
        """Periodic linear interpolation into a beat template.

        Interpolated rather than nearest-index: nearest-index turns a 1e-16
        phase difference at a window boundary into a whole template step, which
        breaks seamlessness across generate() calls (and adds quantization
        steps to the signal).
        """
        n = template.size
        x = phase_cycles * n
        floor = np.floor(x)
        i0 = floor.astype(np.intp) % n
        i1 = (i0 + 1) % n
        frac = x - floor
        return template[i0] * (1.0 - frac) + template[i1] * frac

    def generate_art(
        self,
        duration_s: float,
        heart_rate_bpm,
        sbp_mmhg,
        dbp_mmhg,
        *, phase_cycles=None,
    ) -> np.ndarray:
        """Arterial pressure in mmHg; peaks track SBP, troughs track DBP."""
        n = len(phase_cycles) if phase_cycles is not None else int(round(duration_s * self.fs))
        if n <= 0:
            return np.zeros(0)
        phase = (self._advance_phase(n, heart_rate_bpm) if phase_cycles is None
                 else np.asarray(phase_cycles, dtype=float) % 1.0)
        shape = self._sample_template(_ART_TEMPLATE, phase)
        sbp = np.broadcast_to(np.asarray(sbp_mmhg, dtype=float), (n,))
        dbp = np.broadcast_to(np.asarray(dbp_mmhg, dtype=float), (n,))
        noise = self.rng.normal(0.0, self.art_noise_mmhg, n)
        return dbp + (sbp - dbp) * shape + noise

    def generate_pleth(
        self,
        duration_s: float,
        heart_rate_bpm,
        spo2_pct=98.0,
        perfusion: float = 1.0,
        *, phase_cycles=None,
    ) -> np.ndarray:
        """Normalized [0, 1] pleth. Amplitude scales with perfusion."""
        n = len(phase_cycles) if phase_cycles is not None else int(round(duration_s * self.fs))
        if n <= 0:
            return np.zeros(0)
        phase = (self._advance_phase(n, heart_rate_bpm) if phase_cycles is None
                 else np.asarray(phase_cycles, dtype=float) % 1.0)
        shape = self._sample_template(_PLETH_TEMPLATE, phase)
        amp = float(np.clip(perfusion, 0.05, 1.0))
        noise = self.rng.normal(0.0, self.pleth_noise, n)
        # Desaturation dims the pulsatile amplitude a little, as real probes show.
        sat_scale = np.clip(np.asarray(spo2_pct, dtype=float) / 100.0, 0.5, 1.0)
        sig = 0.5 * (1.0 - amp * sat_scale) + amp * sat_scale * shape + noise
        return np.clip(sig, 0.0, 1.0)
