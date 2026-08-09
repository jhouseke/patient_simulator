"""Window accumulation for high-frequency waveform channels.

Waveforms are stored as fixed-duration windows rather than one row per sample:
a 125 Hz arterial line over a 6-hour case is 2.7M samples/channel, so
row-per-sample is untenable (Arch Decision 4).

Channel rates reflect where each signal comes from (Arch Decision 9):
Pulse-native channels are capped at the engine's 50 Hz timestep, while
ART/PLETH/ECG_II are synthesized at true monitor rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

# channel -> (sample_rate_hz, unit, source)
WAVEFORM_CHANNELS: dict[str, tuple[float, str, str]] = {
    "ART": (125.0, "mmHg", "synthesized"),
    "PLETH": (125.0, "unitless", "synthesized"),
    "ECG_II": (250.0, "mV", "synthesized"),
    "CO2": (50.0, "mmHg", "pulse"),
    "AWP": (50.0, "cmH2O", "pulse"),
    "FLOW": (50.0, "mL/s", "pulse"),
    "VOL": (50.0, "mL", "pulse"),
}

WAVEFORM_RATES: dict[str, float] = {c: v[0] for c, v in WAVEFORM_CHANNELS.items()}
WAVEFORM_UNITS: dict[str, str] = {c: v[1] for c, v in WAVEFORM_CHANNELS.items()}

DEFAULT_WINDOW_S = 10.0


@dataclass
class WaveformWindow:
    channel: str
    sample_rate_hz: float
    duration_s: float
    samples: list[float]
    start_offset_s: float = 0.0
    """Seconds from the buffer's first sample to this window's first sample.

    The engine adds this to the barrier time to get the window's wall-clock
    start, so a barrier that produces several windows timestamps them apart
    instead of collapsing them onto one instant.
    """


class WaveformBuffer:
    """Accumulates samples for one channel, emitting fixed-duration windows."""

    def __init__(
        self,
        channel: str,
        sample_rate_hz: float,
        window_s: float = DEFAULT_WINDOW_S,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.channel = channel
        self.fs = sample_rate_hz
        self.window_s = window_s
        self.window_n = int(round(sample_rate_hz * window_s))
        if self.window_n <= 0:
            raise ValueError("window must contain at least one sample")
        self._buf: list[float] = []
        self._emitted = 0

    def add(self, samples: Sequence[float]) -> Iterator[WaveformWindow]:
        """Buffer samples, yielding each complete window as it fills."""
        self._buf.extend(samples)
        while len(self._buf) >= self.window_n:
            chunk = self._buf[: self.window_n]
            self._buf = self._buf[self.window_n :]
            yield WaveformWindow(
                channel=self.channel,
                sample_rate_hz=self.fs,
                duration_s=self.window_s,
                samples=chunk,
                start_offset_s=self._emitted * self.window_s,
            )
            self._emitted += 1

    @property
    def pending(self) -> int:
        """Samples buffered but not yet part of a complete window."""
        return len(self._buf)
