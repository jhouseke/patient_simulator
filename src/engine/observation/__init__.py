"""Causal, reproducible observation of physiological truth.

Unconfigured metrics are observed ideally. Missing inputs and measurements that
are not due (or that drop out) are omitted, never filled from stale values.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from random import Random
from typing import Protocol


class ObservationModel(Protocol):
    def observe(self, at: datetime, truth: dict[str, float]) -> dict[str, float]: ...
    def snapshot_state(self) -> object: ...
    def restore_state(self, state: object) -> None: ...


@dataclass(frozen=True)
class MetricObservationConfig:
    """Durations in seconds; noise is a Gaussian standard deviation.

    Averaging is the arithmetic mean of available samples within the trailing
    window ending at the lagged observation time. A zero window selects the
    latest sample at or before that time. No interpolation uses future data.
    """
    noise_std: float = 0.0
    lag_s: float = 0.0
    averaging_s: float = 0.0
    interval_s: float = 0.0
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        for name in ("noise_std", "lag_s", "averaging_s", "interval_s", "dropout_probability"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.dropout_probability > 1:
            raise ValueError("dropout_probability must be at most 1")


class ConfigurableObservationModel:
    """Sample per-metric observations from a stream of timestamped truth.

    Calls must be chronological. Intervals are measured from the last attempted
    measurement, including dropout. At startup, lagged values are unavailable
    until a sufficiently old sample exists. Input dictionaries are never changed.
    """
    def __init__(self, seed: int = 0, metrics: dict[str, MetricObservationConfig] | None = None):
        self.metrics = dict(metrics or {})
        if any(not isinstance(config, MetricObservationConfig) for config in self.metrics.values()):
            raise TypeError("metrics values must be MetricObservationConfig instances")
        self._rng = Random(seed)
        self._buffers: dict[str, list[tuple[datetime, float]]] = {}
        self._last_attempt: dict[str, datetime] = {}
        self._last_values: dict[str, float] = {}
        self._last_time: datetime | None = None

    def observe(self, at: datetime, truth: dict[str, float]) -> dict[str, float]:
        if self._last_time is not None and at < self._last_time:
            raise ValueError("observation times must be chronological")
        for metric, value in truth.items():
            if not isfinite(value):
                raise ValueError(f"truth value for {metric!r} must be finite")
        self._last_time = at
        result: dict[str, float] = {}
        for metric, value in truth.items():
            config = self.metrics.get(metric, MetricObservationConfig())
            buffer = self._buffers.setdefault(metric, [])
            buffer.append((at, value))
            cutoff = at - timedelta(seconds=config.lag_s)
            start = cutoff - timedelta(seconds=config.averaging_s)
            # Keep the last preceding sample for zero-window lookup, and all
            # samples in the averaging window plus lagged future samples.
            keep = 0
            for i, (timestamp, _) in enumerate(buffer):
                if timestamp < start:
                    keep = i
                else:
                    break
            if keep:
                del buffer[:keep]
            previous = self._last_attempt.get(metric)
            if previous is not None and (at - previous).total_seconds() < config.interval_s:
                continue
            self._last_attempt[metric] = at
            if config.dropout_probability and self._rng.random() < config.dropout_probability:
                continue
            if config.averaging_s:
                values = [v for t, v in buffer if start <= t <= cutoff]
            else:
                values = [v for t, v in buffer if t <= cutoff][-1:]
            if not values:
                continue
            observed = sum(values) / len(values)
            if config.noise_std:
                observed += self._rng.gauss(0.0, config.noise_std)
            result[metric] = observed
            self._last_values[metric] = observed
        return result

    def snapshot_state(self) -> object:
        return deepcopy((self._rng.getstate(), self._buffers, self._last_attempt,
                         self._last_values, self._last_time))

    def restore_state(self, state: object) -> None:
        rng, self._buffers, self._last_attempt, self._last_values, self._last_time = deepcopy(state)
        self._rng.setstate(rng)


# Default construction is an ideal, zero-latency observation model.
IdealObservationModel = ConfigurableObservationModel

__all__ = ["ObservationModel", "MetricObservationConfig", "ConfigurableObservationModel", "IdealObservationModel"]
