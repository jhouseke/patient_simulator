"""Deterministic, clinician-reviewable checks over supplied time series.

Times are seconds on a common clock. Values and bounds use the channel's
units. This module does not supply clinical thresholds or infer missing data.
"""

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Literal, Mapping, Sequence

Status = Literal["draft", "approved"]
Outcome = Literal["pass", "fail", "not_evaluable"]
Direction = Literal["increase", "decrease", "either"]
Kind = Literal["response", "recovery", "consistency", "paired"]
Trace = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class Window:
    """Inclusive seconds relative to the intervention time."""
    start: float
    end: float
    min_samples: int = 1
    max_gap: float | None = None

    def __post_init__(self):
        if not (isfinite(self.start) and isfinite(self.end) and self.start <= self.end):
            raise ValueError("window must have finite, ordered endpoints")
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if self.max_gap is not None and (not isfinite(self.max_gap) or self.max_gap <= 0):
            raise ValueError("max_gap must be finite and positive")


@dataclass(frozen=True)
class Metric:
    name: str
    kind: Kind
    channel: str
    window: Window
    lower: float
    upper: float
    direction: Direction = "either"
    baseline: Window | None = None
    other_channel: str | None = None

    def __post_init__(self):
        if not self.name or not self.channel:
            raise ValueError("metric name and channel are required")
        if self.kind not in ("response", "recovery", "consistency", "paired"):
            raise ValueError("unknown metric kind")
        if self.direction not in ("increase", "decrease", "either"):
            raise ValueError("unknown direction")
        if not (isfinite(self.lower) and isfinite(self.upper) and self.lower <= self.upper):
            raise ValueError("bounds must be finite and ordered")
        if self.kind in ("response", "recovery") and self.baseline is None:
            raise ValueError("response and recovery require a baseline window")
        if self.kind == "consistency" and not self.other_channel:
            raise ValueError("consistency requires another channel")


@dataclass(frozen=True)
class Challenge:
    name: str
    status: Status
    reviewer: str | None
    review_date: date | None
    provenance: str
    metrics: tuple[Metric, ...]

    def __post_init__(self):
        if not self.name or self.status not in ("draft", "approved") or not self.provenance.strip():
            raise ValueError("name, valid status, and provenance are required")
        if not self.metrics or len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("metrics must be nonempty and uniquely named")
        if self.review_date is not None and not isinstance(self.review_date, date):
            raise ValueError("review_date must be a date")
        if self.status == "approved" and (not self.reviewer or not self.reviewer.strip() or self.review_date is None):
            raise ValueError("approved challenges require reviewer and review date")


@dataclass(frozen=True)
class MetricResult:
    name: str
    outcome: Outcome
    value: float | None
    reason: str


@dataclass(frozen=True)
class ChallengeResult:
    outcome: Outcome
    clinical_approval: bool
    metrics: tuple[MetricResult, ...]


def _validate(traces: Mapping[str, Trace]) -> None:
    for name, trace in traces.items():
        previous = None
        for timestamp, value in trace:
            if not (isfinite(timestamp) and isfinite(value)):
                raise ValueError(f"{name}: timestamps and values must be finite")
            if previous is not None and timestamp <= previous:
                raise ValueError(f"{name}: timestamps must be strictly increasing")
            previous = timestamp


def _samples(traces: Mapping[str, Trace], channel: str, window: Window, event: float) -> dict[float, float]:
    return {t: value for t, value in traces.get(channel, ()) if event + window.start <= t <= event + window.end}


def _covered(samples: Mapping[float, float], window: Window, event: float) -> bool:
    if len(samples) < window.min_samples:
        return False
    if window.max_gap is None:
        return True
    times = [event + window.start, *samples, event + window.end]
    return all(right - left <= window.max_gap for left, right in zip(times, times[1:]))


def evaluate(challenge: Challenge, truth: Mapping[str, Trace], observations: Mapping[str, Trace],
             intervention_time: float, *, baseline_truth: Mapping[str, Trace] | None = None) -> ChallengeResult:
    """Evaluate intervention truth and observed traces; paired truth shares the same clock.

    Response/recovery compare mean truth in the named window with mean truth in
    the baseline window. Consistency checks the largest absolute truth/observation
    difference at exactly matching timestamps. Paired checks the mean signed
    intervention-minus-baseline difference at exactly matching timestamps.
    """
    if not isfinite(intervention_time):
        raise ValueError("intervention_time must be finite")
    _validate(truth)
    _validate(observations)
    if baseline_truth is not None:
        _validate(baseline_truth)
    results = []
    for metric in challenge.metrics:
        current = _samples(truth, metric.channel, metric.window, intervention_time)
        value = None
        covered = _covered(current, metric.window, intervention_time)
        if metric.kind in ("response", "recovery"):
            base = _samples(truth, metric.channel, metric.baseline, intervention_time)
            covered = covered and _covered(base, metric.baseline, intervention_time)
            if covered:
                value = sum(current.values()) / len(current) - sum(base.values()) / len(base)
        elif metric.kind == "consistency":
            other = _samples(observations, metric.other_channel, metric.window, intervention_time)
            # Every supplied sample must have a matching counterpart; no interpolation.
            covered = covered and _covered(other, metric.window, intervention_time)
            if covered and current.keys() == other.keys():
                value = max(abs(current[t] - other[t]) for t in current)
        else:
            other = _samples(baseline_truth or {}, metric.channel, metric.window, intervention_time)
            covered = covered and _covered(other, metric.window, intervention_time)
            if covered and current.keys() == other.keys():
                value = sum(current[t] - other[t] for t in current) / len(current)
        if value is None or not isfinite(value):
            reason = "insufficient sample coverage" if not covered else "unmatched or non-finite derived data"
            results.append(MetricResult(metric.name, "not_evaluable", None, reason))
            continue
        direction_ok = (metric.direction == "either" or
                        metric.direction == "increase" and value > 0 or
                        metric.direction == "decrease" and value < 0)
        passed = direction_ok and metric.lower <= value <= metric.upper
        results.append(MetricResult(metric.name, "pass" if passed else "fail", value,
                                    "within bounds and direction" if passed else "outside bounds or direction"))
    outcomes = {result.outcome for result in results}
    outcome = "fail" if "fail" in outcomes else "not_evaluable" if "not_evaluable" in outcomes else "pass"
    return ChallengeResult(outcome, challenge.status == "approved" and outcome == "pass", tuple(results))


# Illustrative only: these values are not clinical guidance and require review.
ILLUSTRATIVE_CHALLENGE = Challenge(
    name="illustrative time-local response and recovery",
    status="draft", reviewer=None, review_date=None,
    provenance="Invented example values; requires clinician approval before clinical use.",
    metrics=(
        Metric("onset", "response", "signal", Window(1, 5), 2, 10, "increase", Window(-5, 0)),
        Metric("recovery", "recovery", "signal", Window(10, 15), -1, 1, baseline=Window(-5, 0)),
        Metric("channel agreement", "consistency", "signal", Window(1, 5), 0, 0.5, other_channel="display"),
        Metric("paired intervention effect", "paired", "signal", Window(1, 5), 2, 10, "increase"),
    ),
)
