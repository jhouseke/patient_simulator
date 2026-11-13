from dataclasses import replace
from datetime import date

import pytest

from engine.validation.challenges import (
    Challenge, ILLUSTRATIVE_CHALLENGE, Metric, Window, evaluate,
)


def data():
    truth = {"signal": [(-5, 10), (0, 10), (1, 13), (5, 13), (10, 10), (15, 10)]}
    observations = {"display": [(1, 13.5), (5, 12.5)]}
    baseline = {"signal": [(1, 10), (5, 10)]}
    return truth, observations, baseline


def run(challenge=ILLUSTRATIVE_CHALLENGE, truth=None, observations=None, baseline=None):
    t, o, b = data()
    return evaluate(challenge, t if truth is None else truth, o if observations is None else observations,
                    0, baseline_truth=b if baseline is None else baseline)


def test_boundary_windows_and_draft_gate():
    result = run()
    assert result.outcome == "pass"
    assert [m.value for m in result.metrics] == [3, 0, 0.5, 3]
    assert not result.clinical_approval


def test_approved_requires_review_and_all_metrics_pass():
    with pytest.raises(ValueError):
        replace(ILLUSTRATIVE_CHALLENGE, status="approved")
    approved = replace(ILLUSTRATIVE_CHALLENGE, status="approved", reviewer="Clinician", review_date=date(2026, 1, 1))
    assert run(approved).clinical_approval
    assert not run(approved, observations={}).clinical_approval
    bad = {"display": [(1, 20), (5, 20)]}
    result = run(approved, observations=bad)
    assert result.outcome == "fail"
    assert not result.clinical_approval


def test_missing_and_unmatched_data_cannot_pass():
    assert run(truth={}).outcome == "not_evaluable"
    assert run(observations={}).metrics[2].outcome == "not_evaluable"
    assert run(observations={"display": [(1, 13)]}).metrics[2].outcome == "not_evaluable"
    assert run(baseline={}).metrics[3].outcome == "not_evaluable"
    assert run(baseline={"signal": [(1, 10)]}).metrics[3].outcome == "not_evaluable"


@pytest.mark.parametrize("trace", [[(1, 1), (1, 2)], [(2, 1), (1, 2)], [(float('nan'), 1)], [(1, float('inf'))]])
def test_invalid_traces_rejected(trace):
    with pytest.raises(ValueError):
        run(truth={"signal": trace})


def test_invalid_event_and_windows_rejected():
    with pytest.raises(ValueError):
        evaluate(ILLUSTRATIVE_CHALLENGE, {}, {}, float("inf"))
    with pytest.raises(ValueError):
        Window(2, 1)
    with pytest.raises(ValueError):
        Window(0, float("nan"))


def test_direction_and_bounds_are_both_enforced():
    metric = Metric("change", "response", "signal", Window(1, 1), -5, 5,
                    direction="increase", baseline=Window(0, 0))
    challenge = Challenge("direction", "draft", None, None, "example", (metric,))
    result = evaluate(challenge, {"signal": [(0, 10), (1, 9)]}, {}, 0)
    assert result.outcome == "fail"


def test_failure_takes_precedence_over_missing_metric():
    result = run(observations={"display": [(1, 20), (5, 20)]}, baseline={})
    assert result.outcome == "fail"
    assert result.metrics[-1].outcome == "not_evaluable"


def test_coverage_applies_to_response_and_baseline_windows():
    metric = Metric("response", "response", "signal", Window(1, 5, min_samples=2, max_gap=2),
                    1, 5, baseline=Window(-2, 0, min_samples=2, max_gap=2))
    challenge = Challenge("coverage", "approved", "Clinician", date(2026, 1, 1), "example", (metric,))
    def check(trace):
        return evaluate(challenge, {"signal": trace}, {}, 0)

    assert check([(-2, 10), (0, 10), (1, 12), (3, 12), (5, 12)]).clinical_approval
    for trace in (
        [(-2, 10), (0, 10), (3, 12)],  # one response sample
        [(0, 10), (1, 12), (3, 12), (5, 12)],  # one baseline sample
        [(-2, 10), (0, 10), (1, 12), (5, 12)],  # interior gap
        [(-2, 10), (0, 10), (4, 12), (5, 12)],  # gap from window start
    ):
        result = check(trace)
        assert result.outcome == "not_evaluable"
        assert not result.clinical_approval
        assert result.metrics[0].reason == "insufficient sample coverage"


@pytest.mark.parametrize("kind", ["consistency", "paired"])
def test_coverage_applies_to_both_synchronized_traces(kind):
    metric = Metric("comparison", kind, "signal", Window(0, 4, min_samples=2, max_gap=2),
                    -1, 1, other_channel="display" if kind == "consistency" else None)
    challenge = Challenge("comparison", "draft", None, None, "example", (metric,))
    truth = {"signal": [(0, 2), (2, 2), (4, 2)]}
    other = [(0, 2), (2, 2), (4, 2)]
    def check(samples):
        return evaluate(challenge, truth, {"display": samples}, 0,
                        baseline_truth={"signal": samples})
    assert check(other).outcome == "pass"
    assert check([(0, 2), (4, 2)]).outcome == "not_evaluable"
    assert check([(0, 2), (2, 2)]).outcome == "not_evaluable"  # unmatched timestamps


@pytest.mark.parametrize("kwargs", [
    {"min_samples": 0}, {"min_samples": 1.5}, {"min_samples": True},
    {"max_gap": 0}, {"max_gap": -1}, {"max_gap": float("inf")}, {"max_gap": float("nan")},
])
def test_invalid_coverage_requirements(kwargs):
    with pytest.raises(ValueError):
        Window(0, 1, **kwargs)
