# Clinical fidelity challenges

`engine.validation.challenges` defines deterministic acceptance checks over supplied timestamped simulator truth, observations, and optionally a paired baseline trajectory. It does not invoke a model or Pulse. All examples and numbers below are invented illustrations, require clinician review and approval, and are not treatment guidelines.

A `Challenge` records a name, `draft` or `approved` status, reviewer, review date, provenance, and uniquely named metrics. An approved definition requires a named reviewer and review date; provenance should identify the clinical evidence, version, population, units, and rationale for the thresholds. A draft can produce a numerical pass, but `clinical_approval` is always false. Approval also requires every metric to pass. This metadata is a declaration, not an authentication or review workflow.

Each `Metric` specifies a channel, inclusive window relative to the intervention time in seconds, inclusive numerical bounds, and expected direction (`increase`, `decrease`, or `either`). Increase and decrease are strict: zero satisfies neither. Define onset and recovery windows explicitly using response and recovery metrics; these metrics compare the window's mean truth value to the mean in a specified baseline window. They assess average change within the window. Neither metric measures threshold crossing time, time-to-onset, or sustained recovery; a passing onset-window mean does not establish when a response began. Choose sampling and windows appropriate to the clinical question.

A consistency metric compares a truth channel with a named observation channel, reporting the maximum absolute difference across exactly matching timestamps in its window. A paired metric compares intervention truth against a separately supplied baseline truth trajectory, reporting the mean signed intervention-minus-baseline difference at matching timestamps. These comparisons require identical nonempty timestamp sets within the window; there is no interpolation or implicit clock alignment. Units must match, and both trajectories must share the intervention clock. Each `Window` optionally sets `min_samples` (a positive integer, default 1) and `max_gap` (positive finite seconds, default unset). The maximum gap includes the distance from each window boundary to the nearest sample as well as gaps between consecutive samples; equality is allowed. Requirements apply independently to the response and baseline windows and to each trace in paired or consistency comparisons. Insufficient coverage yields `not_evaluable`. With the defaults, one sample is sufficient and temporal coverage is not established. Even configured coverage does not prove behavior between samples or establish crossing times.

All supplied traces must have finite values and finite, strictly increasing, unique timestamps. Invalid traces or definitions raise `ValueError`. Missing samples, unmatched timestamps, and non-finite derived results yield `not_evaluable`, never a pass. Overall outcome is `fail` if any metric fails, otherwise `not_evaluable` if any metric cannot be evaluated, otherwise `pass`. Individual results remain available even when the overall outcome fails.

For example, `ILLUSTRATIVE_CHALLENGE` contains a draft response in seconds 1–5, recovery in seconds 10–15, synchronized channel agreement, and a paired effect. With compatible units and timestamps it can be evaluated as follows:

```python
from engine.validation.challenges import ILLUSTRATIVE_CHALLENGE, evaluate

result = evaluate(
    ILLUSTRATIVE_CHALLENGE,
    truth={"signal": [(-5, 10), (0, 10), (1, 13), (5, 13), (10, 10), (15, 10)]},
    observations={"display": [(1, 13), (5, 13)]},
    intervention_time=0,
    baseline_truth={"signal": [(1, 10), (5, 10)]},
)
assert result.outcome == "pass"
assert not result.clinical_approval  # illustrative definition remains draft
```

Before clinical use, a qualified clinician must review and approve the actual definition, evidence, population, units, thresholds, temporal windows, sampling adequacy, and interpretation. Version and retain that approved definition alongside its results.

## Evaluate an exported trajectory

The CLI consumes the JSON written by `scripts/run_single_sim.py --output`: `truth` and `observations` are arrays of `{"time_s": number, "values": {"channel": number}}` records. Record times must be strictly increasing within each array. Missing channel values are omitted, not filled in. Supply an explicit challenge JSON file; there is no default clinical rule. For example, this invented, unapproved definition checks a mean change:

```json
{
  "name": "illustrative response",
  "status": "draft",
  "reviewer": null,
  "review_date": null,
  "provenance": "Invented example; requires clinician approval",
  "metrics": [{
    "name": "response",
    "kind": "response",
    "channel": "signal",
    "window": {"start": 1, "end": 5, "min_samples": 2, "max_gap": 3},
    "baseline": {"start": -5, "end": 0, "min_samples": 2, "max_gap": 5},
    "direction": "increase",
    "lower": 2,
    "upper": 10
  }]
}
```

Save this as `challenge.json`, then run:

```sh
python scripts/evaluate_fidelity.py --challenge challenge.json --trajectory trajectory.json \
  --intervention-time 30 --baseline baseline.json --output result.json
```

`--baseline` is optional and supplies the `truth` records for paired metrics, using the same clock; omit it when no paired trajectory is available. `--output` is optional; without it, result JSON goes to standard output. Review dates use ISO format (`YYYY-MM-DD`). Result JSON includes the overall outcome, `clinical_approval`, and each metric's outcome, value, and reason. Exit status is 0 for pass, 1 for fail, and 2 for not evaluable; invalid input or an output error also exits 2 with an error on standard error. A draft can pass numerically but cannot produce clinical approval.
