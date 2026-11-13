"""Evaluate an explicitly supplied challenge against exported trajectories."""

import argparse
from dataclasses import asdict
from datetime import date
import json
from math import isfinite
from pathlib import Path
import sys

# Support running directly from a source checkout without an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engine.validation.challenges import Challenge, Metric, Window, evaluate


def _read_json(path):
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    with path.open(encoding="utf-8") as source:
        return json.load(source, parse_constant=invalid_constant)


def _challenge(raw):
    data = dict(raw)
    if data.get("review_date") is not None:
        data["review_date"] = date.fromisoformat(data["review_date"])
    metrics = []
    for item in data.pop("metrics"):
        fields = dict(item)
        fields["window"] = Window(**fields["window"])
        if fields.get("baseline") is not None:
            fields["baseline"] = Window(**fields["baseline"])
        metrics.append(Metric(**fields))
    return Challenge(metrics=tuple(metrics), **data)


def _traces(records):
    traces = {}
    previous = None
    for record in records:
        time = record["time_s"]
        if not isfinite(time) or (previous is not None and time <= previous):
            raise ValueError("record timestamps must be finite and strictly increasing")
        previous = time
        for channel, value in record["values"].items():
            traces.setdefault(channel, []).append((time, value))
    return traces


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--intervention-time", type=float, required=True, help="seconds on the trajectory clock")
    parser.add_argument("--baseline", type=Path, help="paired trajectory JSON; uses its truth records")
    parser.add_argument("--output", type=Path, help="write result JSON instead of stdout")
    args = parser.parse_args(argv)
    try:
        challenge = _challenge(_read_json(args.challenge))
        trajectory = _read_json(args.trajectory)
        baseline = _traces(_read_json(args.baseline)["truth"]) if args.baseline else None
        result = evaluate(challenge, _traces(trajectory["truth"]), _traces(trajectory["observations"]),
                          args.intervention_time, baseline_truth=baseline)
        output = json.dumps(asdict(result), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return {"pass": 0, "fail": 1, "not_evaluable": 2}[result.outcome]


if __name__ == "__main__":
    sys.exit(main())
