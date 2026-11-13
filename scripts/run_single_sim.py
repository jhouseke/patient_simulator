"""Run one Pulse patient and export separate truth and observed trajectories.

Requires a compiled Pulse engine; see docs/pulse_api_notes.md. Monitor settings
are JSON metric -> MetricObservationConfig keyword arguments. Example:
{"hr": {"lag_s": 5}, "map": {"interval_s": 60}}.
"""

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from engine.observation import ConfigurableObservationModel, MetricObservationConfig
from engine.pdes.engine import PDES_Engine
from engine.physiology.phenotypes import sample_patient
from engine.physiology.pulse_adapter import PatientConfig, PulseAdapter
from engine.physiology.waveforms import WAVEFORM_CHANNELS
from engine.scenario.library import SCENARIOS, get_scenario
from engine.scenario.machine import ScenarioStateMachine

START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=float, default=1.0, help="duration when no scenario")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--scenario", choices=sorted(SCENARIOS))
    p.add_argument("--charting-interval", type=float, default=5.0, help="seconds")
    p.add_argument("--phenotype", choices=["adult", "older_adult", "larger_adult"],
                   help="illustrative correlated profile; overrides demographic arguments")
    p.add_argument("--age", type=int, default=65)
    p.add_argument("--sex", default="MALE", choices=["MALE", "FEMALE"])
    p.add_argument("--height-cm", type=float, default=175.0)
    p.add_argument("--weight-kg", type=float, default=80.0)
    p.add_argument("--observation-config", type=Path, help="JSON monitor configuration")
    p.add_argument("--waveforms", nargs="*", choices=sorted(WAVEFORM_CHANNELS), default=[])
    p.add_argument("--output", type=Path, help="write reviewable JSON truth/observations/waveforms")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sampled = sample_patient(args.phenotype, args.seed) if args.phenotype else None
    cfg = sampled.patient if sampled else PatientConfig(
        sex=args.sex, age_yr=args.age, height_cm=args.height_cm, weight_kg=args.weight_kg)
    settings = json.loads(args.observation_config.read_text()) if args.observation_config else {}
    observation = ConfigurableObservationModel(seed=args.seed,
        metrics={code: MetricObservationConfig(**config) for code, config in settings.items()})
    adapter = PulseAdapter(seed=args.seed, patient_cfg=cfg, waveform_channels=tuple(args.waveforms))
    record = {"format_version": 1, "seed": args.seed, "patient": asdict(cfg),
              "phenotype": asdict(sampled) if sampled else None,
              "observation_config": settings, "scenario": args.scenario,
              "clinical_validation": "pending", "truth": [], "observations": [], "waveforms": []}
    machine = (ScenarioStateMachine(get_scenario(args.scenario), adapter, None, start_time=START)
               if args.scenario else None)
    duration = machine.scenario.total_duration_s if machine else args.hours * 3600
    if duration <= 0:
        raise ValueError("duration must be positive")
    try:
        engine = PDES_Engine(None, START, START + timedelta(seconds=duration),
            physiology=adapter, scenario=machine, observation_model=observation,
            charting_interval_s=args.charting_interval)
        def collect(kind, at, values):
            record[kind].append({"time_s": (at - START).total_seconds(), "values": values})
        engine.on_truth_sampled = lambda at, values: collect("truth", at, values)
        def observed(at, values):
            collect("observations", at, values)
            if not args.quiet:
                print(at.isoformat(), json.dumps(values))
        engine.on_vitals_sampled = observed
        def waveforms(at, values):
            collect("waveforms", at, values)
            record["waveforms"][-1]["channel_start_time_s"] = dict(adapter.last_waveform_start_s)
        engine.on_waveforms_sampled = waveforms
        asyncio.run(engine.run())
    finally:
        adapter.close()
    if args.output:
        record["waveform_channels"] = {c: WAVEFORM_CHANNELS[c] for c in args.waveforms}
        args.output.write_text(json.dumps(record, allow_nan=False, indent=2) + "\n")
    print(f'{len(record["observations"])} observed samples; {len(record["truth"])} truth samples')


if __name__ == "__main__":
    main()
