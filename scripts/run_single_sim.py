"""Run one patient end-to-end and print the vitals trajectory.

Requires a compiled Pulse engine (see docs/pulse_api_notes.md). With --scenario
it runs a full OR->ICU course; otherwise it runs a flat ICU stretch.

    python scripts/run_single_sim.py --hours 1
    python scripts/run_single_sim.py --scenario elective_laparoscopic_cholecystectomy
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone

from engine.pdes.engine import PDES_Engine
from engine.physiology.pulse_adapter import PatientConfig, PulseAdapter
from engine.scenario.library import SCENARIOS, get_scenario
from engine.scenario.machine import ScenarioStateMachine

START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=float, default=1.0, help="duration when no scenario")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), help="run a full OR course")
    p.add_argument("--charting-interval", type=float, default=5.0, help="seconds")
    p.add_argument("--age", type=int, default=65)
    p.add_argument("--sex", default="MALE", choices=["MALE", "FEMALE"])
    p.add_argument("--height-cm", type=float, default=175.0)
    p.add_argument("--weight-kg", type=float, default=80.0)
    p.add_argument("--quiet", action="store_true", help="print only a summary")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = PatientConfig(
        sex=args.sex,
        age_yr=args.age,
        height_cm=args.height_cm,
        weight_kg=args.weight_kg,
    )
    adapter = PulseAdapter(seed=args.seed, patient_cfg=cfg)
    samples: list[tuple[datetime, dict[str, float]]] = []

    def observe(t: datetime, vitals: dict[str, float]) -> None:
        samples.append((t, vitals))
        if not args.quiet:
            print(t.isoformat(), json.dumps({k: round(v, 2) for k, v in vitals.items()}))

    if args.scenario:
        machine = ScenarioStateMachine(
            scenario=get_scenario(args.scenario),
            physiology=adapter,
            store=None,
            start_time=START,
        )
        machine.on_phase_enter = lambda p: print(f"--- {p.value} ---")

        async def _go() -> None:
            engine = PDES_Engine(
                store=None,
                start_time=START,
                end_time=START
                + timedelta(seconds=machine.scenario.total_duration_s),
                physiology=adapter,
                scenario=machine,
                charting_interval_s=args.charting_interval,
            )
            engine.on_vitals_sampled = observe
            await engine.run()

        asyncio.run(_go())
    else:
        engine = PDES_Engine(
            store=None,
            start_time=START,
            end_time=START + timedelta(hours=args.hours),
            physiology=adapter,
            charting_interval_s=args.charting_interval,
        )
        engine.on_vitals_sampled = observe
        asyncio.run(engine.run())

    adapter.close()

    if samples:
        hrs = [v.get("hr", float("nan")) for _, v in samples]
        maps = [v.get("map", float("nan")) for _, v in samples]
        print(
            f"\n{len(samples)} samples | "
            f"HR {min(hrs):.0f}-{max(hrs):.0f} | MAP {min(maps):.0f}-{max(maps):.0f}"
        )


if __name__ == "__main__":
    main()
