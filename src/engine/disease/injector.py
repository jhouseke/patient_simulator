"""
Disease Injection Protocol — structured trajectory dataclasses.

When the Constraint Validator approves an ADMINISTER_MED intent,
the engine writes:
  1. The medication event to TimescaleDB (canonical numerical state)
  2. The clinical relationship to Apache AGE (knowledge graph)

This module manages disease trajectories that alter patient physiology
between time barriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import numpy as np


@dataclass
class DiseaseTrajectory:
    """Base class for disease trajectories."""
    onset_time: datetime
    severity: float  # 0-1

    def vital_delta_at(self, sim_time: datetime) -> dict[str, float]:
        """Return vital sign deltas at a given simulation time."""
        raise NotImplementedError


@dataclass
class OccultHemorrhage(DiseaseTrajectory):
    """
    Occult hemorrhage trajectory.

    Produces:
      - Decreasing MAP (blood loss)
      - Increasing HR (compensatory tachycardia)
      - Decreasing CVP (reduced preload)
      - Decreasing SPO2 (if severe, tissue hypoperfusion)
    """
    rate_mmhg_per_min: float = 2.0   # MAP decline rate
    max_hr_bpm: float = 140.0        # Maximum compensatory HR

    def vital_delta_at(self, sim_time: datetime) -> dict[str, float]:
        minutes = max(0, (sim_time - self.onset_time).total_seconds() / 60.0)
        # MAP drops linearly then plateaus
        map_drop = min(self.rate_mmhg_per_min * minutes * self.severity, 40.0)
        # HR rises as compensation
        hr_rise = min(0.8 * map_drop, self.max_hr_bpm - 70.0)
        # CVP drops with preload loss
        cvp_drop = 0.3 * map_drop
        # SPO2 starts dropping after 20 min of significant blood loss
        spo2_drop = 0.0
        if minutes > 20 and map_drop > 20:
            spo2_drop = min(0.1 * (minutes - 20), 5.0)

        return {
            "map": -map_drop,
            "hr": hr_rise,
            "cvp": -cvp_drop,
            "spo2": -spo2_drop,
        }


@dataclass
class SepsisTrajectory(DiseaseTrajectory):
    """
    Sepsis trajectory.

    Produces:
      - Initial MAP drop (vasodilation)
      - Later HR rise
      - Fever (temperature increase)
      - RR increase (compensatory hyperventilation)
    """
    temp_rise_per_hour: float = 1.0
    map_drop_max: float = 30.0

    def vital_delta_at(self, sim_time: datetime) -> dict[str, float]:
        hours = max(0, (sim_time - self.onset_time).total_seconds() / 3600.0)
        # MAP drops quickly then stabilizes
        map_drop = self.map_drop_max * (1 - np.exp(-hours / 0.5)) * self.severity
        # HR rises with cytokine response
        hr_rise = min(40.0 * hours * self.severity, 60.0)
        # Temperature rises
        temp_rise = min(self.temp_rise_per_hour * hours * self.severity, 3.0)
        # RR increases
        rr_rise = min(5.0 * hours * self.severity, 15.0)

        return {
            "map": -map_drop,
            "hr": hr_rise,
            "temp_c": temp_rise,
            "rr": rr_rise,
        }


class DiseaseInjector:
    """
    Manages active disease trajectories and applies them to patient state.
    """

    def __init__(self) -> None:
        self._trajectories: list[DiseaseTrajectory] = []

    def inject(self, trajectory: DiseaseTrajectory) -> str:
        """Add a disease trajectory to the active set."""
        self._trajectories.append(trajectory)
        return f"TRJ-{len(self._trajectories):04d}"

    def compute_vital_deltas(self, sim_time: datetime) -> dict[str, float]:
        """
        Aggregate vital sign deltas from all active trajectories at sim_time.
        """
        deltas: dict[str, float] = {}
        for trj in self._trajectories:
            if sim_time < trj.onset_time:
                continue
            for metric, delta in trj.vital_delta_at(sim_time).items():
                deltas[metric] = deltas.get(metric, 0.0) + delta
        return deltas

    def apply_to_baseline(
        self,
        baseline: dict[str, float],
        sim_time: datetime,
    ) -> dict[str, float]:
        """
        Apply disease deltas to a baseline vital sign set.
        """
        deltas = self.compute_vital_deltas(sim_time)
        result = dict(baseline)
        for metric, delta in deltas.items():
            result[metric] = result.get(metric, 0.0) + delta
        return result
