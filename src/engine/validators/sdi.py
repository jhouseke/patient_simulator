"""
SDI Calculator — Simulation Distress Index.

Computes a patient-specific z-score based on vital sign deviation
from normal ranges. 0 = healthy, 1 = critical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VitalRange:
    """Normal range for a vital sign metric."""
    metric_code: str
    mean: float
    std: float
    weight: float = 1.0


# Simplified normal ranges for 65-year-old ICU patient
VITAL_RANGES: list[VitalRange] = [
    VitalRange("hr", mean=75.0, std=10.0, weight=1.0),      # Heart rate
    VitalRange("map", mean=80.0, std=10.0, weight=1.5),     # Mean arterial pressure
    VitalRange("rr", mean=16.0, std=3.0, weight=1.2),       # Respiratory rate
    VitalRange("spo2", mean=97.0, std=2.0, weight=1.0),     # Oxygen saturation
    VitalRange("temp_c", mean=37.0, std=0.5, weight=0.8),   # Temperature
]


class SDICalculator:
    """
    Computes the Simulation Distress Index (0-1).

    Uses a weighted z-score approach: for each vital, compute the
    absolute z-score from normal, weight it, and aggregate.
    """

    def __init__(self) -> None:
        self.ranges: dict[str, VitalRange] = {vr.metric_code: vr for vr in VITAL_RANGES}

    def compute(self, vitals: dict[str, float]) -> float:
        """
        Compute SDI from a dict of {metric_code: value}.

        Returns float in [0, 1].
        """
        total_score = 0.0
        total_weight = 0.0

        for metric, value in vitals.items():
            vr = self.ranges.get(metric)
            if not vr:
                continue

            z = abs(value - vr.mean) / vr.std if vr.std > 0 else 0.0
            # Clamp z to reasonable range (5 sigma = nearly dead)
            z = min(z, 5.0)
            # Convert to 0-1 contribution
            contribution = (z / 5.0) * vr.weight

            total_score += contribution
            total_weight += vr.weight

        if total_weight == 0:
            return 0.0

        raw = total_score / total_weight
        return float(np.clip(raw, 0.0, 1.0))

    def compute_from_snapshot(self, snapshot_vitals: list) -> float:
        """
        Compute SDI from a list of VitalSign protobuf messages.
        """
        vitals = {v.metric_code: v.value for v in snapshot_vitals}
        return self.compute(vitals)
