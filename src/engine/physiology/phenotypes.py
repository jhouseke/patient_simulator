"""Illustrative adult patient profiles; these distributions are not validated.

The sampled fields are all consumed by PulseAdapter.initialize. Profiles and
ranges require clinician review before clinical or educational interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

from engine.physiology.pulse_adapter import PatientConfig


@dataclass(frozen=True)
class PhenotypeSample:
    patient: PatientConfig
    phenotype: str
    seed: int
    parameters: dict[str, float]
    review_status: str = "clinician-review-pending"


def _bound(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def sample_patient(phenotype: str, seed: int) -> PhenotypeSample:
    """Sample a bounded adult profile with shared latent factors.

    Available profiles: adult, older_adult, and larger_adult. Their labels are
    descriptive examples, not diagnoses or validated patient populations.
    """
    profiles = {
        "adult": (45, 170, 25, 72, 85),
        "older_adult": (70, 167, 26, 75, 90),
        "larger_adult": (52, 172, 32, 76, 90),
    }
    if phenotype not in profiles:
        raise ValueError(f"unknown phenotype {phenotype!r}; choose from {', '.join(profiles)}")
    age_mean, height_mean, bmi_mean, hr_mean, map_mean = profiles[phenotype]
    rng = Random(seed)
    size = _bound(rng.gauss(0, 1), -2.5, 2.5)
    cardiovascular = _bound(rng.gauss(0, 1), -2.5, 2.5)
    age = round(_bound(rng.gauss(age_mean, 7), 18, 85))
    height = _bound(height_mean + 7 * size, 150, 195)
    bmi = _bound(bmi_mean + 2 * size + rng.gauss(0, 1.5), 19, 38)
    weight = _bound(bmi * (height / 100) ** 2, 45, 140)
    hr = _bound(hr_mean + 6 * cardiovascular + rng.gauss(0, 2), 50, 100)
    mean_pressure = _bound(map_mean + 7 * cardiovascular + rng.gauss(0, 2), 65, 110)
    patient = PatientConfig(sex=rng.choice(("MALE", "FEMALE")), age_yr=age,
                            height_cm=height, weight_kg=weight,
                            hr_baseline=hr, map_baseline=mean_pressure)
    return PhenotypeSample(patient, phenotype, seed,
                           {"size_factor": size, "cardiovascular_factor": cardiovascular,
                            "bmi": weight / (height / 100) ** 2})
