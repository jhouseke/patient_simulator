"""Translate approved AgentIntents (and scenario phase steps) into Pulse actions.

Our intent vocabulary -> Pulse action classes. Each PulseAction dataclass has a
`to_pulse()` method that PulseAdapter.apply_action hands to the engine. This
module deliberately does NOT import pulse, so intent translation is testable
without the compiled engine; `to_pulse()` is the only Pulse-aware seam and is
implemented by the adapter layer (see docs/pulse_api_notes.md §5).

Drug names must match Pulse substance names; SUBSTANCE_MAP handles aliasing
(e.g. 'Levophed' -> 'Norepinephrine').
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

SUBSTANCE_MAP = {
    "Levophed": "Norepinephrine",
    "Norepi": "Norepinephrine",
    "Neo": "Phenylephrine",
}

# Pulse's SESubstanceBolus takes a VOLUME (SEScalarVolume) plus a concentration
# (SEScalarMassPerVolume) — NOT a mass. A clinical order of "propofol 100 mg IV"
# must be converted to a vial concentration and a syringe volume. These are
# standard formulation strengths; keep consistent with ConstraintValidator.
CONCENTRATION_MG_PER_ML = {
    "Propofol": 10.0,
    "Fentanyl": 0.05,       # 50 mcg/mL
    "Rocuronium": 10.0,
    "Sugammadex": 100.0,
    "Succinylcholine": 20.0,
    "Midazolam": 1.0,
    "Norepinephrine": 0.016,  # 16 mcg/mL
    "Phenylephrine": 0.1,     # 100 mcg/mL
}
DEFAULT_CONCENTRATION_MG_PER_ML = 10.0

# pulse.cdm.patient_actions.eIntubationType
INTUBATION_TYPES = frozenset(
    {"Off", "Esophageal", "LeftMainstem", "RightMainstem",
     "Tracheal", "Oropharyngeal", "Nasopharyngeal"}
)

# pulse.cdm.patient_actions.eHemorrhage_Compartment — anatomic, no "Abdomen".
HEMORRHAGE_COMPARTMENTS = frozenset(
    {"Aorta", "Brain", "Muscle", "LargeIntestine", "LeftArm", "LeftKidney",
     "LeftLeg", "Liver", "RightArm", "RightKidney", "RightLeg", "Skin",
     "SmallIntestine", "Splanchnic", "Spleen", "VenaCava"}
)

# Disease types we can express as Pulse actions. Anything else is recorded in
# `unmapped_events` so a missing mapping is loud, never a silent no-op (Risk 2).
unmapped_events: set[str] = set()


def _canon(name: str) -> str:
    return SUBSTANCE_MAP.get(name, name)


def _concentration_for(substance: str) -> float:
    return CONCENTRATION_MG_PER_ML.get(substance, DEFAULT_CONCENTRATION_MG_PER_ML)


class PulseAction:
    """Base class. `to_pulse()` is filled in by the adapter layer."""

    def to_pulse(self):
        raise NotImplementedError


@dataclass
class SubstanceBolus(PulseAction):
    substance: str
    dose_mg: float
    concentration_mg_per_ml: float = DEFAULT_CONCENTRATION_MG_PER_ML

    @property
    def volume_ml(self) -> float:
        """Pulse wants a volume; clinicians order a mass."""
        return self.dose_mg / self.concentration_mg_per_ml


@dataclass
class SubstanceInfusion(PulseAction):
    substance: str
    rate_mg_per_min: float
    concentration_mg_per_ml: float = DEFAULT_CONCENTRATION_MG_PER_ML

    @property
    def rate_ml_per_min(self) -> float:
        return self.rate_mg_per_min / self.concentration_mg_per_ml


@dataclass
class VentilatorSettings(PulseAction):
    peep_cmh2o: float
    tidal_volume_ml: float
    rate_per_min: float
    fio2: float = 0.5


@dataclass
class Intubation(PulseAction):
    airway: str = "ETT"
    pulse_type: str = "Tracheal"

    def __post_init__(self) -> None:
        if self.pulse_type not in INTUBATION_TYPES:
            raise ValueError(
                f"unknown intubation type {self.pulse_type!r}; "
                f"expected one of {sorted(INTUBATION_TYPES)}"
            )


@dataclass
class Extubation(PulseAction):
    """Modeled in Pulse as SEIntubation(type=Off)."""

    pulse_type: str = "Off"


@dataclass
class InhaledAnesthetic(PulseAction):
    substance: str
    mac: float


@dataclass
class Hemorrhage(PulseAction):
    severity: float                 # 0..1 -> SEHemorrhage severity
    compartment: str = "Liver"      # anatomic; see HEMORRHAGE_COMPARTMENTS
    internal: bool = True

    def __post_init__(self) -> None:
        if self.compartment not in HEMORRHAGE_COMPARTMENTS:
            raise ValueError(
                f"unknown hemorrhage compartment {self.compartment!r}; "
                f"expected one of {sorted(HEMORRHAGE_COMPARTMENTS)}"
            )


def _event_to_action(payload: dict) -> PulseAction | None:
    disease_type = payload.get("disease_type", "")
    params = payload.get("params", {})
    if disease_type == "OCCULT_HEMORRHAGE":
        return Hemorrhage(
            severity=float(params.get("severity", 0.5)),
            compartment=params.get("compartment", "Liver"),
        )
    # Disease types Pulse lacks fall back to the SciPy disease injector — log
    # the gap loudly; NEVER silently no-op an event (Risk 2).
    unmapped_events.add(disease_type)
    logger.warning("event_has_no_pulse_mapping", disease_type=disease_type)
    return None


def intent_to_pulse_action(action_type: str, payload: dict) -> PulseAction | None:
    """Map an intent to a Pulse action, or None if it has no physiologic effect."""
    if action_type == "ADMINISTER_MED":
        sub = _canon(payload.get("drug_name", payload.get("drug", "")))
        route = payload.get("route", "")
        if route == "inhaled":
            return InhaledAnesthetic(substance=sub, mac=float(payload.get("mac", 1.0)))
        conc = float(payload.get("concentration_mg_per_ml", _concentration_for(sub)))
        if route.endswith("bolus"):
            return SubstanceBolus(
                substance=sub,
                dose_mg=float(payload.get("dose", 0.0)),
                concentration_mg_per_ml=conc,
            )
        return SubstanceInfusion(
            substance=sub,
            rate_mg_per_min=float(payload.get("rate", payload.get("rate_mg_per_min", 0.0))),
            concentration_mg_per_ml=conc,
        )

    if action_type == "ADJUST_VENT":
        return VentilatorSettings(
            peep_cmh2o=float(payload.get("peep", 5)),
            tidal_volume_ml=float(payload.get("tidal_volume", 500)),
            rate_per_min=float(payload.get("rate", 12)),
            fio2=float(payload.get("fio2", 0.5)),
        )

    if action_type == "INTUBATE":
        return Intubation(
            airway=payload.get("airway", "ETT"),
            pulse_type=payload.get("pulse_type", "Tracheal"),
        )

    if action_type == "EXTUBATE":
        return Extubation()

    if action_type == "INJECT_EVENT":
        return _event_to_action(payload)

    # CALL_ATTENDING, CHECK_VITALS, etc. have no physiologic effect.
    return None
