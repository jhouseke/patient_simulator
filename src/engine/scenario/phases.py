"""OR/ICU phase model: ordering, per-phase cadence, and scripted entry actions.

Each phase declares how often agents deliberate (barrier cadence) and which
actions fire on entry. Entry actions are ordinary intents: the engine validates
and commits them exactly like agent intents, so they inherit the same safety
checks, audit trail, and rollback semantics (Arch Decision 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    PREOP = "PREOP"
    INDUCTION = "INDUCTION"
    INTUBATION = "INTUBATION"
    MAINTENANCE = "MAINTENANCE"
    EMERGENCE = "EMERGENCE"
    EXTUBATION = "EXTUBATION"
    PACU = "PACU"
    ICU = "ICU"


PHASE_ORDER: list[Phase] = [
    Phase.PREOP,
    Phase.INDUCTION,
    Phase.INTUBATION,
    Phase.MAINTENANCE,
    Phase.EMERGENCE,
    Phase.EXTUBATION,
    Phase.PACU,
    Phase.ICU,
]


@dataclass
class PhaseConfig:
    phase: Phase
    barrier_seconds: float
    expected_roles: list[str] = field(default_factory=list)
    #: Intent specs applied on entry, as {"action_type": ..., **payload}.
    entry_action_specs: list[dict] = field(default_factory=list)


# Maintenance is TIVA (propofol + fentanyl infusions), not a volatile agent.
# Pulse 4.x appears to have dropped the anesthesia machine, so inhaled agents
# have no confirmed delivery path — see docs/pulse_api_notes.md §6. TIVA maps
# cleanly onto SESubstanceInfusion and is clinically defensible. Once a volatile
# path is confirmed, add a variant scenario rather than changing this default.
_CONFIGS: dict[Phase, PhaseConfig] = {
    Phase.PREOP: PhaseConfig(
        Phase.PREOP,
        barrier_seconds=300,
        expected_roles=["Anesthesiologist", "Nurse"],
    ),
    Phase.INDUCTION: PhaseConfig(
        Phase.INDUCTION,
        barrier_seconds=30,
        expected_roles=["Anesthesiologist", "CRNA", "Nurse"],
        entry_action_specs=[
            {"action_type": "ADMINISTER_MED", "drug_name": "Propofol",
             "dose": 150, "unit": "mg", "route": "IV_bolus"},
            {"action_type": "ADMINISTER_MED", "drug_name": "Fentanyl",
             "dose": 0.1, "unit": "mg", "route": "IV_bolus"},
            {"action_type": "ADMINISTER_MED", "drug_name": "Rocuronium",
             "dose": 50, "unit": "mg", "route": "IV_bolus"},
        ],
    ),
    Phase.INTUBATION: PhaseConfig(
        Phase.INTUBATION,
        barrier_seconds=15,
        expected_roles=["Anesthesiologist", "CRNA"],
        entry_action_specs=[
            {"action_type": "INTUBATE", "airway": "ETT", "pulse_type": "Tracheal"},
            {"action_type": "ADJUST_VENT", "peep": 5, "tidal_volume": 480,
             "rate": 12, "fio2": 0.5},
        ],
    ),
    Phase.MAINTENANCE: PhaseConfig(
        Phase.MAINTENANCE,
        barrier_seconds=300,
        expected_roles=["Anesthesiologist", "CRNA", "Nurse"],
        entry_action_specs=[
            {"action_type": "ADMINISTER_MED", "drug_name": "Propofol",
             "rate": 9.0, "unit": "mg/min", "route": "IV_infusion"},
            {"action_type": "ADMINISTER_MED", "drug_name": "Fentanyl",
             "rate": 0.02, "unit": "mg/min", "route": "IV_infusion"},
        ],
    ),
    Phase.EMERGENCE: PhaseConfig(
        Phase.EMERGENCE,
        barrier_seconds=60,
        expected_roles=["Anesthesiologist", "CRNA"],
        entry_action_specs=[
            {"action_type": "ADMINISTER_MED", "drug_name": "Propofol",
             "rate": 0.0, "unit": "mg/min", "route": "IV_infusion"},
            {"action_type": "ADMINISTER_MED", "drug_name": "Sugammadex",
             "dose": 200, "unit": "mg", "route": "IV_bolus"},
        ],
    ),
    Phase.EXTUBATION: PhaseConfig(
        Phase.EXTUBATION,
        barrier_seconds=30,
        expected_roles=["Anesthesiologist", "CRNA"],
        entry_action_specs=[{"action_type": "EXTUBATE"}],
    ),
    Phase.PACU: PhaseConfig(
        Phase.PACU,
        barrier_seconds=300,
        expected_roles=["Nurse", "Anesthesiologist"],
    ),
    Phase.ICU: PhaseConfig(
        Phase.ICU,
        barrier_seconds=15 * 60,
        expected_roles=["Nurse", "Resident", "RespiratoryTherapist"],
    ),
}


def phase_config(phase: Phase) -> PhaseConfig:
    return _CONFIGS[phase]
