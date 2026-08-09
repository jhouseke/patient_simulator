"""Built-in surgical scenarios: phase durations + injected events."""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.scenario.phases import Phase


@dataclass
class Scenario:
    name: str
    #: Phase -> seconds spent in that phase.
    phase_durations_s: dict[Phase, float]
    #: [{phase, at_s, disease_type, params}] — at_s is relative to phase start.
    events: list[dict] = field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return float(sum(self.phase_durations_s.values()))


def elective_lap_chole() -> Scenario:
    return Scenario(
        name="elective_laparoscopic_cholecystectomy",
        phase_durations_s={
            Phase.PREOP: 600,
            Phase.INDUCTION: 120,
            Phase.INTUBATION: 60,
            Phase.MAINTENANCE: 3600,
            Phase.EMERGENCE: 300,
            Phase.EXTUBATION: 120,
            Phase.PACU: 1800,
            Phase.ICU: 3600,
        },
    )


def trauma_laparotomy() -> Scenario:
    s = elective_lap_chole()
    s.name = "trauma_exploratory_laparotomy"
    s.phase_durations_s[Phase.MAINTENANCE] = 7200
    s.events = [
        {
            "phase": Phase.MAINTENANCE,
            "at_s": 1800,
            "disease_type": "OCCULT_HEMORRHAGE",
            # Pulse compartments are anatomic — see actions.HEMORRHAGE_COMPARTMENTS.
            "params": {"severity": 0.7, "compartment": "Liver"},
        }
    ]
    return s


SCENARIOS = {
    "elective_laparoscopic_cholecystectomy": elective_lap_chole,
    "trauma_exploratory_laparotomy": trauma_laparotomy,
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]()
    except KeyError:
        raise ValueError(
            f"unknown scenario {name!r}; expected one of {sorted(SCENARIOS)}"
        ) from None
