"""OR/ICU phase model and scenario policy (Tasks 3.1-3.3).

The scenario machine is a *policy* over PDES_Engine, not a second
orchestration loop (Arch Decision 7): it never advances physiology or writes
vitals itself, it only answers "what cadence?" and "what actions are due?".
"""

import asyncio

import pytest

from engine.scenario.phases import PHASE_ORDER, Phase, phase_config


class FakeAdapter:
    def __init__(self):
        self.actions = []
        self.t = 0.0

    def initialize(self):
        pass

    def advance(self, s):
        self.t += s

    def apply_action(self, a):
        self.actions.append(type(a).__name__)

    def sample_scalars(self):
        return {"hr": 75.0, "map": 80.0}

    def save_state(self):
        return b""

    def load_state(self, blob):
        pass

    def close(self):
        pass


# ----------------------------------------------------------------------
# Task 3.1 — phase definitions
# ----------------------------------------------------------------------


def test_phase_order_is_complete():
    assert PHASE_ORDER[0] == Phase.PREOP
    assert Phase.INDUCTION in PHASE_ORDER
    assert Phase.INTUBATION in PHASE_ORDER
    assert PHASE_ORDER[-1] == Phase.ICU


def test_induction_uses_fine_barriers():
    assert phase_config(Phase.INDUCTION).barrier_seconds <= 60
    assert phase_config(Phase.MAINTENANCE).barrier_seconds >= 300
    assert phase_config(Phase.ICU).barrier_seconds == 15 * 60


def test_every_phase_has_config_and_roles():
    for phase in PHASE_ORDER:
        cfg = phase_config(phase)
        assert cfg.barrier_seconds > 0
        assert cfg.expected_roles, f"{phase} has no expected roles"


def test_entry_actions_are_valid_intents():
    """Scripted actions flow through ConstraintValidator, so they must pass."""
    from engine.validators.constraint import ConstraintValidator

    validator = ConstraintValidator()
    for phase in PHASE_ORDER:
        for spec in phase_config(phase).entry_action_specs:
            payload = {k: v for k, v in spec.items() if k != "action_type"}
            validator.validate({"action_type": spec["action_type"], **payload})


def test_entry_actions_map_to_pulse_actions():
    """No scripted action may be silently unmapped."""
    from engine.physiology.actions import intent_to_pulse_action

    for phase in PHASE_ORDER:
        for spec in phase_config(phase).entry_action_specs:
            payload = {k: v for k, v in spec.items() if k != "action_type"}
            action = intent_to_pulse_action(spec["action_type"], payload)
            assert action is not None, f"{phase} spec {spec} has no Pulse mapping"


def test_maintenance_uses_tiva_not_volatile_by_default():
    """Pulse 4.x has no anesthesia machine, so maintenance is TIVA by default."""
    specs = phase_config(Phase.MAINTENANCE).entry_action_specs
    routes = {s.get("route") for s in specs}
    assert "inhaled" not in routes, (
        "volatile maintenance needs a confirmed Pulse delivery path first"
    )
    assert any(s.get("drug_name") == "Propofol" for s in specs)


# ----------------------------------------------------------------------
# Task 3.3 — scenario as a policy over the engine
# ----------------------------------------------------------------------


def test_policy_reports_phase_and_cadence_by_elapsed_time():
    from engine.scenario.library import elective_lap_chole
    from engine.scenario.machine import ScenarioStateMachine

    sm = ScenarioStateMachine(
        scenario=elective_lap_chole(), physiology=FakeAdapter(), store=None
    )
    assert sm.phase_at(0.0) == Phase.PREOP
    assert sm.barrier_seconds(0.0) == phase_config(Phase.PREOP).barrier_seconds
    assert sm.phase_at(650.0) == Phase.INDUCTION
    assert sm.barrier_seconds(650.0) == phase_config(Phase.INDUCTION).barrier_seconds


def test_full_case_visits_all_phases_with_fake_adapter():
    from engine.scenario.library import elective_lap_chole
    from engine.scenario.machine import ScenarioStateMachine

    scenario = elective_lap_chole()
    fake = FakeAdapter()
    sm = ScenarioStateMachine(scenario=scenario, physiology=fake, store=None)
    visited = []
    sm.on_phase_enter = lambda p: visited.append(p)
    asyncio.run(sm.run())

    assert visited == PHASE_ORDER
    assert "Intubation" in fake.actions and "SubstanceBolus" in fake.actions
    total = sum(scenario.phase_durations_s.values())
    assert fake.t == pytest.approx(float(total)), (
        "the engine, not the machine, advances physiology — and exactly once"
    )


def test_scenario_events_reach_physiology():
    from engine.scenario.library import trauma_laparotomy
    from engine.scenario.machine import ScenarioStateMachine

    fake = FakeAdapter()
    sm = ScenarioStateMachine(scenario=trauma_laparotomy(), physiology=fake, store=None)
    asyncio.run(sm.run())
    assert "Hemorrhage" in fake.actions, "scenario.events must be consumed, not dead data"


def test_events_fire_once():
    from engine.scenario.library import trauma_laparotomy
    from engine.scenario.machine import ScenarioStateMachine

    fake = FakeAdapter()
    sm = ScenarioStateMachine(scenario=trauma_laparotomy(), physiology=fake, store=None)
    asyncio.run(sm.run())
    assert fake.actions.count("Hemorrhage") == 1


def test_machine_does_not_advance_physiology_itself():
    """The policy must be inert — only the engine drives Pulse."""
    from engine.scenario.library import elective_lap_chole
    from engine.scenario.machine import ScenarioStateMachine

    fake = FakeAdapter()
    sm = ScenarioStateMachine(
        scenario=elective_lap_chole(), physiology=fake, store=None
    )
    sm.pending_action_specs(0.0)
    sm.barrier_seconds(0.0)
    sm.phase_at(0.0)
    assert fake.t == 0.0, "policy methods must not advance physiology"


def test_phase_entry_actions_fire_once_per_phase():
    from engine.scenario.library import elective_lap_chole
    from engine.scenario.machine import ScenarioStateMachine

    sm = ScenarioStateMachine(
        scenario=elective_lap_chole(), physiology=FakeAdapter(), store=None
    )
    sm.pending_action_specs(0.0)  # enter PREOP (assessment only, no drug specs)
    # INDUCTION starts at 600s and does have entry actions.
    entering = sm.pending_action_specs(600.0)
    still_inside = sm.pending_action_specs(630.0)
    assert [s["drug_name"] for s in entering] == ["Propofol", "Fentanyl", "Rocuronium"]
    assert still_inside == [], "entry actions must not repeat within a phase"


def test_skipped_phases_still_fire_entry_actions():
    """A coarse barrier must not step over a short phase silently."""
    from engine.scenario.library import elective_lap_chole
    from engine.scenario.machine import ScenarioStateMachine

    sm = ScenarioStateMachine(
        scenario=elective_lap_chole(), physiology=FakeAdapter(), store=None
    )
    sm.pending_action_specs(0.0)
    # Jump straight into MAINTENANCE, skipping INDUCTION and INTUBATION
    specs = sm.pending_action_specs(1000.0)
    types = [s["action_type"] for s in specs]
    assert "INTUBATE" in types, "intubation must not be skipped"
