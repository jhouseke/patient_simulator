"""Causal timing, replay and observation integration independent of Pulse."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from engine.pdes.engine import PDES_Engine
from engine.pdes.snapshot import build_snapshot
from engine.proto.v1 import simulation_pb2 as pb
from engine.observation import ConfigurableObservationModel, MetricObservationConfig
from engine.scenario.library import elective_lap_chole
from engine.scenario.machine import ScenarioStateMachine
from engine.scenario.phases import Phase

START = datetime(2026, 1, 1, tzinfo=timezone.utc)

class Patient:
    def __init__(self):
        self.t = 0.0
        self.actions = []
    def initialize(self): pass
    def advance(self, seconds): self.t += seconds
    def sample_scalars(self): return {"hr": 70.0 + self.t, "map": 80.0}
    def apply_action(self, action): self.actions.append((self.t, type(action).__name__))
    def save_state(self): return json.dumps([self.t, self.actions]).encode()
    def load_state(self, state):
        self.t, actions = json.loads(state)
        self.actions = [tuple(a) for a in actions]


def engine(patient=None, seconds=17, **kwargs):
    return PDES_Engine(None, START, START + timedelta(seconds=seconds),
                       physiology=patient or Patient(), **kwargs)


def med(snapshot, delay=0, stale=False):
    result = pb.AgentIntent(agent_id="R", action_type="ADMINISTER_MED",
                            payload=b'{"drug_name":"Propofol","dose":10,"unit":"mg","route":"IV_bolus"}',
                            based_on_hash="stale" if stale else snapshot.version_hash)
    result.proposed_time.FromDatetime(snapshot.sim_time.ToDatetime(tzinfo=timezone.utc) + timedelta(seconds=delay))
    return result


def test_future_order_is_administered_at_its_actual_time():
    patient = Patient()
    e = engine(patient, barrier_seconds=20)
    def agent(snapshot):
        if snapshot.sim_time.ToDatetime(tzinfo=timezone.utc) == START:
            return med(snapshot, 7)
        result = pb.AgentIntent(agent_id="R", action_type="CHECK_VITALS", based_on_hash=snapshot.version_hash)
        result.proposed_time.CopyFrom(snapshot.sim_time)
        return result
    e.register_agent("R", agent)
    times = []
    e.on_vitals_sampled = lambda t, v: times.append((t - START).total_seconds())
    asyncio.run(e.run())
    assert patient.actions == [(7, "SubstanceBolus")]
    assert times == [5, 10, 15, 17]
    assert patient.t == 17
    assert e.clock.current_time == START + timedelta(seconds=17)


def test_sequential_fallback_applies_each_action_once():
    patient = Patient()
    e = engine(patient, seconds=0)
    e._sequential_mode = True
    e.register_agent("R", med)
    asyncio.run(e.run())
    assert patient.actions == [(0, "SubstanceBolus")]


def test_agent_sees_observations_and_store_keeps_distinct_truth():
    patient = Patient()
    observation = ConfigurableObservationModel(metrics={"hr": MetricObservationConfig(lag_s=5)})
    e = engine(patient, seconds=10, barrier_seconds=5, observation_model=observation)
    seen = []
    def agent(snapshot):
        seen.append({v.metric_code: v.value for v in snapshot.vitals})
        return pb.AgentIntent(agent_id="R", action_type="CHECK_VITALS", based_on_hash=snapshot.version_hash)
    e.register_agent("R", agent)
    truth, observed = [], []
    e.on_truth_sampled = lambda t, v: truth.append(v)
    e.on_vitals_sampled = lambda t, v: observed.append(v)
    asyncio.run(e.run())
    assert "hr" not in seen[0]
    assert seen[1]["hr"] == 70
    assert truth[0]["hr"] == 75
    assert observed[0]["hr"] == 70


def test_phase_boundaries_and_non_aligned_events_are_exact():
    scenario = elective_lap_chole()
    scenario.phase_durations_s = {p: 1 for p in scenario.phase_durations_s}
    scenario.events = [{"phase": Phase.PREOP, "at_s": 0.5,
                        "disease_type": "OCCULT_HEMORRHAGE", "params": {"severity": 0.1, "compartment": "Liver"}}]
    patient = Patient()
    policy = ScenarioStateMachine(scenario, patient, None)
    e = engine(patient, seconds=8, scenario=policy)
    asyncio.run(e.run())
    assert (0.5, "Hemorrhage") in patient.actions
    assert (1, "SubstanceBolus") in patient.actions
    assert (2, "Intubation") in patient.actions
    assert patient.t == 8


def test_retry_restores_scenario_actions_and_observation_rng():
    scenario = elective_lap_chole()
    scenario.events = [{"phase": Phase.PREOP, "at_s": 0, "disease_type": "OCCULT_HEMORRHAGE",
                        "params": {"severity": 0.1, "compartment": "Liver"}}]
    patient = Patient()
    policy = ScenarioStateMachine(scenario, patient, None)
    e = engine(patient, seconds=5, scenario=policy)
    calls = 0
    def agent(snapshot):
        nonlocal calls
        calls += 1
        return med(snapshot, stale=calls == 1)
    e.register_agent("R", agent)
    asyncio.run(e.run())
    assert patient.actions.count((0, "Hemorrhage")) == 1
    assert patient.actions.count((0, "SubstanceBolus")) == 1


def test_hash_changes_when_visible_state_changes_at_same_time():
    async def run():
        a = await build_snapshot(None, START, observed_vitals={"hr": 70})
        b = await build_snapshot(None, START, observed_vitals={"hr": 71})
        assert a.version_hash != b.version_hash
    asyncio.run(run())


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_chart_cadence_rejected(value):
    with pytest.raises(ValueError): engine(charting_interval_s=value)


def test_causal_retry_publishes_only_final_actions_and_distinct_sources():
    class Store:
        def __init__(self): self.actions, self.samples = [], []
        async def fetch(self, *args): return []
        async def execute(self, query, *args): self.actions.append(args)
        async def insert_vitals_batch(self, patient_id, batch, source): self.samples.append((source, batch))
    store = Store()
    patient = Patient()
    e = PDES_Engine(store, START, START + timedelta(seconds=5), physiology=patient)
    calls = 0
    def stale_once(snapshot):
        nonlocal calls
        calls += 1
        return med(snapshot, stale=calls == 1)
    e.register_agent("A", med)
    e.register_agent("B", stale_once)
    asyncio.run(e.run())
    # Two agents at the start and end; first speculative A is absent.
    assert len(store.actions) == 4
    assert len(patient.actions) == 4
    assert [source for source, _ in store.samples] == ["truth", "observed"]


def test_observation_and_queue_replay_is_identical():
    patient = Patient()
    observation = ConfigurableObservationModel(seed=42, metrics={"hr": MetricObservationConfig(noise_std=2)})
    e = engine(patient, seconds=20, observation_model=observation)
    from engine.pdes.clock import TimeBarrier
    from engine.pdes.engine import RollbackCheckpoint
    async def run():
        snapshot = await e._snapshot(START)
        record = e._pb_to_record(med(snapshot, 7))
        await e._commit_single(record, snapshot.version_hash)
        checkpoint = RollbackCheckpoint(START, snapshot.version_hash,
            physiology_state=patient.save_state(), runtime_state=e._runtime_state())
        outputs = []
        e.on_vitals_sampled = lambda t, v: outputs.append((t, v))
        await e._advance_physiology(TimeBarrier(START))
        expected = list(outputs)
        expected_actions = list(patient.actions)
        await e._rollback(checkpoint)
        outputs.clear()
        await e._advance_physiology(TimeBarrier(START))
        assert outputs == expected
        assert patient.actions == expected_actions
    asyncio.run(run())


def test_observation_snapshot_keeps_measurement_publication_time():
    async def check():
        snapshot = await build_snapshot(None, START + timedelta(seconds=7),
            observed_vitals={"hr": 70}, observed_at=START + timedelta(seconds=5))
        assert snapshot.vitals[0].measured_at.ToDatetime(tzinfo=timezone.utc) == START + timedelta(seconds=5)
    asyncio.run(check())


def test_phase_notifications_are_published_once_after_causal_retry():
    scenario = elective_lap_chole()
    patient = Patient()
    policy = ScenarioStateMachine(scenario, patient, None)
    visited = []
    policy.on_phase_enter = visited.append
    e = engine(patient, seconds=5, scenario=policy)
    calls = 0
    def once_stale(snapshot):
        nonlocal calls
        calls += 1
        return med(snapshot, stale=calls == 1)
    e.register_agent("R", once_stale)
    asyncio.run(e.run())
    assert visited == [Phase.PREOP]


def test_scenario_action_precedes_coincident_chart():
    class EventPatient(Patient):
        def sample_scalars(self): return {"hr": 100 if self.actions else 70}
    scenario = elective_lap_chole()
    scenario.events = [{"phase": Phase.PREOP, "at_s": 5, "disease_type": "OCCULT_HEMORRHAGE",
                        "params": {"severity": 0.1, "compartment": "Liver"}}]
    patient = EventPatient()
    policy = ScenarioStateMachine(scenario, patient, None)
    e = engine(patient, seconds=10, scenario=policy)
    truth = []
    e.on_truth_sampled = lambda t, v: truth.append((t, v))
    asyncio.run(e.run())
    assert truth[0] == (START + timedelta(seconds=5), {"hr": 100})
