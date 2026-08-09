"""PDES engine <-> physiology integration (Task 1.5).

Offline only: store=None and a fake adapter, so this stays in the fast suite.
Postgres-backed and real-Pulse tests live in test_physiology_store.py and are
marked slow / DB-gated.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from engine.pdes.engine import PDES_Engine

START = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class FakeAdapter:
    """Stand-in for PulseAdapter with the same interface."""

    def __init__(self):
        self.t = 0.0
        self.advances = 0
        self.actions = []
        self.initialized = False
        self.loaded = []

    def initialize(self):
        self.initialized = True

    def advance(self, s):
        self.t += s
        self.advances += 1

    def apply_action(self, a):
        self.actions.append(a)

    def sample_scalars(self):
        return {"hr": 70.0 + self.advances, "map": 80.0}

    def save_state(self):
        return f"state@{self.t}".encode()

    def load_state(self, blob):
        self.loaded.append(blob)

    def close(self):
        pass


def _run(engine):
    asyncio.run(engine.run())


def test_engine_advances_pulse_and_charts_between_barriers():
    fake = FakeAdapter()
    captured = []
    engine = PDES_Engine(
        store=None,
        start_time=START,
        end_time=START + timedelta(hours=1),
        physiology=fake,
    )
    engine.on_vitals_sampled = lambda t, v: captured.append((t, v))
    _run(engine)

    assert fake.initialized, "engine must initialize physiology"
    assert fake.t == pytest.approx(3600.0), "must advance exactly the simulated hour"
    assert len(captured) == 720, "1h at the default 5s charting cadence => 720 samples"
    assert captured[-1][1]["hr"] > captured[0][1]["hr"]


def test_charting_timestamps_are_ordered_and_bounded():
    fake = FakeAdapter()
    captured = []
    engine = PDES_Engine(
        store=None,
        start_time=START,
        end_time=START + timedelta(minutes=30),
        physiology=fake,
        charting_interval_s=60.0,
    )
    engine.on_vitals_sampled = lambda t, v: captured.append(t)
    _run(engine)

    assert captured == sorted(captured), "charting timestamps must be monotonic"
    assert captured[0] == START + timedelta(seconds=60)
    assert captured[-1] == START + timedelta(minutes=30)
    assert len(captured) == 30


def test_engine_without_physiology_is_unchanged():
    """physiology defaults to None so existing behavior is preserved."""
    engine = PDES_Engine(store=None, start_time=START, end_time=START + timedelta(hours=1))
    _run(engine)  # must not raise


def test_committed_med_intent_reaches_physiology():
    from engine.physiology.actions import SubstanceBolus
    from engine.proto.v1 import simulation_pb2 as pb

    fake = FakeAdapter()
    engine = PDES_Engine(
        store=None,
        start_time=START,
        end_time=START + timedelta(minutes=30),
        physiology=fake,
    )

    def resident(snapshot):
        intent = pb.AgentIntent(
            agent_id="R1",
            action_type="ADMINISTER_MED",
            payload=b'{"drug_name":"Propofol","dose":100,"unit":"mg","route":"IV_bolus"}',
            based_on_hash=snapshot.version_hash,
        )
        intent.proposed_time.CopyFrom(snapshot.sim_time)
        return intent

    engine.register_agent("R1", resident)
    _run(engine)

    assert fake.actions, "approved intents must reach Pulse"
    assert all(isinstance(a, SubstanceBolus) for a in fake.actions)
    assert fake.actions[0].substance == "Propofol"
    assert fake.actions[0].volume_ml == pytest.approx(10.0)


def test_non_physiologic_intent_does_not_reach_physiology():
    from engine.proto.v1 import simulation_pb2 as pb

    fake = FakeAdapter()
    engine = PDES_Engine(
        store=None,
        start_time=START,
        end_time=START + timedelta(minutes=30),
        physiology=fake,
    )

    def nurse(snapshot):
        intent = pb.AgentIntent(
            agent_id="N1", action_type="CALL_ATTENDING", based_on_hash=snapshot.version_hash
        )
        intent.proposed_time.CopyFrom(snapshot.sim_time)
        return intent

    engine.register_agent("N1", nurse)
    _run(engine)
    assert fake.actions == []


def test_checkpoint_captures_physiology_state():
    fake = FakeAdapter()
    engine = PDES_Engine(
        store=None,
        start_time=START,
        end_time=START + timedelta(minutes=30),
        physiology=fake,
    )
    _run(engine)
    assert engine._checkpoints
    assert all(cp.physiology_state is not None for cp in engine._checkpoints)


def test_rollback_reverts_physiology():
    """A causal violation must restore Pulse to the checkpoint state."""
    from engine.proto.v1 import simulation_pb2 as pb

    fake = FakeAdapter()
    engine = PDES_Engine(
        store=None,
        start_time=START,
        end_time=START + timedelta(minutes=15),
        physiology=fake,
    )

    calls = {"n": 0}

    def stale_agent(snapshot):
        calls["n"] += 1
        # First deliberation is based on a stale hash -> causal violation
        hash_ = "stale-hash" if calls["n"] == 1 else snapshot.version_hash
        intent = pb.AgentIntent(
            agent_id="R1", action_type="CHECK_VITALS", based_on_hash=hash_
        )
        intent.proposed_time.CopyFrom(snapshot.sim_time)
        return intent

    engine.register_agent("R1", stale_agent)
    _run(engine)

    assert fake.loaded, "rollback must restore physiology state"
