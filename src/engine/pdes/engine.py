"""
PDES Engine — Parallel Discrete Event Simulation core.

Manages the simulation lifecycle:
  1. Advance to next time barrier
  2. Build state snapshot
  3. Distribute snapshot to agents for deliberation
  4. Collect intents
  5. Sort by proposed_time
  6. Commit sequentially, detecting causal violations
  7. Rollback if needed (max depth 2, sequential fallback at depth 3)
"""

from __future__ import annotations

import json
import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import structlog

from engine.pdes.clock import PDESClock, TimeBarrier
from engine.pdes.snapshot import build_snapshot
from engine.physiology.actions import intent_to_pulse_action
from engine.proto.v1 import simulation_pb2 as pb
from engine.store.postgres import PostgresStore
from engine.validators.constraint import ConstraintValidator, ValidationError

logger = structlog.get_logger()


@dataclass
class IntentRecord:
    """Internal representation of an agent intent."""
    agent_id: str
    action_type: str
    payload: dict[str, Any]
    proposed_time: datetime
    based_on_hash: str
    raw: pb.AgentIntent


@dataclass
class RollbackCheckpoint:
    """State checkpoint for rollback."""
    barrier_time: datetime
    version_hash: str
    committed_intents: list[IntentRecord] = field(default_factory=list)
    physiology_state: bytes | None = None
    runtime_state: Any = None


class CausalViolation(Exception):
    """Raised when an intent was based on a stale snapshot."""
    def __init__(self, intent: IntentRecord, expected_hash: str) -> None:
        self.intent = intent
        self.expected_hash = expected_hash
        super().__init__(
            f"Causal violation: agent {intent.agent_id} "
            f"based_on_hash={intent.based_on_hash} != current={expected_hash}"
        )


class PDES_Engine:
    """
    Core simulation engine.

    Usage:
        engine = PDES_Engine(store, start, end)
        engine.register_agent("N1", nurse_agent)
        await engine.run()
    """

    MAX_ROLLBACK_DEPTH = 2
    BARRIER_MINUTES = 15

    #: Default cadence at which scalar vitals are sampled *within* a barrier.
    #: Barriers control when agents deliberate; charting controls output
    #: resolution. VitalDB (our calibration reference) records at 1-2 s, so one
    #: sample per 15-minute barrier would be useless for research.
    CHARTING_INTERVAL_S = 5.0

    def __init__(
        self,
        store: PostgresStore | None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        patient_id: int = 1,
        physiology=None,                          # PulseAdapter | None
        barrier_seconds: float | None = None,
        charting_interval_s: float = CHARTING_INTERVAL_S,
        scenario=None,                            # ScenarioStateMachine | None
        observation_model=None,
    ) -> None:
        self.store = store
        self.barrier_seconds = (
            float(barrier_seconds)
            if barrier_seconds is not None
            else self.BARRIER_MINUTES * 60.0
        )
        self.clock = PDESClock(start_time, end_time, interval_s=self.barrier_seconds)
        self.patient_id = patient_id
        self.physiology = physiology
        self.charting_interval_s = charting_interval_s
        self.scenario = scenario
        self.observation_model = observation_model
        self._truth: dict[str, float] = {}
        self._observed: dict[str, float] = {}
        self._observed_at = self.clock.start_time
        self._revision = 0
        self._pending: list[IntentRecord] = []
        self._staged_actions: list[IntentRecord] = []
        self._physiology_time = self.clock.start_time
        self.on_truth_sampled = None
        self.on_waveforms_sampled = None
        if not math.isfinite(self.barrier_seconds) or self.barrier_seconds <= 0:
            raise ValueError("barrier_seconds must be finite and positive")
        if not math.isfinite(charting_interval_s) or charting_interval_s <= 0:
            raise ValueError("charting_interval_s must be finite and positive")
        self._next_chart_time = self.clock.start_time + timedelta(seconds=charting_interval_s)
        if self._next_chart_time <= self.clock.start_time:
            raise ValueError("charting_interval_s is below clock resolution")
        #: Optional Callable[[datetime, dict[str, float]], None] observer.
        self.on_vitals_sampled: Callable[[datetime, dict[str, float]], None] | None = None
        self._agents: dict[str, Callable[[pb.StateSnapshot], pb.AgentIntent]] = {}
        self._checkpoints: list[RollbackCheckpoint] = []
        self._current_version_hash = ""
        self._rollback_count = 0
        self._sequential_mode = False
        self._validator = ConstraintValidator()
        if self.physiology is not None:
            self.physiology.initialize()
            self._truth = dict(self.physiology.sample_scalars())
            self._observed = self._observe(self.clock.start_time, self._truth)

    def _observe(self, at, truth):
        return (self.observation_model.observe(at, dict(truth))
                if self.observation_model is not None else dict(truth))

    async def _snapshot(self, at):
        return await build_snapshot(
            self.store, at, self.patient_id,
            observed_vitals=self._observed if self.physiology is not None else None,
            state_revision=self._revision, observed_at=self._observed_at,
        )

    def _runtime_state(self):
        return copy.deepcopy({
            "pending": self._pending, "staged": self._staged_actions,
            "truth": self._truth, "observed": self._observed, "observed_at": self._observed_at,
            "revision": self._revision, "time": self._physiology_time,
            "chart_time": self._next_chart_time,
            "scenario": self.scenario.snapshot_state() if self.scenario is not None else None,
            "observation": self.observation_model.snapshot_state() if self.observation_model is not None else None,
        })

    def register_agent(
        self,
        agent_id: str,
        deliberator: Callable[[pb.StateSnapshot], pb.AgentIntent],
    ) -> None:
        """Register an agent function that receives a snapshot and returns an intent."""
        self._agents[agent_id] = deliberator

    async def run(self) -> None:
        """Run the full PDES simulation."""
        logger.info("pdes_engine_start", start=self.clock.start_time.isoformat())

        for barrier in self.clock.barriers():
            await self._process_barrier(barrier)

        logger.info("pdes_engine_complete", end=self.clock.current_time.isoformat())

    async def _process_barrier(self, barrier: TimeBarrier) -> None:
        """Process a single time barrier with optional rollback."""
        self.clock.current_time = barrier.sim_time
        logger.info("barrier_start", time=barrier.sim_time.isoformat())

        # A scenario policy sets this barrier's cadence before anything else
        # happens, so the clock's next step and the physiology advance agree.
        elapsed = (barrier.sim_time - self.clock.start_time).total_seconds()
        if self.scenario is not None:
            self.barrier_seconds = float(self.scenario.barrier_seconds(elapsed))
            next_event = self.scenario.next_event_s(elapsed)
            if next_event is not None:
                self.barrier_seconds = min(self.barrier_seconds, next_event - elapsed)
            self.clock.interval_s = self.barrier_seconds

        # Build snapshot at barrier time
        snapshot = await self._snapshot(barrier.sim_time)
        self._current_version_hash = snapshot.version_hash

        # Save checkpoint before deliberation
        checkpoint = RollbackCheckpoint(
            barrier_time=barrier.sim_time,
            version_hash=snapshot.version_hash,
            runtime_state=self._runtime_state(),
            physiology_state=(
                self.physiology.save_state() if self.physiology is not None else None
            ),
        )
        self._checkpoints.append(checkpoint)

        await self._apply_due(barrier.sim_time)

        # Scripted scenario actions go through the SAME validate -> commit ->
        # apply path as agent intents, so they get identical safety checks and
        # audit trail (Arch Decision 7).
        if self.scenario is not None:
            await self._commit_scenario_actions(barrier, snapshot, elapsed)

        snapshot = await self._snapshot(barrier.sim_time)
        self._current_version_hash = snapshot.version_hash
        # Deliberate and commit with rollback support
        await self._deliberate_and_commit(snapshot, barrier, depth=0)
        await self._flush_actions()

        # Advance physiology across the barrier, charting as we go
        await self._advance_physiology(barrier)

    async def _commit_scenario_actions(
        self, barrier: TimeBarrier, snapshot: pb.StateSnapshot, elapsed: float
    ) -> None:
        """Commit phase-entry and scheduled-event actions from the scenario."""
        for spec in self.scenario.pending_action_specs(elapsed):
            action_type = spec["action_type"]
            payload = {k: v for k, v in spec.items() if k != "action_type"}
            try:
                self._validator.validate({"action_type": action_type, **payload})
            except ValidationError as exc:
                # Scripted scenarios must be safe by construction; a failure
                # here is a bug in the scenario library, not agent behavior.
                logger.error(
                    "scenario_action_failed_validation", spec=spec, reason=exc.reason
                )
                continue
            record = IntentRecord(
                agent_id="SCENARIO",
                action_type=action_type,
                payload=payload,
                proposed_time=barrier.sim_time,
                based_on_hash=snapshot.version_hash,
                raw=None,
            )
            await self._commit_single(record, snapshot.version_hash)

    async def _advance_physiology(self, barrier: TimeBarrier) -> None:
        """Advance at exact action times while retaining a global charting grid."""
        stop = min(barrier.sim_time + timedelta(seconds=self.barrier_seconds), self.clock.end_time)
        batch, truth_batch = [], []
        while self._physiology_time < stop:
            future = [i.proposed_time for i in self._pending if i.proposed_time > self._physiology_time]
            target = min([stop, self._next_chart_time] + future)
            if self.physiology is not None:
                self.physiology.advance((target - self._physiology_time).total_seconds())
            self._physiology_time = target
            # Previously scheduled actions precede charts. Agents react to
            # published observations in the next deliberation, never retroactively.
            await self._apply_due(target)
            if self.scenario is not None and target == stop:
                before = await self._snapshot(target)
                await self._commit_scenario_actions(
                    TimeBarrier(target), before, (target - self.clock.start_time).total_seconds())
            if target == self._next_chart_time or target == self.clock.end_time:
                if self.physiology is not None:
                    self._truth = dict(self.physiology.sample_scalars())
                    self._observed = self._observe(target, self._truth)
                    self._observed_at = target
                    truth_batch.append((target, dict(self._truth)))
                    batch.append((target, dict(self._observed)))
                if target == self._next_chart_time:
                    self._next_chart_time += timedelta(seconds=self.charting_interval_s)
        await self._flush_actions()
        if self.store and batch:
            await self.store.insert_vitals_batch(self.patient_id, truth_batch, source="truth")
            await self.store.insert_vitals_batch(self.patient_id, batch, source="observed")
        for at, vitals in truth_batch:
            if self.on_truth_sampled:
                self.on_truth_sampled(at, dict(vitals))
        for at, vitals in batch:
            if self.on_vitals_sampled:
                self.on_vitals_sampled(at, dict(vitals))
        # Always drain enabled channels, even when no observer is attached.
        channels = getattr(self.physiology, "waveform_channels", ())
        if channels and stop > barrier.sim_time:
            waves = self.physiology.sample_waveforms((stop - barrier.sim_time).total_seconds())
            if self.on_waveforms_sampled:
                self.on_waveforms_sampled(barrier.sim_time, waves)

    async def _apply_due(self, at):
        due = sorted((i for i in self._pending if i.proposed_time <= at),
                     key=lambda i: (i.proposed_time, i.agent_id))
        self._pending = [i for i in self._pending if i.proposed_time > at]
        for intent in due:
            self._execute_action(intent)

    def _execute_action(self, intent):
        if self.physiology is not None:
            action = intent_to_pulse_action(intent.action_type, intent.payload)
            if action is not None:
                self.physiology.apply_action(action)
        self._revision += 1
        self._staged_actions.append(intent)

    async def _flush_actions(self):
        # Causal retries finish before any action audit row is published.
        if self.store:
            for intent in self._staged_actions:
                await self.store.execute(
                    """INSERT INTO agent_actions (time, agent_id, action_type, payload, based_on_hash)
                    VALUES ($1, $2, $3, $4, $5)""",
                    intent.proposed_time, intent.agent_id, intent.action_type,
                    json.dumps(intent.payload), intent.based_on_hash)
        self._staged_actions.clear()
        if self.scenario is not None:
            self.scenario.publish_notifications()

    async def _deliberate_and_commit(
        self,
        snapshot: pb.StateSnapshot,
        barrier: TimeBarrier,
        depth: int,
    ) -> None:
        """
        Collect intents from agents, sort by time, commit sequentially.

        If a causal violation is detected, rollback and retry up to
        MAX_ROLLBACK_DEPTH times. At depth 3, switch to sequential mode
        (agents deliberate one at a time in deterministic order).
        If sequential mode also fails, reject the offending intent and continue.
        """
        if depth > self.MAX_ROLLBACK_DEPTH:
            logger.warning(
                "rollback_exhausted_switching_to_sequential",
                barrier=barrier.sim_time.isoformat(),
            )
            self._sequential_mode = True
            depth = 0  # Reset depth for sequential mode

        try:
            if self._sequential_mode:
                intents = await self._deliberate_sequential(snapshot)
            else:
                intents = await self._deliberate_parallel(snapshot)

            if not self._sequential_mode:
                await self._commit_intents(intents, snapshot.version_hash)
            self._rollback_count = 0

        except CausalViolation as cv:
            if self._sequential_mode:
                # Sequential mode also failed — reject this intent and move on
                logger.error(
                    "causal_violation_in_sequential_mode_rejecting_intent",
                    agent=cv.intent.agent_id,
                    barrier=barrier.sim_time.isoformat(),
                )
                return

            self._rollback_count += 1
            logger.warning(
                "causal_violation_rollback",
                agent=cv.intent.agent_id,
                barrier=barrier.sim_time.isoformat(),
                depth=depth + 1,
            )
            await self._rollback(checkpoint=self._checkpoints[-1])
            # Rebuild snapshot after rollback
            await self._apply_due(barrier.sim_time)
            if self.scenario is not None:
                before = await self._snapshot(barrier.sim_time)
                await self._commit_scenario_actions(barrier, before, (barrier.sim_time - self.clock.start_time).total_seconds())
            new_snapshot = await self._snapshot(barrier.sim_time)
            await self._deliberate_and_commit(new_snapshot, barrier, depth + 1)

    async def _deliberate_parallel(
        self, snapshot: pb.StateSnapshot
    ) -> list[IntentRecord]:
        """
        Send snapshot to all agents concurrently.

        In production this would hit vLLM; here agents are Python callables.
        """
        intents: list[IntentRecord] = []
        for agent_id, deliberator in self._agents.items():
            pb_intent = deliberator(snapshot)
            record = self._pb_to_record(pb_intent)
            if record.proposed_time is None:
                record.proposed_time = self._physiology_time
            intents.append(record)
            logger.debug("intent_collected", agent=agent_id, action=record.action_type)
        return intents

    async def _deliberate_sequential(
        self, snapshot: pb.StateSnapshot
    ) -> list[IntentRecord]:
        """
        Agents deliberate one at a time in deterministic order.

        After each intent is committed, the next agent deliberates on the
        updated snapshot. This eliminates causal violations at the cost of
        throughput.
        """
        intents: list[IntentRecord] = []
        current_snapshot = snapshot

        for agent_id, deliberator in sorted(self._agents.items()):
            pb_intent = deliberator(current_snapshot)
            record = self._pb_to_record(pb_intent)
            if record.proposed_time is None:
                record.proposed_time = self._physiology_time

            # Commit immediately so next agent sees updated state
            await self._commit_single(record, current_snapshot.version_hash)
            intents.append(record)

            # Rebuild snapshot for next agent
            current_snapshot = await self._snapshot(self.clock.current_time)

        return intents

    async def _commit_intents(
        self, intents: list[IntentRecord], expected_hash: str
    ) -> None:
        """Commit intents in proposed_time order, checking causal validity."""
        # Sort by proposed_time (tie-break by agent_id for determinism)
        sorted_intents = sorted(intents, key=lambda i: (i.proposed_time, i.agent_id))

        for intent in sorted_intents:
            await self._commit_single(intent, expected_hash)

    async def _commit_single(self, intent: IntentRecord, expected_hash: str) -> None:
        """Commit a single intent after causal check."""
        if intent.based_on_hash != expected_hash:
            raise CausalViolation(intent, expected_hash)

        if not (self._physiology_time <= intent.proposed_time <= self.clock.end_time):
            raise ValueError("intent time must be within the remaining simulation")
        self._validator.validate({**intent.payload, "action_type": intent.action_type})
        if intent.proposed_time > self._physiology_time:
            self._pending.append(copy.deepcopy(intent))
            self._revision += 1
        else:
            self._execute_action(intent)

    async def _rollback(self, checkpoint: RollbackCheckpoint) -> None:
        """
        Rollback to a checkpoint.

        Restore a speculative barrier before its output is published.

        Audit writes are staged until causal retries succeed. This is not a
        database rewind API for previously published barriers.
        """
        logger.warning(
            "rollback_executing",
            checkpoint_time=checkpoint.barrier_time.isoformat(),
        )

        if self.physiology is not None and checkpoint.physiology_state is not None:
            self.physiology.load_state(checkpoint.physiology_state)
        state = copy.deepcopy(checkpoint.runtime_state)
        if state is not None:
            self._pending, self._staged_actions = state["pending"], state["staged"]
            self._truth, self._observed = state["truth"], state["observed"]
            self._observed_at = state["observed_at"]
            self._revision, self._physiology_time = state["revision"], state["time"]
            self._next_chart_time = state["chart_time"]
            if self.scenario is not None:
                self.scenario.restore_state(state["scenario"])
            if self.observation_model is not None:
                self.observation_model.restore_state(state["observation"])

        self._current_version_hash = checkpoint.version_hash

    @staticmethod
    def _pb_to_record(pb_intent: pb.AgentIntent) -> IntentRecord:
        """Convert protobuf AgentIntent to internal IntentRecord.

        `AgentIntent.payload` is JSON-encoded bytes holding the action details
        (drug_name, dose, route, ...), so it is decoded here rather than
        wrapping the whole message — the validator, the Pulse action mapper,
        and the agent_actions audit row all want the action details themselves.
        """
        proposed_time = (pb_intent.proposed_time.ToDatetime(tzinfo=timezone.utc)
                         if pb_intent.HasField("proposed_time") else None)
        payload: dict[str, Any] = {}
        if pb_intent.payload:
            try:
                payload = json.loads(pb_intent.payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning(
                    "intent_payload_not_json",
                    agent=pb_intent.agent_id,
                    action=pb_intent.action_type,
                )
        return IntentRecord(
            agent_id=pb_intent.agent_id,
            action_type=pb_intent.action_type,
            payload=payload,
            proposed_time=proposed_time,
            based_on_hash=pb_intent.based_on_hash,
            raw=pb_intent,
        )
