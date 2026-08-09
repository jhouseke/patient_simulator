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
            self.clock.interval_s = self.barrier_seconds

        # Build snapshot at barrier time
        snapshot = await build_snapshot(self.store, barrier.sim_time, self.patient_id)
        self._current_version_hash = snapshot.version_hash

        # Save checkpoint before deliberation
        checkpoint = RollbackCheckpoint(
            barrier_time=barrier.sim_time,
            version_hash=snapshot.version_hash,
            physiology_state=(
                self.physiology.save_state() if self.physiology is not None else None
            ),
        )
        self._checkpoints.append(checkpoint)

        # Scripted scenario actions go through the SAME validate -> commit ->
        # apply path as agent intents, so they get identical safety checks and
        # audit trail (Arch Decision 7).
        if self.scenario is not None:
            await self._commit_scenario_actions(barrier, snapshot, elapsed)

        # Deliberate and commit with rollback support
        await self._deliberate_and_commit(snapshot, barrier, depth=0)

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
        """Advance Pulse across the barrier, charting scalars as we go.

        Charting runs at `charting_interval_s` (default 5 s) rather than once
        per barrier: barriers control deliberation cadence, not data resolution
        (Arch Decision 8). Samples are batched into a single DB round-trip.
        """
        if self.physiology is None:
            return

        # Never advance past the simulation end — the clock yields a barrier at
        # end_time itself, which must not run a further interval of physiology.
        remaining = (self.clock.end_time - barrier.sim_time).total_seconds()
        total = min(self.barrier_seconds, max(remaining, 0.0))
        if total <= 0:
            return

        elapsed = 0.0
        batch: list[tuple[datetime, dict[str, float]]] = []
        while elapsed < total:
            step = min(self.charting_interval_s, total - elapsed)
            self.physiology.advance(step)
            elapsed += step
            batch.append(
                (
                    barrier.sim_time + timedelta(seconds=elapsed),
                    self.physiology.sample_scalars(),
                )
            )

        if self.store:
            await self.store.insert_vitals_batch(self.patient_id, batch, source="engine")
        if self.on_vitals_sampled:
            for sample_time, vitals in batch:
                self.on_vitals_sampled(sample_time, vitals)

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
            new_snapshot = await build_snapshot(
                self.store, barrier.sim_time, self.patient_id
            )
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

            # Commit immediately so next agent sees updated state
            await self._commit_single(record, current_snapshot.version_hash)
            intents.append(record)

            # Rebuild snapshot for next agent
            current_snapshot = await build_snapshot(
                self.store, self.clock.current_time, self.patient_id
            )

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

        # Write to agent_actions log
        if self.store:
            await self.store.execute(
                """
                INSERT INTO agent_actions (time, agent_id, action_type, payload, based_on_hash)
                VALUES ($1, $2, $3, $4, $5)
                """,
                intent.proposed_time,
                intent.agent_id,
                intent.action_type,
                # agent_actions.payload is JSONB and no asyncpg JSON codec is
                # registered, so hand it a JSON string, not a dict.
                json.dumps(intent.payload),
                intent.based_on_hash,
            )

        # Translate the approved intent into a Pulse action. Intents with no
        # physiologic effect (CALL_ATTENDING, CHECK_VITALS) map to None.
        if self.physiology is not None:
            pulse_action = intent_to_pulse_action(intent.action_type, intent.payload)
            if pulse_action is not None:
                self.physiology.apply_action(pulse_action)

        logger.debug(
            "intent_committed",
            agent=intent.agent_id,
            action=intent.action_type,
            time=intent.proposed_time.isoformat(),
        )

    async def _rollback(self, checkpoint: RollbackCheckpoint) -> None:
        """
        Rollback to a checkpoint.

        Removes all agent_actions after the checkpoint time and restores
        the version hash. In a full implementation this would also revert
        vital_signs and MAR writes caused by committed intents.
        """
        logger.warning(
            "rollback_executing",
            checkpoint_time=checkpoint.barrier_time.isoformat(),
        )

        if self.store:
            # Mark rolled-back actions
            await self.store.execute(
                """
                UPDATE agent_actions
                SET was_rolled_back = true
                WHERE time >= $1
                """,
                checkpoint.barrier_time,
            )

        # Revert physiology to the checkpoint so a rolled-back drug does not
        # leave its effect in the engine.
        if self.physiology is not None and checkpoint.physiology_state is not None:
            self.physiology.load_state(checkpoint.physiology_state)

        self._current_version_hash = checkpoint.version_hash

    @staticmethod
    def _pb_to_record(pb_intent: pb.AgentIntent) -> IntentRecord:
        """Convert protobuf AgentIntent to internal IntentRecord.

        `AgentIntent.payload` is JSON-encoded bytes holding the action details
        (drug_name, dose, route, ...), so it is decoded here rather than
        wrapping the whole message — the validator, the Pulse action mapper,
        and the agent_actions audit row all want the action details themselves.
        """
        proposed_time = pb_intent.proposed_time.ToDatetime(tzinfo=timezone.utc)
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
