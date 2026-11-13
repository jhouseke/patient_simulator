"""Scenario as a *policy* over PDES_Engine — one orchestration loop.

ScenarioStateMachine never advances physiology, samples vitals, or writes to
the store. It answers two questions the engine asks at each barrier:

  1. `barrier_seconds(elapsed)` — how often should agents deliberate now?
  2. `pending_action_specs(elapsed)` — which scripted actions have come due?

The engine validates and commits those specs through the same path as agent
intents (agent_id "SCENARIO"), so scripted OR actions inherit identical safety
validation, audit trail, and rollback behavior. This is Arch Decision 7; a
second advance/sample/persist loop here would duplicate the engine and
silently bypass all of that.

Phase 4 LLM agents simply register on the same engine — there is no
scripted-vs-agent mode switch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy

from engine.scenario.phases import PHASE_ORDER, Phase, phase_config

DEFAULT_START = datetime(2026, 1, 1, 7, 0, tzinfo=timezone.utc)


class ScenarioStateMachine:
    def __init__(
        self,
        scenario,
        physiology,
        store,
        start_time: datetime | None = None,
        patient_id: int = 1,
        note_writer=None,
    ) -> None:
        self.scenario = scenario
        self.physiology = physiology
        self.store = store
        self.patient_id = patient_id
        self.start = start_time or DEFAULT_START
        self.note_writer = note_writer
        #: Optional Callable[[Phase], None] observer.
        self.on_phase_enter = None
        self._entered_idx = -1
        self._fired_events: set[int] = set()
        self._phase_notifications: list[Phase] = []

    # ------------------------------------------------------------------
    # Policy interface consumed by PDES_Engine
    # ------------------------------------------------------------------

    def phase_at(self, elapsed_s: float) -> Phase:
        acc = 0.0
        for phase in PHASE_ORDER:
            acc += self.scenario.phase_durations_s[phase]
            if elapsed_s < acc:
                return phase
        return PHASE_ORDER[-1]

    def phase_start_s(self, phase: Phase) -> float:
        acc = 0.0
        for candidate in PHASE_ORDER:
            if candidate is phase:
                return acc
            acc += self.scenario.phase_durations_s[candidate]
        return acc

    def barrier_seconds(self, elapsed_s: float) -> float:
        return phase_config(self.phase_at(elapsed_s)).barrier_seconds

    def next_event_s(self, elapsed_s: float) -> float | None:
        """Next phase boundary or insult, so the engine never steps over it."""
        times = [self.phase_start_s(p) for p in PHASE_ORDER]
        times.extend(self.phase_start_s(e["phase"]) + e["at_s"]
                     for i, e in enumerate(self.scenario.events) if i not in self._fired_events)
        future = [t for t in times if t > elapsed_s]
        return min(future) if future else None

    def snapshot_state(self):
        return copy.deepcopy((self._entered_idx, self._fired_events, self._phase_notifications))

    def restore_state(self, state) -> None:
        self._entered_idx, self._fired_events, self._phase_notifications = copy.deepcopy(state)

    def publish_notifications(self) -> None:
        pending, self._phase_notifications = self._phase_notifications, []
        for phase in pending:
            if self.on_phase_enter:
                self.on_phase_enter(phase)

    def pending_action_specs(self, elapsed_s: float) -> list[dict]:
        """Phase-entry actions on transition, plus scenario events now due."""
        specs: list[dict] = []

        phase = self.phase_at(elapsed_s)
        idx = PHASE_ORDER.index(phase)
        if idx != self._entered_idx:
            # Fire entry actions for every phase crossed since the last barrier,
            # so a coarse cadence never steps over a short phase silently.
            for i in range(self._entered_idx + 1, idx + 1):
                crossed = PHASE_ORDER[i]
                self._phase_notifications.append(crossed)
                specs.extend(phase_config(crossed).entry_action_specs)
            self._entered_idx = idx

        for i, event in enumerate(self.scenario.events):
            if i in self._fired_events:
                continue
            due_at = self.phase_start_s(event["phase"]) + event["at_s"]
            if elapsed_s >= due_at:
                self._fired_events.add(i)
                specs.append(
                    {
                        "action_type": "INJECT_EVENT",
                        "disease_type": event["disease_type"],
                        "params": event.get("params", {}),
                    }
                )

        return specs

    # ------------------------------------------------------------------
    # Convenience wrapper: build the engine and run the whole case
    # ------------------------------------------------------------------

    async def run(self) -> None:
        from engine.pdes.engine import PDES_Engine

        engine = PDES_Engine(
            store=self.store,
            start_time=self.start,
            end_time=self.start + timedelta(seconds=self.scenario.total_duration_s),
            patient_id=self.patient_id,
            physiology=self.physiology,
            scenario=self,
        )
        await engine.run()
