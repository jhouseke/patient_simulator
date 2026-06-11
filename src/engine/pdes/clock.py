"""
PDES Clock — Parallel Discrete Event Simulation time barrier loop.

Week 1: skeleton only. The actual barrier loop (collect intents,
sort by time, detect causal violations, rollback) is implemented
in Week 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable


@dataclass
class TimeBarrier:
    """A single simulation time barrier."""
    sim_time: datetime
    intents: list[dict] = field(default_factory=list)


class PDESClock:
    """
    Manages simulation time progression.

    ICU mode: 15-minute barriers (96 barriers per simulated 24h).
    """

    BARRIER_MINUTES = 15

    def __init__(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        self.start_time = start_time or datetime.utcnow()
        self.end_time = end_time or (self.start_time + timedelta(hours=24))
        self.current_time = self.start_time
        self._barrier_callbacks: list[Callable[[TimeBarrier], None]] = []

    def register_callback(self, callback: Callable[[TimeBarrier], None]) -> None:
        self._barrier_callbacks.append(callback)

    def barriers(self):
        """Generate time barriers from start to end."""
        t = self.start_time
        while t <= self.end_time:
            yield TimeBarrier(sim_time=t)
            t += timedelta(minutes=self.BARRIER_MINUTES)

    async def run(self) -> None:
        """
        Run the PDES loop.

        Week 1: no-op skeleton. In Week 2 this will:
          1. Snapshot state at barrier time
          2. Send snapshot to agents (stub or LLM)
          3. Collect intents
          4. Sort by proposed_time
          5. Commit sequentially, detecting causal violations
          6. Rollback if needed (max depth 2)
        """
        for barrier in self.barriers():
            self.current_time = barrier.sim_time
            # TODO(week2): snapshot, deliberate, collect, sort, commit
            for cb in self._barrier_callbacks:
                cb(barrier)
