"""
End-to-end integration tests.

Week 4 Exit Criteria:
  1. 1-hour ICU simulation with 3 concurrent stub agents without deadlocking
  2. PK/PD synergy produces expected respiratory depression
  3. Constraint Validator rejects 2g Propofol, passes 200mg
  4. DiseaseInjection produces recognizable vital sign pattern
  5. Rollback circuit breaker fires under deliberate causal conflict
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from engine.disease.injector import DiseaseInjector, OccultHemorrhage
from engine.pdes.engine import CausalViolation, PDES_Engine
from engine.pkpd.models import HemodynamicState, PKEngine
from engine.proto.v1 import simulation_pb2 as pb
from engine.validators.constraint import ConstraintValidator, ValidationError
from engine.validators.sdi import SDICalculator


class TestDiseaseInjection:
    """Disease trajectories produce recognizable vital sign patterns."""

    def test_occult_hemorrhage_map_drop(self) -> None:
        """
        Inject occult hemorrhage, run for 30 min.
        MAP should drop recognizably.
        """
        injector = DiseaseInjector()
        onset = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        trj = OccultHemorrhage(onset_time=onset, severity=0.8)
        injector.inject(trj)

        baseline = {"map": 80.0, "hr": 70.0, "cvp": 8.0, "spo2": 97.0}
        sim_time = datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc)
        vitals = injector.apply_to_baseline(baseline, sim_time)

        assert vitals["map"] < baseline["map"] - 20, (
            f"MAP should drop significantly: {vitals['map']:.1f}"
        )
        assert vitals["hr"] > baseline["hr"] + 20, (
            f"HR should rise compensatorily: {vitals['hr']:.1f}"
        )

    def test_sdi_increases_with_disease(self) -> None:
        """SDI should rise as disease progresses."""
        injector = DiseaseInjector()
        onset = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        injector.inject(OccultHemorrhage(onset_time=onset, severity=0.8))

        calc = SDICalculator()
        baseline = {"map": 80.0, "hr": 70.0, "rr": 16.0, "spo2": 97.0, "temp_c": 37.0}

        sdi_0 = calc.compute(baseline)

        sim_time = datetime(2024, 1, 1, 12, 45, tzinfo=timezone.utc)
        diseased = injector.apply_to_baseline(baseline, sim_time)
        sdi_45 = calc.compute(diseased)

        assert sdi_45 > sdi_0, (
            f"SDI should increase with disease: {sdi_0:.3f} -> {sdi_45:.3f}"
        )
        assert sdi_45 > 0.3, f"SDI after 45min hemorrhage should be elevated: {sdi_45:.3f}"


class TestPDESWithStubAgents:
    """PDES barrier loop with stub agents."""

    def test_three_agents_no_deadlock(self) -> None:
        """
        1-hour simulation with 3 stub agents.
        Should complete without deadlock.
        """
        start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)

        engine = PDES_Engine(store=None, start_time=start, end_time=end)

        # Stub agents: each returns a harmless intent
        def nurse_agent(snapshot: pb.StateSnapshot) -> pb.AgentIntent:
            return pb.AgentIntent(
                agent_id="N1",
                action_type="CHECK_VITALS",
                payload=b'{}',
                based_on_hash=snapshot.version_hash,
            )

        def rt_agent(snapshot: pb.StateSnapshot) -> pb.AgentIntent:
            return pb.AgentIntent(
                agent_id="RT1",
                action_type="ADJUST_VENT",
                payload=b'{"peep": 10, "tidal_volume": 500}',
                based_on_hash=snapshot.version_hash,
            )

        def resident_agent(snapshot: pb.StateSnapshot) -> pb.AgentIntent:
            return pb.AgentIntent(
                agent_id="R1",
                action_type="ADMINISTER_MED",
                payload=b'{"drug_name": "Propofol", "dose": 100, "unit": "mg", "route": "IV_bolus"}',
                based_on_hash=snapshot.version_hash,
            )

        engine.register_agent("N1", nurse_agent)
        engine.register_agent("RT1", rt_agent)
        engine.register_agent("R1", resident_agent)

        import asyncio
        asyncio.run(engine.run())

        # Should have advanced through all barriers (4 barriers in 1 hour at 15-min)
        assert engine.clock.current_time >= end

    def test_causal_violation_rollback(self) -> None:
        """
        Deliberate causal conflict: agent deliberates on stale snapshot.
        Rollback should fire.
        """
        start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc)

        engine = PDES_Engine(store=None, start_time=start, end_time=end)

        # Agent that always uses wrong hash (simulating stale snapshot)
        call_count = [0]

        def conflicting_agent(snapshot: pb.StateSnapshot) -> pb.AgentIntent:
            call_count[0] += 1
            return pb.AgentIntent(
                agent_id="R1",
                action_type="ADMINISTER_MED",
                payload=b'{"drug_name": "Fentanyl", "dose": 0.05, "unit": "mg", "route": "IV_bolus"}',
                based_on_hash="wrong-hash",  # Deliberate mismatch
            )

        engine.register_agent("R1", conflicting_agent)

        import asyncio
        asyncio.run(engine.run())

        # Agent should have been called multiple times due to rollback retries
        assert call_count[0] > 1, (
            f"Expected multiple calls due to rollback, got {call_count[0]}"
        )


class TestPKPDSynergy:
    """PK/PD drug-drug interaction in simulation context."""

    def test_propofol_fentanyl_respiratory_depression(self) -> None:
        """
        Propofol + Fentanyl synergy produces expected respiratory depression.
        Week 4 Exit Criteria #2.
        """
        engine = PKEngine()
        regimen = {
            "Propofol": (10.0, 100.0 / 60.0),
            "Fentanyl": (10.0, 0.02),
        }
        combo = engine.simulate_combination_infusion(regimen)

        shared = combo["shared_effect_fraction"]
        peak_idx = np.argmax(shared)

        # Combined effect should be greater than either drug alone
        prop_frac = combo["Propofol_fractional"][peak_idx]
        fent_frac = combo["Fentanyl_fractional"][peak_idx]

        assert shared[peak_idx] > prop_frac, "Synergy should exceed propofol alone"
        assert shared[peak_idx] > fent_frac, "Synergy should exceed fentanyl alone"

    def test_hemodynamic_feedback_on_clearance(self) -> None:
        """
        Low cardiac output reduces clearance, increasing concentration.
        High cardiac output increases clearance, decreasing concentration.
        """
        engine = PKEngine()
        normal = HemodynamicState(co=5.0, co_baseline=5.0, map_mmhg=70.0, hr=70.0)
        shock = HemodynamicState(co=3.0, co_baseline=5.0, map_mmhg=55.0, hr=110.0)

        normal_res = engine.simulate_infusion("Propofol", 10.0, 2.0, normal)
        shock_res = engine.simulate_infusion("Propofol", 10.0, 2.0, shock)

        assert shock_res["C1"][-1] > normal_res["C1"][-1], (
            "Shock should increase propofol concentration"
        )
