"""
PK/PD unit test suite.

Assertions mirror clinical expectations:
  - Fentanyl bolus produces dose-dependent RR depression
  - Propofol infusion produces sedation/respiratory depression
  - Norepinephrine increases MAP
  - Propofol + Fentanyl synergy produces more respiratory depression than either alone
  - Hemodynamic feedback alters clearance
  - Concentration profiles are physiologically plausible
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.pkpd.models import (
    FENTANYL,
    PROPOFOL,
    NOREPINEPHRINE,
    HemodynamicState,
    PKEngine,
    directional_bliss,
    minto_response_surface,
    pd_effect,
)


class TestSingleDrugPKPD:
    """Single-drug pharmacokinetic / pharmacodynamic assertions."""

    def test_fentanyl_dose_dependent_rr_depression(self) -> None:
        """
        Higher fentanyl doses should produce more respiratory depression.
        Verify monotonic dose-response relationship.
        """
        engine = PKEngine()
        doses = [50, 100, 200]  # mcg bolus over 1 min
        rr_drops = []

        for dose_mcg in doses:
            result = engine.simulate_infusion("Fentanyl", duration_min=1.0, rate_mg_per_min=dose_mcg / 1000.0)
            idx_5min = np.argmin(np.abs(result["t"] - 5.0))
            ce_ng_ml = result["Ce"][idx_5min] * 1000.0
            rr_drop = -engine.pd_response("Fentanyl", ce_ng_ml)
            rr_drops.append(rr_drop)

        # Dose-dependent response
        assert rr_drops[0] < rr_drops[1] < rr_drops[2], (
            f"Expected monotonic dose-response: {rr_drops}"
        )
        # Some effect at highest dose
        assert rr_drops[-1] > 0.5, f"Expected meaningful RR depression at 200mcg, got {rr_drops[-1]:.2f}"

    def test_propofol_infusion_sedation(self) -> None:
        """
        Propofol 100 mg/hr (1.67 mg/min) for 10 min.
        Expected: Effect-site concentration rises and produces RR depression.
        """
        engine = PKEngine()
        result = engine.simulate_infusion("Propofol", duration_min=10.0, rate_mg_per_min=100.0 / 60.0)

        idx_10min = -1
        ce_ng_ml = result["Ce"][idx_10min] * 1000.0

        # Concentration should be in a plausible range for propofol
        assert 1.0 <= ce_ng_ml <= 500.0, f"Propofol Ce {ce_ng_ml:.1f} ng/mL outside plausible range"

        rr_drop = -engine.pd_response("Propofol", ce_ng_ml)
        assert rr_drop > 0, f"Expected some RR depression, got {rr_drop:.2f} bpm"
        # At 100 mg/hr for 10 min, expect some depression (up to Emax=30)
        assert 1.0 <= rr_drop <= 30.0, f"RR drop {rr_drop:.2f} bpm outside plausible range"

    def test_norepinephrine_map_increase(self) -> None:
        """
        Norepinephrine 0.1 mcg/kg/min for 70kg = 7 mcg/min = 0.007 mg/min.
        Expected: MAP increase.
        """
        engine = PKEngine()
        result = engine.simulate_infusion(
            "Norepinephrine", duration_min=10.0, rate_mg_per_min=0.007
        )

        idx_10min = -1
        ce_ng_ml = result["Ce"][idx_10min] * 1000.0

        # Plausible concentration range for norepinephrine
        assert 0.1 <= ce_ng_ml <= 50.0, f"Norepi Ce {ce_ng_ml:.2f} ng/mL outside plausible range"

        map_increase = engine.pd_response("Norepinephrine", ce_ng_ml)
        assert map_increase > 0, f"Expected MAP increase, got {map_increase:.2f} mmHg"
        assert 1.0 <= map_increase <= 30.0, (
            f"MAP increase {map_increase:.2f} mmHg outside plausible range"
        )


class TestHemodynamicFeedback:
    """Clearance scales with cardiac output."""

    def test_low_co_reduces_clearance(self) -> None:
        """
        Low cardiac output (shock) should reduce effective clearance.
        Same infusion should produce higher concentrations.
        """
        engine = PKEngine()

        normal_hemo = HemodynamicState(co=5.0, co_baseline=5.0, map_mmhg=70.0, hr=70.0)
        low_hemo = HemodynamicState(co=3.0, co_baseline=5.0, map_mmhg=55.0, hr=100.0)

        normal = engine.simulate_infusion("Propofol", 10.0, 2.0, normal_hemo)
        low = engine.simulate_infusion("Propofol", 10.0, 2.0, low_hemo)

        assert low["C1"][-1] > normal["C1"][-1], (
            f"Low CO should increase concentration: normal={normal['C1'][-1]:.3f}, "
            f"low={low['C1'][-1]:.3f}"
        )

    def test_high_co_increases_clearance(self) -> None:
        """High cardiac output (hyperdynamic) should increase clearance."""
        engine = PKEngine()

        normal_hemo = HemodynamicState(co=5.0, co_baseline=5.0, map_mmhg=70.0, hr=70.0)
        high_hemo = HemodynamicState(co=8.0, co_baseline=5.0, map_mmhg=85.0, hr=110.0)

        normal = engine.simulate_infusion("Fentanyl", 10.0, 0.05, normal_hemo)
        high = engine.simulate_infusion("Fentanyl", 10.0, 0.05, high_hemo)

        assert high["C1"][-1] < normal["C1"][-1], (
            f"High CO should decrease concentration: normal={normal['C1'][-1]:.3f}, "
            f"high={high['C1'][-1]:.3f}"
        )


class TestDrugDrugInteractions:
    """Drug-drug interaction models."""

    def test_propofol_fentanyl_synergy(self) -> None:
        """
        Propofol + Fentanyl together should produce MORE respiratory
        depression than either alone at the same concentrations.

        Uses directional Bliss Independence.
        """
        engine = PKEngine()
        regimen = {
            "Propofol": (10.0, 100.0 / 60.0),   # 10 min at 100 mg/hr
            "Fentanyl": (10.0, 0.02),            # 10 min at 20 mcg/min
        }

        combo = engine.simulate_combination_infusion(regimen)
        shared = combo["shared_effect_fraction"]

        # At some point during infusion, shared effect should exceed
        # either drug's individual fractional effect
        found_synergy = False
        for i in range(len(shared)):
            prop_frac = combo["Propofol_fractional"][i]
            fent_frac = combo["Fentanyl_fractional"][i]
            if shared[i] > prop_frac and shared[i] > fent_frac and shared[i] > 0.01:
                found_synergy = True
                break

        assert found_synergy, (
            f"Combined effect should exceed individual effects at some timepoint. "
            f"Max shared={max(shared):.4f}, max prop={max(combo['Propofol_fractional']):.4f}, "
            f"max fent={max(combo['Fentanyl_fractional']):.4f}"
        )

    def test_minto_surface_monotonic(self) -> None:
        """Minto response surface should be monotonic in both dimensions."""
        ce_props = np.linspace(0, 10, 50)
        ce_fents = np.linspace(0, 5, 50)

        for cp in ce_props[1:]:
            for cf in ce_fents[1:]:
                e = minto_response_surface(cp, cf)
                e_prev_p = minto_response_surface(ce_props[ce_props < cp][-1], cf)
                e_prev_f = minto_response_surface(cp, ce_fents[ce_fents < cf][-1])
                assert e >= e_prev_p, "Minto surface not monotonic in Propofol"
                assert e >= e_prev_f, "Minto surface not monotonic in Fentanyl"

    def test_directional_bliss_bounds(self) -> None:
        """Bliss combined effect should be in [max(a,b), a+b-a*b]."""
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
                combined = directional_bliss(a, b, direction="depress")
                assert 0.0 <= combined <= 1.0
                assert combined >= max(a, b)


class TestConcentrationProfiles:
    """Sanity checks on PK concentration-time profiles."""

    def test_concentration_nonnegative(self) -> None:
        """All compartment amounts and concentrations must be >= 0."""
        engine = PKEngine()
        for drug in ["Propofol", "Fentanyl", "Norepinephrine"]:
            result = engine.simulate_infusion(drug, 10.0, 1.0)
            assert np.all(result["A1"] >= 0), f"{drug} A1 negative"
            assert np.all(result["C1"] >= 0), f"{drug} C1 negative"
            assert np.all(result["Ce"] >= 0), f"{drug} Ce negative"

    def test_effect_lags_central(self) -> None:
        """Effect-site concentration should lag behind central."""
        engine = PKEngine()
        result = engine.simulate_infusion("Propofol", 10.0, 2.0)

        # Find time of peak central concentration
        peak_c1_idx = np.argmax(result["C1"])
        peak_ce_idx = np.argmax(result["Ce"])

        assert peak_ce_idx >= peak_c1_idx, (
            "Effect-site peak should lag or coincide with central peak"
        )
