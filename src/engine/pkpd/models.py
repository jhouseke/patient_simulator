"""
PK/PD Engine — Pharmacokinetic / Pharmacodynamic models.

Week 3 implements:
  - 3-compartment PK for Propofol, Fentanyl, Norepinephrine
  - Hemodynamic feedback: CL_effective = CL_intrinsic * (CO/CO_baseline)^alpha
  - Shared Effect Compartment with directional Bliss Independence
  - Minto response surface for Propofol + Fentanyl synergy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class DrugPKParameters:
    """3-compartment PK parameters."""
    name: str
    v1: float       # Central compartment volume (L)
    v2: float       # Peripheral 1 volume (L)
    v3: float       # Peripheral 2 volume (L)
    cl: float       # Intrinsic clearance (L/min)
    q2: float       # Intercompartmental clearance 1 (L/min)
    q3: float       # Intercompartmental clearance 2 (L/min)
    ke0: float      # Effect compartment equilibration rate (1/min)
    # Hemodynamic scaling exponent (alpha in CL_eff = CL * (CO/CO0)^alpha)
    cl_alpha: float = 1.0


# Population PK estimates for a 70 kg adult (absolute volumes in L)
# Propofol: Schnider model-ish parameters scaled to 70 kg
PROPOFOL = DrugPKParameters(
    name="Propofol",
    v1=15.96, v2=32.41, v3=202.3,
    cl=1.88, q2=3.88, q3=2.12,
    ke0=0.456,
    cl_alpha=0.25,
)

# Fentanyl: Scott / Stanski population estimates scaled to 70 kg
FENTANYL = DrugPKParameters(
    name="Fentanyl",
    v1=7.35, v2=18.69, v3=157.5,
    cl=0.67, q2=5.83, q3=3.23,
    ke0=0.147,
    cl_alpha=0.30,
)

# Norepinephrine: simplified pressor model scaled to 70 kg
NOREPINEPHRINE = DrugPKParameters(
    name="Norepinephrine",
    v1=3.5, v2=7.0, v3=14.0,
    cl=2.1, q2=1.4, q3=0.7,
    ke0=0.5,
    cl_alpha=0.20,
)


@dataclass
class HemodynamicState:
    """Patient hemodynamic state for clearance scaling."""
    co: float          # Cardiac output (L/min)
    co_baseline: float # Baseline cardiac output (L/min)
    map_mmhg: float    # Mean arterial pressure
    hr: float          # Heart rate

    def effective_clearance(self, params: DrugPKParameters) -> float:
        """Scale intrinsic clearance by cardiac output."""
        ratio = self.co / self.co_baseline if self.co_baseline > 0 else 1.0
        return params.cl * (ratio ** params.cl_alpha)


# ------------------------------------------------------------------
# PD Response parameters (simplified Emax models)
# ------------------------------------------------------------------

@dataclass
class PDParameters:
    """Pharmacodynamic Emax model parameters."""
    ec50_ng_ml: float       # Effect-site concentration for 50% effect (ng/mL)
    emax: float             # Maximum effect magnitude
    gamma: float            # Hill coefficient (steepness)
    baseline: float         # Baseline physiologic value
    direction: str          # "depress" or "stimulate"


PROPofol_PD_RESP = PDParameters(
    ec50_ng_ml=2.0, emax=30.0, gamma=3.0,
    baseline=15.0, direction="depress",  # Respiratory rate
)

FENTANYL_PD_RESP = PDParameters(
    ec50_ng_ml=0.5, emax=10.0, gamma=2.0,
    baseline=15.0, direction="depress",  # Respiratory rate
)

NOREPI_PD_MAP = PDParameters(
    ec50_ng_ml=2.0, emax=40.0, gamma=2.0,
    baseline=70.0, direction="stimulate",  # MAP mmHg
)


def pd_effect(ce_ng_ml: float, pd: PDParameters) -> float:
    """
    Emax/Hill equation for pharmacodynamic effect.

    Returns the *change* from baseline (negative for depress, positive for stimulate).
    """
    if ce_ng_ml <= 0:
        return 0.0
    fraction = (ce_ng_ml ** pd.gamma) / ((ce_ng_ml ** pd.gamma) + (pd.ec50_ng_ml ** pd.gamma))
    magnitude = fraction * pd.emax
    return -magnitude if pd.direction == "depress" else magnitude


# ------------------------------------------------------------------
# PK ODE with hemodynamic feedback
# ------------------------------------------------------------------

def pk_3compartment_ode(
    t: float,
    state: list[float],
    params: DrugPKParameters,
    hemo: HemodynamicState,
    infusion_rate: Callable[[float], float],
) -> list[float]:
    """
    ODE for 3-compartment model with effect compartment.

    state = [A1, A2, A3, A4]  (amounts in mg, A4 = effect compartment)
    infusion_rate(t) in mg/min
    """
    a1, a2, a3, a4 = state
    c1 = a1 / params.v1 if params.v1 > 0 else 0.0
    c2 = a2 / params.v2 if params.v2 > 0 else 0.0
    c3 = a3 / params.v3 if params.v3 > 0 else 0.0
    c4 = a4 / params.v1 if params.v1 > 0 else 0.0  # effect compartment "volume"

    cl_eff = hemo.effective_clearance(params)
    rate_in = infusion_rate(t)

    da1dt = (
        rate_in
        - cl_eff * c1
        - params.q2 * (c1 - c2)
        - params.q3 * (c1 - c3)
        - params.ke0 * (c1 - c4)
    )
    da2dt = params.q2 * (c1 - c2)
    da3dt = params.q3 * (c1 - c3)
    da4dt = params.ke0 * (c1 - c4)

    return [da1dt, da2dt, da3dt, da4dt]


# ------------------------------------------------------------------
# Minto Response Surface (Propofol + Fentanyl synergy)
# ------------------------------------------------------------------

def minto_response_surface(
    ce_prop: float, ce_fent: float,
    emax_prop: float = 30.0,
    emax_fent: float = 10.0,
    gamma: float = 1.8,
) -> float:
    """
    Minto interaction model for two hypnotic/analgesic drugs.

    Returns the *fractional* effect (0-1) on a shared endpoint
    (e.g. probability of unconsciousness, or respiratory depression).

    Uses an interaction term with gamma for synergy.

    Ref: Minto et al. Anesthesiology 2000
    """
    # Normalized concentrations (assuming ec50 = 1 for simplicity in
    # fractional terms; caller should scale by actual EC50 if needed)
    u_prop = ce_prop
    u_fent = ce_fent

    # Interaction surface with gamma
    interaction = (u_prop + u_fent + u_prop * u_fent) ** gamma
    effect = interaction / (1.0 + interaction)
    return effect


# ------------------------------------------------------------------
# Directional Bliss Independence
# ------------------------------------------------------------------

def directional_bliss(
    effect_a: float,
    effect_b: float,
    direction: str = "depress",
) -> float:
    """
    Directional Bliss Independence for shared effect compartment.

    For depressant effects (e.g. respiratory depression):
        E_ab = E_a + E_b - E_a * E_b   (multiplicative on survival fraction)

    For stimulant effects:
        E_ab = 1 - (1 - E_a)(1 - E_b)  (additive on response fraction)

    Here effect_a and effect_b are the *fractional* effects (0-1)
    of each drug alone.
    """
    ea = max(0.0, min(1.0, effect_a))
    eb = max(0.0, min(1.0, effect_b))

    if direction == "depress":
        # Both reduce a physiologic function
        return ea + eb - (ea * eb)
    else:
        # Both enhance a physiologic function
        return 1.0 - (1.0 - ea) * (1.0 - eb)


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class PKEngine:
    """
    Pharmacokinetic / Pharmacodynamic simulation engine.

    Simulates drug concentrations and physiologic effects over time,
    including hemodynamic feedback and drug-drug interactions.
    """

    def __init__(self) -> None:
        self.drugs: dict[str, DrugPKParameters] = {
            "Propofol": PROPOFOL,
            "Fentanyl": FENTANYL,
            "Norepinephrine": NOREPINEPHRINE,
        }
        self.pd: dict[str, PDParameters] = {
            "Propofol": PROPofol_PD_RESP,
            "Fentanyl": FENTANYL_PD_RESP,
            "Norepinephrine": NOREPI_PD_MAP,
        }

    def simulate_infusion(
        self,
        drug_name: str,
        duration_min: float,
        rate_mg_per_min: float,
        hemo: HemodynamicState | None = None,
        dt: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """
        Run a constant-rate infusion simulation.

        Returns dict with keys:
          't': time points (min)
          'A1', 'A2', 'A3', 'A4': compartment amounts (mg)
          'C1': central concentration (mg/L)
          'Ce': effect-site concentration (mg/L)
        """
        params = self.drugs.get(drug_name)
        if not params:
            raise ValueError(f"Unknown drug: {drug_name}")

        hemo = hemo or HemodynamicState(co=5.0, co_baseline=5.0, map_mmhg=70.0, hr=70.0)

        t_eval = np.arange(0, duration_min + dt, dt)
        infusion = lambda tau: rate_mg_per_min if tau <= duration_min else 0.0
        y0 = [0.0, 0.0, 0.0, 0.0]

        sol = solve_ivp(
            fun=lambda t, y: pk_3compartment_ode(t, y, params, hemo, infusion),
            t_span=(0, duration_min),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
        )

        a1, a2, a3, a4 = sol.y
        c1 = a1 / params.v1
        ce = a4 / params.v1

        return {
            "t": sol.t,
            "A1": a1, "A2": a2, "A3": a3, "A4": a4,
            "C1": c1,
            "Ce": ce,
        }

    def pd_response(
        self,
        drug_name: str,
        ce_ng_ml: float,
    ) -> float:
        """
        Compute physiologic effect change from effect-site concentration.

        Returns the *delta* from baseline (e.g. -5 bpm for RR depression).
        """
        pd = self.pd.get(drug_name)
        if not pd:
            raise ValueError(f"Unknown drug: {drug_name}")
        return pd_effect(ce_ng_ml, pd)

    def simulate_combination_infusion(
        self,
        regimen: dict[str, tuple[float, float]],
        hemo: HemodynamicState | None = None,
        dt: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """
        Simulate multiple simultaneous drug infusions.

        regimen: {drug_name: (duration_min, rate_mg_per_min)}

        Returns combined results with 'shared_effect' key using
        directional Bliss Independence.
        """
        hemo = hemo or HemodynamicState(co=5.0, co_baseline=5.0, map_mmhg=70.0, hr=70.0)

        # Determine common time span
        max_dur = max(dur for dur, _ in regimen.values())
        t_eval = np.arange(0, max_dur + dt, dt)

        results: dict[str, dict[str, np.ndarray]] = {}
        for drug, (dur, rate) in regimen.items():
            results[drug] = self.simulate_infusion(drug, dur, rate, hemo, dt)

        # Align time vectors (all should be identical if same dt and max_dur)
        # Compute PD effect at each time point for each drug
        pd_effects: dict[str, np.ndarray] = {}
        for drug, res in results.items():
            # Convert Ce (mg/L) to ng/mL: 1 mg/L = 1000 ng/mL
            ce_ng_ml = res["Ce"] * 1000.0
            pd_params = self.pd[drug]
            # Fractional effect (magnitude, 0-1)
            pd_effects[drug] = np.array([
                abs(pd_effect(ce, pd_params)) / pd_params.emax if pd_params.emax > 0 else 0.0
                for ce in ce_ng_ml
            ])

        # Directional Bliss for shared endpoint (respiratory depression)
        if "Propofol" in pd_effects and "Fentanyl" in pd_effects:
            shared = np.array([
                directional_bliss(p, f, direction="depress")
                for p, f in zip(pd_effects["Propofol"], pd_effects["Fentanyl"])
            ])
        else:
            shared = np.zeros_like(t_eval)

        # Return aligned result set
        return {
            "t": t_eval,
            **{f"{k}_Ce": results[k]["Ce"] for k in results},
            **{f"{k}_fractional": pd_effects[k] for k in pd_effects},
            "shared_effect_fraction": shared,
        }
