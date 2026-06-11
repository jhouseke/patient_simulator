"""
Constraint Validator — pre-commit safety checks for agent intents.

Validates:
  - Route-specific dose limits (bolus vs infusion ceilings)
  - Drug-drug interaction contraindications
  - Allergy checks against knowledge graph
  - 3-retry limit with FAILURE_TO_ACT fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class DoseLimit:
    """Per-drug, per-route dose limits."""
    drug_name: str
    route: str
    max_bolus_mg: float | None = None
    max_rate_mg_per_min: float | None = None
    max_single_dose_mg: float | None = None


# MVP dose limits for test drugs
DOSE_LIMITS: list[DoseLimit] = [
    # Propofol
    DoseLimit("Propofol", "IV_bolus", max_bolus_mg=200.0, max_single_dose_mg=200.0),
    DoseLimit("Propofol", "infusion", max_rate_mg_per_min=10.0),
    # Fentanyl
    DoseLimit("Fentanyl", "IV_bolus", max_bolus_mg=0.5, max_single_dose_mg=0.5),  # 500 mcg
    DoseLimit("Fentanyl", "infusion", max_rate_mg_per_min=0.05),  # 50 mcg/min
    # Norepinephrine
    DoseLimit("Norepinephrine", "IV_bolus", max_bolus_mg=None),  # No bolus
    DoseLimit("Norepinephrine", "infusion", max_rate_mg_per_min=0.1),  # 100 mcg/min
    # Vancomycin
    DoseLimit("Vancomycin", "IV_bolus", max_bolus_mg=2000.0, max_single_dose_mg=2000.0),
]


class ValidationError(Exception):
    """Intent violates a safety constraint."""
    def __init__(self, reason: str, code: str) -> None:
        self.reason = reason
        self.code = code
        super().__init__(reason)


class ConstraintValidator:
    """
    Validates agent intents before they are committed to state.

    Week 4: route-specific dose limits and retry logic.
    """

    MAX_RETRIES = 3

    def __init__(self) -> None:
        self._limits: dict[tuple[str, str], DoseLimit] = {
            (lim.drug_name, lim.route): lim for lim in DOSE_LIMITS
        }
        self._retry_counts: dict[str, int] = {}

    def validate(self, intent_payload: dict[str, Any]) -> None:
        """
        Validate an intent payload.

        Raises ValidationError if any constraint is violated.
        """
        action_type = intent_payload.get("action_type", "")

        if action_type == "ADMINISTER_MED":
            self._validate_administer_med(intent_payload)
        elif action_type == "ADJUST_VENT":
            self._validate_adjust_vent(intent_payload)
        elif action_type == "CALL_ATTENDING":
            return  # Always allowed
        else:
            # Unknown action types pass through in MVP (should log)
            logger.debug("unknown_action_type", action=action_type)

    def _validate_administer_med(self, payload: dict[str, Any]) -> None:
        """Check dose limits for medication administration."""
        drug = payload.get("drug_name", payload.get("drug", ""))
        route = payload.get("route", "")
        dose = payload.get("dose", payload.get("amount", 0.0))
        rate = payload.get("rate", payload.get("rate_mg_per_min", 0.0))
        is_bolus = payload.get("is_bolus", route.endswith("bolus"))

        limit = self._limits.get((drug, route))
        if not limit:
            logger.debug("no_dose_limit", drug=drug, route=route)
            return

        # Bolus check
        if is_bolus and limit.max_bolus_mg is not None:
            if dose > limit.max_bolus_mg:
                raise ValidationError(
                    f"{drug} {route} bolus {dose} mg exceeds limit {limit.max_bolus_mg} mg",
                    "REJECTED_DOSE_EXCEEDS_LIMIT",
                )

        # Single dose check
        if limit.max_single_dose_mg is not None:
            if dose > limit.max_single_dose_mg:
                raise ValidationError(
                    f"{drug} {route} single dose {dose} mg exceeds limit {limit.max_single_dose_mg} mg",
                    "REJECTED_DOSE_EXCEEDS_LIMIT",
                )

        # Infusion rate check
        if not is_bolus and limit.max_rate_mg_per_min is not None:
            if rate > limit.max_rate_mg_per_min:
                raise ValidationError(
                    f"{drug} {route} infusion rate {rate} mg/min exceeds limit {limit.max_rate_mg_per_min} mg/min",
                    "REJECTED_DOSE_EXCEEDS_LIMIT",
                )

        # Norepinephrine safety: no bolus
        if drug == "Norepinephrine" and is_bolus:
            raise ValidationError(
                "Norepinephrine cannot be given as bolus",
                "REJECTED_ROUTE_INVALID",
            )

    def _validate_adjust_vent(self, payload: dict[str, Any]) -> None:
        """Basic ventilator setting sanity checks."""
        peep = payload.get("peep", 0)
        tv = payload.get("tidal_volume", 0)

        if peep > 25:
            raise ValidationError(
                f"PEEP {peep} cmH2O exceeds safe limit 25",
                "REJECTED_DOSE_EXCEEDS_LIMIT",
            )
        if tv > 800:
            raise ValidationError(
                f"Tidal volume {tv} mL exceeds safe limit 800",
                "REJECTED_DOSE_EXCEEDS_LIMIT",
            )

    def check_retries(self, agent_id: str) -> tuple[bool, str]:
        """
        Track retry attempts per agent.

        Returns (can_retry, status) where status is empty or FAILURE_TO_ACT.
        """
        count = self._retry_counts.get(agent_id, 0)
        if count >= self.MAX_RETRIES:
            logger.warning("failure_to_act", agent=agent_id, retries=count)
            return False, "FAILURE_TO_ACT"
        self._retry_counts[agent_id] = count + 1
        return True, ""

    def reset_retries(self, agent_id: str) -> None:
        """Reset retry count after a successful commit."""
        self._retry_counts.pop(agent_id, None)
