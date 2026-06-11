"""
Constraint Validator unit tests.

Week 4 Exit Criteria #3:
  - The Constraint Validator correctly rejects a 2g Propofol bolus
    and passes a 200mg Propofol bolus.
"""

from __future__ import annotations

import pytest

from engine.validators.constraint import ConstraintValidator, ValidationError


class TestDoseLimits:
    """Route-specific dose limit validation."""

    def test_rejects_excessive_propofol_bolus(self) -> None:
        """2g Propofol IV bolus should be rejected."""
        validator = ConstraintValidator()
        payload = {
            "action_type": "ADMINISTER_MED",
            "drug_name": "Propofol",
            "route": "IV_bolus",
            "dose": 2000.0,
            "is_bolus": True,
        }
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(payload)
        assert "exceeds limit" in str(exc_info.value)

    def test_accepts_safe_propofol_bolus(self) -> None:
        """200mg Propofol IV bolus should be accepted."""
        validator = ConstraintValidator()
        payload = {
            "action_type": "ADMINISTER_MED",
            "drug_name": "Propofol",
            "route": "IV_bolus",
            "dose": 200.0,
            "is_bolus": True,
        }
        validator.validate(payload)  # Should not raise

    def test_rejects_norepinephrine_bolus(self) -> None:
        """Norepinephrine should never be given as bolus."""
        validator = ConstraintValidator()
        payload = {
            "action_type": "ADMINISTER_MED",
            "drug_name": "Norepinephrine",
            "route": "IV_bolus",
            "dose": 0.1,
            "is_bolus": True,
        }
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(payload)
        assert "cannot be given as bolus" in str(exc_info.value)

    def test_rejects_excessive_fentanyl(self) -> None:
        """Fentanyl bolus > 500mcg should be rejected."""
        validator = ConstraintValidator()
        payload = {
            "action_type": "ADMINISTER_MED",
            "drug_name": "Fentanyl",
            "route": "IV_bolus",
            "dose": 1.0,  # 1000 mcg
            "is_bolus": True,
        }
        with pytest.raises(ValidationError):
            validator.validate(payload)

    def test_rejects_high_peep(self) -> None:
        """PEEP > 25 should be rejected."""
        validator = ConstraintValidator()
        payload = {
            "action_type": "ADJUST_VENT",
            "peep": 30,
        }
        with pytest.raises(ValidationError):
            validator.validate(payload)


class TestRetryLogic:
    """3-retry limit with FAILURE_TO_ACT fallback."""

    def test_allows_three_retries(self) -> None:
        validator = ConstraintValidator()
        for i in range(3):
            can_retry, status = validator.check_retries("agent-1")
            assert can_retry, f"Retry {i+1} should be allowed"
            assert status == ""

    def test_failure_to_act_after_three_retries(self) -> None:
        validator = ConstraintValidator()
        for _ in range(3):
            validator.check_retries("agent-2")
        can_retry, status = validator.check_retries("agent-2")
        assert not can_retry
        assert status == "FAILURE_TO_ACT"

    def test_reset_retries(self) -> None:
        validator = ConstraintValidator()
        for _ in range(2):
            validator.check_retries("agent-3")
        validator.reset_retries("agent-3")
        can_retry, status = validator.check_retries("agent-3")
        assert can_retry
        assert status == ""
