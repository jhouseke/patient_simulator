"""Intent -> Pulse action translation (Task 1.4, extended in Task 3.2).

Pure translation logic — no Pulse required, so this stays in the fast suite.
"""

import pytest

from engine.physiology.actions import (
    Extubation,
    Hemorrhage,
    InhaledAnesthetic,
    Intubation,
    SubstanceBolus,
    SubstanceInfusion,
    VentilatorSettings,
    intent_to_pulse_action,
)


def test_administer_med_maps_to_substance_bolus():
    action = intent_to_pulse_action(
        action_type="ADMINISTER_MED",
        payload={"drug_name": "Propofol", "dose": 100.0, "unit": "mg", "route": "IV_bolus"},
    )
    assert isinstance(action, SubstanceBolus)
    assert action.substance == "Propofol"
    assert action.dose_mg == 100.0
    # Pulse takes a VOLUME + concentration, not a mass: 100 mg at 10 mg/mL = 10 mL
    assert action.concentration_mg_per_ml == pytest.approx(10.0)
    assert action.volume_ml == pytest.approx(10.0)


def test_bolus_volume_uses_per_drug_concentration():
    """Fentanyl is 50 mcg/mL = 0.05 mg/mL, so 0.1 mg is 2 mL, not 0.01 mL."""
    action = intent_to_pulse_action(
        "ADMINISTER_MED",
        {"drug_name": "Fentanyl", "dose": 0.1, "unit": "mg", "route": "IV_bolus"},
    )
    assert action.concentration_mg_per_ml == pytest.approx(0.05)
    assert action.volume_ml == pytest.approx(2.0)


def test_explicit_concentration_overrides_table():
    action = intent_to_pulse_action(
        "ADMINISTER_MED",
        {
            "drug_name": "Propofol",
            "dose": 100.0,
            "route": "IV_bolus",
            "concentration_mg_per_ml": 20.0,
        },
    )
    assert action.volume_ml == pytest.approx(5.0)


def test_infusion_converts_rate_to_volume_per_time():
    action = intent_to_pulse_action(
        "ADMINISTER_MED",
        {"drug_name": "Propofol", "rate": 20.0, "route": "IV_infusion"},
    )
    assert isinstance(action, SubstanceInfusion)
    assert action.rate_mg_per_min == pytest.approx(20.0)
    assert action.rate_ml_per_min == pytest.approx(2.0)


def test_drug_aliases_are_canonicalized():
    action = intent_to_pulse_action(
        "ADMINISTER_MED",
        {"drug_name": "Levophed", "rate": 0.1, "route": "IV_infusion"},
    )
    assert action.substance == "Norepinephrine"


def test_adjust_vent_maps_to_ventilator_settings():
    action = intent_to_pulse_action(
        "ADJUST_VENT", {"peep": 8, "tidal_volume": 450, "rate": 14, "fio2": 0.6}
    )
    assert isinstance(action, VentilatorSettings)
    assert action.peep_cmh2o == 8.0
    assert action.tidal_volume_ml == 450.0
    assert action.rate_per_min == 14.0
    assert action.fio2 == pytest.approx(0.6)


def test_intubate_and_extubate_map():
    assert isinstance(intent_to_pulse_action("INTUBATE", {"airway": "ETT"}), Intubation)
    assert isinstance(intent_to_pulse_action("EXTUBATE", {}), Extubation)


def test_intubation_carries_pulse_type():
    """Pulse's eIntubationType supports misadventures, not just Tracheal."""
    assert intent_to_pulse_action("INTUBATE", {}).pulse_type == "Tracheal"
    eso = intent_to_pulse_action("INTUBATE", {"pulse_type": "Esophageal"})
    assert eso.pulse_type == "Esophageal"
    with pytest.raises(ValueError):
        intent_to_pulse_action("INTUBATE", {"pulse_type": "Nonsense"})


def test_inhaled_anesthetic_maps():
    action = intent_to_pulse_action(
        "ADMINISTER_MED", {"drug_name": "Sevoflurane", "route": "inhaled", "mac": 1.0}
    )
    assert isinstance(action, InhaledAnesthetic)
    assert action.mac == 1.0


def test_hemorrhage_event_maps():
    action = intent_to_pulse_action(
        "INJECT_EVENT",
        {"disease_type": "OCCULT_HEMORRHAGE", "params": {"severity": 0.7}},
    )
    assert isinstance(action, Hemorrhage)
    assert action.severity == pytest.approx(0.7)
    # Pulse compartments are anatomic; "Abdomen" is not one of them.
    assert action.compartment == "Liver"


def test_hemorrhage_rejects_unknown_compartment():
    with pytest.raises(ValueError):
        intent_to_pulse_action(
            "INJECT_EVENT",
            {
                "disease_type": "OCCULT_HEMORRHAGE",
                "params": {"severity": 0.5, "compartment": "Abdomen"},
            },
        )


def test_unknown_action_returns_none():
    assert intent_to_pulse_action("CALL_ATTENDING", {}) is None
    assert intent_to_pulse_action("CHECK_VITALS", {}) is None


def test_unmapped_event_returns_none_and_is_reported():
    """Risk 2: never silently no-op an event — unmapped types must be visible."""
    from engine.physiology.actions import unmapped_events

    unmapped_events.clear()
    assert intent_to_pulse_action("INJECT_EVENT", {"disease_type": "SEPSIS"}) is None
    assert "SEPSIS" in unmapped_events
