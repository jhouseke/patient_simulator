"""Declarative mapping from Pulse outputs to our canonical metric_codes + units.

Each ScalarRequest records: our metric_code, the Pulse physiology data name,
a key naming the Pulse unit, and a converter to our canonical unit (see
engine/pdes/snapshot.py::_metric_unit for the canonical units).

`pulse_unit` is a *string key*, not a Pulse unit object, so this module stays
importable without Pulse installed. PulseAdapter resolves the key to the real
unit object (FrequencyUnit.Per_min, PressureUnit.mmHg, ...) when it builds the
SEDataRequest list. See docs/pulse_api_notes.md §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


def _identity(x: float) -> float:
    return x


@dataclass(frozen=True)
class ScalarRequest:
    metric_code: str
    pulse_name: str          # Pulse physiology data name, e.g. "HeartRate"
    pulse_unit: str          # key into PULSE_UNIT_KEYS, e.g. "mmHg"
    convert: Callable[[float], float] = field(default=_identity)


# Unit keys we know how to resolve to Pulse unit objects. PulseAdapter owns the
# actual mapping (it needs `import pulse`); this set exists so a typo fails in
# the fast unit suite rather than at engine-initialization time.
PULSE_UNIT_KEYS: frozenset[str] = frozenset(
    {
        "1/min",
        "mmHg",
        "degC",
        "L/min",
        "mL",
        "mV",
        "unitless",
    }
)


SCALAR_REQUESTS: list[ScalarRequest] = [
    ScalarRequest("hr", "HeartRate", "1/min"),
    ScalarRequest("map", "MeanArterialPressure", "mmHg"),
    ScalarRequest("sbp", "SystolicArterialPressure", "mmHg"),
    ScalarRequest("dbp", "DiastolicArterialPressure", "mmHg"),
    ScalarRequest("rr", "RespirationRate", "1/min"),
    # Pulse reports OxygenSaturation as a unitless fraction (0..1).
    ScalarRequest("spo2", "OxygenSaturation", "unitless", convert=lambda x: x * 100.0),
    ScalarRequest("etco2", "EndTidalCarbonDioxidePressure", "mmHg"),
    # CONFIRM at Task 0.2 Step 6: Pulse's own default set uses SkinTemperature.
    ScalarRequest("temp_c", "CoreTemperature", "degC"),
    # Requested directly in L/min — no mL/min conversion needed.
    ScalarRequest("co", "CardiacOutput", "L/min"),
    # CONFIRM at Task 0.2 Step 6.
    ScalarRequest("cvp", "CentralVenousPressure", "mmHg"),
    ScalarRequest("blood_volume_ml", "BloodVolume", "mL"),
]
