"""Adapter around the Pulse Physiology Engine.

This is the ONLY module permitted to import `pulse`. Everything else in the
codebase depends on PulseAdapter's stable interface so that a Pulse version
bump (or a swap to a different physiology engine) is contained here.

Written against Pulse REL_4_3_2; see docs/pulse_api_notes.md for the API
reference this implementation follows and for the items still marked CONFIRM.
Pulse is not pip-installable, so `import pulse` is deferred to initialize():
the module stays importable (and the dataclasses stay testable) on machines
without a compiled engine.

Two Pulse behaviors shape this design:

1. `advance_time_s(n)` pulls data only ONCE, at the end — every intermediate
   timestep is discarded. Waveform capture therefore drives the single-timestep
   `advance_time()` instead, and `advance()` picks the mode based on whether
   waveform channels are enabled.
2. `pull_data()` returns a flat list whose FIRST element is simulation time;
   our requested values start at index 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from engine.physiology import actions as act
from engine.physiology.data_requests import SCALAR_REQUESTS
from engine.physiology.ecg_synth import ECGSynthesizer
from engine.physiology.pressure_synth import PressureSynthesizer
from engine.physiology.waveforms import WAVEFORM_CHANNELS, WAVEFORM_RATES

logger = structlog.get_logger()

#: Pulse's fixed engine timestep. Read back from the engine in initialize();
#: this is the documented nominal value used until then.
NOMINAL_TIMESTEP_S = 0.02
PULSE_NATIVE_RATE_HZ = 1.0 / NOMINAL_TIMESTEP_S

#: Pulse-native waveform channels -> the physiology/compartment request feeding them.
#: CONFIRM these request names against a live build (Task 0.2 Step 6).
NATIVE_WAVEFORM_REQUESTS = {
    "CO2": ("gas_compartment_substance", ("Carina", "CarbonDioxide", "PartialPressure"), "mmHg"),
    "AWP": ("physiology", ("AirwayPressure",), "cmH2O"),
    "FLOW": ("physiology", ("TotalPulmonaryVentilation",), "mL/s"),
    "VOL": ("physiology", ("TotalLungVolume",), "mL"),
}

SYNTHESIZED_CHANNELS = ("ART", "PLETH", "ECG_II")


@dataclass
class PatientConfig:
    sex: str            # "MALE" | "FEMALE"
    age_yr: int
    height_cm: float
    weight_kg: float
    hr_baseline: float | None = None     # optional overrides for calibration
    map_baseline: float | None = None
    name: str = "SimPatient"


@dataclass
class _WaveformCapture:
    """Per-timestep accumulation between sample_waveforms() drains."""

    native: dict[str, list[float]] = field(default_factory=dict)
    hr: list[float] = field(default_factory=list)
    sbp: list[float] = field(default_factory=list)
    dbp: list[float] = field(default_factory=list)
    spo2: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.native.clear()
        self.hr.clear()
        self.sbp.clear()
        self.dbp.clear()
        self.spo2.clear()


class PulseAdapter:
    """Stable interface over the Pulse engine.

    The engine drives this; agents never touch it directly.
    """

    def __init__(
        self,
        seed: int,
        patient_cfg: PatientConfig,
        data_root_dir: str = "/pulse/bin/",
        waveform_channels: tuple[str, ...] = (),
    ) -> None:
        self.seed = seed
        self.cfg = patient_cfg
        self.data_root_dir = data_root_dir
        self._engine = None
        self._drm = None
        self._index: dict[str, int] = {}
        self._timestep_s = NOMINAL_TIMESTEP_S
        self._spare_time_s = 0.0

        unknown = set(waveform_channels) - set(WAVEFORM_CHANNELS)
        if unknown:
            raise ValueError(f"unknown waveform channels: {sorted(unknown)}")
        self.waveform_channels = tuple(waveform_channels)
        self._capture = _WaveformCapture()
        # Seeded per-patient so runs are reproducible; offset so the two
        # generators don't share a noise stream.
        self._ecg = ECGSynthesizer(seed=seed, sample_rate_hz=WAVEFORM_RATES["ECG_II"])
        self._pressure = PressureSynthesizer(
            seed=seed + 1, sample_rate_hz=WAVEFORM_RATES["ART"]
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the engine and stabilize the patient to a resting state."""
        from pulse.cdm.engine import SEDataRequest, SEDataRequestManager
        from pulse.cdm.patient import SEPatient, SEPatientConfiguration
        from pulse.cdm.scalars import LengthUnit, MassUnit, TimeUnit
        from pulse.engine.PulseEngine import PulseEngine, eModelType

        self._engine = PulseEngine(eModelType.HumanAdultWholeBody, self.data_root_dir)
        self._engine.log_to_console(False)

        patient = SEPatient()
        patient.set_name(self.cfg.name)
        patient.set_sex(self._sex_enum(self.cfg.sex))
        patient.get_age().set_value(self.cfg.age_yr, TimeUnit.yr)
        patient.get_height().set_value(self.cfg.height_cm, LengthUnit.cm)
        patient.get_weight().set_value(self.cfg.weight_kg, MassUnit.kg)
        if self.cfg.hr_baseline is not None:
            from pulse.cdm.scalars import FrequencyUnit

            patient.get_heart_rate_baseline().set_value(
                self.cfg.hr_baseline, FrequencyUnit.Per_min
            )
        if self.cfg.map_baseline is not None:
            from pulse.cdm.scalars import PressureUnit

            patient.get_mean_arterial_pressure_baseline().set_value(
                self.cfg.map_baseline, PressureUnit.mmHg
            )

        patient_cfg = SEPatientConfiguration()
        patient_cfg.set_data_root_dir(self.data_root_dir)
        patient_cfg.set_patient(patient)

        requests = [self._scalar_request(SEDataRequest, r) for r in SCALAR_REQUESTS]
        # pull_data() puts simulation time first, so our values start at 1.
        self._index = {r.metric_code: i + 1 for i, r in enumerate(SCALAR_REQUESTS)}
        for channel in self.waveform_channels:
            if channel in NATIVE_WAVEFORM_REQUESTS:
                requests.append(self._native_waveform_request(SEDataRequest, channel))
                self._index[f"__wave_{channel}"] = len(requests)

        self._drm = SEDataRequestManager()
        self._drm.set_data_requests(requests)

        if not self._engine.initialize_engine(patient_cfg, self._drm):
            raise RuntimeError("Pulse failed to initialize/stabilize the patient")
        self._timestep_s = self._engine._dt_s or NOMINAL_TIMESTEP_S
        logger.info(
            "pulse_initialized",
            seed=self.seed,
            timestep_s=self._timestep_s,
            waveforms=list(self.waveform_channels),
        )

    def close(self) -> None:
        if self._engine is not None:
            self._engine.clear()
        self._engine = None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def advance(self, seconds: float) -> None:
        """Run physiology forward.

        With waveform channels enabled this steps one Pulse timestep at a time
        so no intermediate sample is lost; otherwise it uses Pulse's own
        advance_time_s, which is cheaper.
        """
        if seconds <= 0:
            return
        self._require_engine()
        if not self.waveform_channels:
            if not self._engine.advance_time_s(seconds):
                raise RuntimeError("Pulse failed to advance")
            return

        # Mirror Pulse's spare-time carry so repeated calls don't drift.
        total = seconds + self._spare_time_s
        steps = int(total / self._timestep_s)
        self._spare_time_s = total - steps * self._timestep_s
        for _ in range(steps):
            if not self._engine.advance_time():
                raise RuntimeError("Pulse failed to advance")
            self._capture_timestep()

    def apply_action(self, action) -> None:
        """Translate one of our PulseAction dataclasses and hand it to Pulse."""
        self._require_engine()
        pulse_action = self._to_pulse(action)
        if pulse_action is None:
            logger.warning("action_has_no_pulse_equivalent", action=type(action).__name__)
            return
        self._engine.process_action(pulse_action)

    def sample_scalars(self) -> dict[str, float]:
        """Current value for each scalar request, in canonical units."""
        self._require_engine()
        results = self._engine.pull_data()
        out: dict[str, float] = {}
        for req in SCALAR_REQUESTS:
            idx = self._index[req.metric_code]
            if idx < len(results):
                out[req.metric_code] = req.convert(float(results[idx]))
        return out

    def sample_waveforms(self, seconds: float) -> dict[str, list[float]]:
        """Drain per-channel samples accumulated over the last `seconds`.

        Native channels come back at Pulse's 50 Hz timestep. ART/PLETH/ECG_II
        are synthesized at monitor rates from the beat-level series captured
        over the same interval (Arch Decision 9).
        """
        out: dict[str, list[float]] = {}
        for channel in self.waveform_channels:
            if channel in NATIVE_WAVEFORM_REQUESTS:
                out[channel] = list(self._capture.native.get(channel, []))

        if any(c in self.waveform_channels for c in SYNTHESIZED_CHANNELS):
            hr = self._capture.hr or [60.0]
            mean_hr = sum(hr) / len(hr)
            if "ECG_II" in self.waveform_channels:
                out["ECG_II"] = self._ecg.generate(seconds, mean_hr).tolist()
            if "ART" in self.waveform_channels:
                sbp = self._capture.sbp or [120.0]
                dbp = self._capture.dbp or [80.0]
                out["ART"] = self._pressure.generate_art(
                    seconds, mean_hr, sum(sbp) / len(sbp), sum(dbp) / len(dbp)
                ).tolist()
            if "PLETH" in self.waveform_channels:
                spo2 = self._capture.spo2 or [98.0]
                out["PLETH"] = self._pressure.generate_pleth(
                    seconds, mean_hr, sum(spo2) / len(spo2)
                ).tolist()

        self._capture.clear()
        return out

    # ------------------------------------------------------------------
    # State (rollback)
    # ------------------------------------------------------------------

    def save_state(self) -> bytes:
        from pulse.cdm.engine import eSerializationFormat

        self._require_engine()
        state = self._engine.serialize_to_string(eSerializationFormat.BINARY)
        if state is None:
            raise RuntimeError("Pulse refused to serialize state")
        return state if isinstance(state, bytes) else str(state).encode("utf-8")

    def load_state(self, blob: bytes) -> None:
        from pulse.cdm.engine import eSerializationFormat

        self._require_engine()
        if not self._engine.serialize_from_string(
            blob, self._drm, eSerializationFormat.BINARY
        ):
            raise RuntimeError("Pulse refused to load state")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_engine(self):
        if self._engine is None:
            raise RuntimeError("PulseAdapter.initialize() has not been called")
        return self._engine

    def _capture_timestep(self) -> None:
        results = self._engine.pull_data()

        def _value(code: str, default: float) -> float:
            idx = self._index.get(code)
            if idx is None or idx >= len(results):
                return default
            return float(results[idx])

        self._capture.hr.append(_value("hr", 60.0))
        self._capture.sbp.append(_value("sbp", 120.0))
        self._capture.dbp.append(_value("dbp", 80.0))
        # spo2 is stored unitless in Pulse; convert to % for the synthesizer.
        self._capture.spo2.append(_value("spo2", 0.98) * 100.0)

        for channel in self.waveform_channels:
            if channel not in NATIVE_WAVEFORM_REQUESTS:
                continue
            idx = self._index.get(f"__wave_{channel}")
            if idx is not None and idx < len(results):
                self._capture.native.setdefault(channel, []).append(float(results[idx]))

    @staticmethod
    def _sex_enum(sex: str):
        from pulse.cdm.patient import eSex

        normalized = sex.strip().upper()
        if normalized in ("MALE", "M"):
            return eSex.Male
        if normalized in ("FEMALE", "F"):
            return eSex.Female
        raise ValueError(f"unknown sex {sex!r}; expected MALE or FEMALE")

    @staticmethod
    def _unit(key: str):
        """Resolve our unit-key strings to Pulse unit objects."""
        from pulse.cdm.scalars import (
            FrequencyUnit,
            PressureUnit,
            TemperatureUnit,
            VolumePerTimeUnit,
            VolumeUnit,
        )

        return {
            "1/min": FrequencyUnit.Per_min,
            "mmHg": PressureUnit.mmHg,
            "degC": TemperatureUnit.C,
            "L/min": VolumePerTimeUnit.L_Per_min,
            "mL": VolumeUnit.mL,
            "unitless": None,
        }[key]

    def _scalar_request(self, SEDataRequest, req):
        unit = self._unit(req.pulse_unit)
        if unit is None:
            return SEDataRequest.create_physiology_request(req.pulse_name)
        return SEDataRequest.create_physiology_request(req.pulse_name, unit=unit)

    def _native_waveform_request(self, SEDataRequest, channel: str):
        from pulse.cdm.scalars import PressureUnit, VolumePerTimeUnit, VolumeUnit

        kind, args, unit_key = NATIVE_WAVEFORM_REQUESTS[channel]
        units = {
            "mmHg": PressureUnit.mmHg,
            "cmH2O": PressureUnit.cmH2O,
            "mL/s": VolumePerTimeUnit.mL_Per_s,
            "mL": VolumeUnit.mL,
        }
        if kind == "gas_compartment_substance":
            return SEDataRequest.create_gas_compartment_substance_request(
                *args, unit=units[unit_key]
            )
        return SEDataRequest.create_physiology_request(*args, unit=units[unit_key])

    def _to_pulse(self, action):
        """Map our PulseAction dataclasses onto Pulse SE* action objects."""
        from pulse.cdm.patient_actions import (
            SEHemorrhage,
            SEIntubation,
            SESubstanceBolus,
            SESubstanceInfusion,
            eHemorrhage_Compartment,
            eHemorrhage_Type,
            eIntubationType,
            eSubstance_Administration,
        )
        from pulse.cdm.scalars import (
            MassPerVolumeUnit,
            VolumePerTimeUnit,
            VolumeUnit,
        )

        if isinstance(action, act.SubstanceBolus):
            bolus = SESubstanceBolus()
            bolus.set_substance(action.substance)
            bolus.set_admin_route(eSubstance_Administration.Intravenous)
            # Pulse takes a VOLUME plus a concentration, not a mass.
            bolus.get_dose().set_value(action.volume_ml, VolumeUnit.mL)
            bolus.get_concentration().set_value(
                action.concentration_mg_per_ml, MassPerVolumeUnit.mg_Per_mL
            )
            return bolus

        if isinstance(action, act.SubstanceInfusion):
            infusion = SESubstanceInfusion()
            infusion.set_substance(action.substance)
            infusion.get_rate().set_value(
                action.rate_ml_per_min, VolumePerTimeUnit.mL_Per_min
            )
            infusion.get_concentration().set_value(
                action.concentration_mg_per_ml, MassPerVolumeUnit.mg_Per_mL
            )
            return infusion

        if isinstance(action, act.Intubation):
            intubation = SEIntubation()
            intubation.set_type(getattr(eIntubationType, action.pulse_type))
            return intubation

        if isinstance(action, act.Extubation):
            extubation = SEIntubation()
            extubation.set_type(eIntubationType.Off)
            return extubation

        if isinstance(action, act.Hemorrhage):
            hemorrhage = SEHemorrhage()
            hemorrhage.set_type(
                eHemorrhage_Type.Internal if action.internal else eHemorrhage_Type.External
            )
            hemorrhage.set_compartment(
                getattr(eHemorrhage_Compartment, action.compartment).name
            )
            hemorrhage.get_severity().set_value(action.severity)
            return hemorrhage

        if isinstance(action, act.VentilatorSettings):
            # CONFIRM: SEMechanicalVentilation vs SERespiratoryMechanicsConfiguration
            # is open question 3 in docs/pulse_api_notes.md §6.
            return self._ventilator_action(action)

        if isinstance(action, act.InhaledAnesthetic):
            # Pulse 4.x appears to have no anesthesia machine — see §6 open
            # question 1. Scenarios default to TIVA for this reason.
            logger.error(
                "inhaled_anesthetic_unsupported",
                substance=action.substance,
                mac=action.mac,
            )
            return None

        return None

    def _ventilator_action(self, action):
        from pulse.cdm.mechanical_ventilator_actions import (
            SEMechanicalVentilatorContinuousPositiveAirwayPressure,
        )
        from pulse.cdm.scalars import PressureUnit

        # Minimal, confirmable configuration: PEEP + FiO2. Tidal volume and
        # rate need the mode-specific action chosen in §6 open question 3.
        vent = SEMechanicalVentilatorContinuousPositiveAirwayPressure()
        vent.get_positive_end_expired_pressure().set_value(
            action.peep_cmh2o, PressureUnit.cmH2O
        )
        vent.get_fraction_inspired_oxygen().set_value(action.fio2)
        return vent
