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

from dataclasses import asdict, dataclass, field
import base64
import json
import math

import numpy as np

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
        self._pending_wave_seconds = 0.0
        self._drain_start_spare = 0.0

        unknown = set(waveform_channels) - set(WAVEFORM_CHANNELS)
        if unknown:
            raise ValueError(f"unknown waveform channels: {sorted(unknown)}")
        self.waveform_channels = tuple(waveform_channels)
        self._capture = _WaveformCapture()
        self._wave_time_s = 0.0
        # First sample time for each nonempty channel in the most recent drain.
        self.last_waveform_start_s: dict[str, float] = {}
        self._previous_capture = None
        self._wave_samples = {c: 0 for c in SYNTHESIZED_CHANNELS}
        # ECG R peak is phase zero; pressure upstroke begins at the same
        # phase, with template peaks delayed by 0.13 (ART) and 0.20 (PLETH)
        # cycles. These fixed delays are illustrative, not measured transit times.
        self._cardiac_phase = 0.5
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
        steps = int(math.floor(total / self._timestep_s + 1e-12))
        self._spare_time_s = max(0.0, total - steps * self._timestep_s)
        for _ in range(steps):
            if not self._engine.advance_time():
                raise RuntimeError("Pulse failed to advance")
            self._capture_timestep()
        self._pending_wave_seconds += seconds

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
        """Drain completed Pulse timesteps for the requested barrier interval.

        ``seconds`` is the sum of advance durations since the previous drain,
        not necessarily a multiple of the Pulse timestep. A partial timestep
        remains in the advance carry; its samples are emitted on a later drain.
        Native samples have timestamps k * timestep and synthesized samples
        have timestamps k / sample_rate (k starts at 1), measured from the
        first advance. A drain returns the samples through the last completed
        Pulse timestep, which can precede the barrier by less than one timestep.
        Consequently, a callback's barrier time is not the timestamp of its
        first sample. After each successful drain, ``last_waveform_start_s``
        maps each nonempty channel to its first sample's elapsed time; empty
        channels are omitted. It is empty before the first drain.

        Pulse values are linearly interpolated at monitor sample times. Phase
        is the integral of that same piecewise-linear HR curve at every rate.
        Templates use fixed cycle-relative delays, not physiological conduction
        or patient-specific pulse transit models.
        """
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be finite and nonnegative")
        count = len(self._capture.hr)
        duration = count * self._timestep_s
        if not math.isclose(seconds, self._pending_wave_seconds, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"requested {seconds}s but advanced {self._pending_wave_seconds}s since drain")
        expected_duration = self._drain_start_spare + self._pending_wave_seconds - self._spare_time_s
        if not math.isclose(duration, expected_duration, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"captured {duration}s but expected {expected_duration}s")
        for name in ("sbp", "dbp", "spo2"):
            if len(getattr(self._capture, name)) != count:
                raise ValueError(f"incomplete {name} capture")
        for channel in self.waveform_channels:
            if channel in NATIVE_WAVEFORM_REQUESTS and len(self._capture.native.get(channel, [])) != count:
                raise ValueError(f"incomplete {channel} capture")
        out = {c: list(self._capture.native.get(c, [])) for c in self.waveform_channels
               if c in NATIVE_WAVEFORM_REQUESTS}
        starts = {c: self._wave_time_s + self._timestep_s for c, values in out.items() if values}
        if not count:
            self.last_waveform_start_s = starts
            out.update({c: [] for c in self.waveform_channels if c in SYNTHESIZED_CHANNELS})
            self._pending_wave_seconds = 0.0
            self._drain_start_spare = self._spare_time_s
            return out

        # Captures are end-of-timestep values. Use the previous drain's last
        # value at the boundary (or hold the first value for the first drain).
        x = np.arange(count + 1) * self._timestep_s
        previous = self._previous_capture or {name: getattr(self._capture, name)[0] for name in ("hr", "sbp", "dbp", "spo2")}
        hr = np.maximum(np.r_[previous["hr"], self._capture.hr], 1.0)
        integral = np.r_[0.0, np.cumsum((hr[:-1] + hr[1:]) * 0.5 * self._timestep_s / 60.0)]
        end_time = self._wave_time_s + duration
        for channel in self.waveform_channels:
            if channel not in SYNTHESIZED_CHANNELS:
                continue
            fs = WAVEFORM_RATES[channel]
            end_count = int(math.floor(end_time * fs + 1e-9))
            if end_count > self._wave_samples[channel]:
                starts[channel] = (self._wave_samples[channel] + 1) / fs
            times = np.arange(self._wave_samples[channel] + 1, end_count + 1) / fs - self._wave_time_s
            times = np.clip(times, 0.0, duration)
            idx = np.minimum(np.searchsorted(x, times, side="right") - 1, count - 1)
            dt = times - x[idx]
            slope = (hr[idx + 1] - hr[idx]) / self._timestep_s
            phases = (self._cardiac_phase + integral[idx] + (hr[idx] * dt + 0.5 * slope * dt**2) / 60.0) % 1.0
            def interp(name):
                return np.interp(times, x, np.r_[previous[name], getattr(self._capture, name)])
            rates = interp("hr")
            if channel == "ECG_II":
                values = self._ecg.generate(duration, rates, phase_cycles=phases)
            elif channel == "ART":
                values = self._pressure.generate_art(duration, rates, interp("sbp"), interp("dbp"), phase_cycles=phases)
            else:
                values = self._pressure.generate_pleth(duration, rates, interp("spo2"), phase_cycles=phases)
            out[channel] = values.tolist()
            self._wave_samples[channel] = end_count
        self._cardiac_phase = float((self._cardiac_phase + integral[-1]) % 1.0)
        self._wave_time_s = end_time
        self._previous_capture = {name: getattr(self._capture, name)[-1] for name in ("hr", "sbp", "dbp", "spo2")}
        self._capture.clear()
        self.last_waveform_start_s = starts
        self._pending_wave_seconds = 0.0
        self._drain_start_spare = self._spare_time_s
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
        state = state if isinstance(state, bytes) else str(state).encode("utf-8")
        snapshot = {"version": 1, "pulse": base64.b64encode(state).decode("ascii"),
                    "channels": self.waveform_channels, "timestep": self._timestep_s,
                    "spare": self._spare_time_s, "capture": asdict(self._capture),
                    "pending_wave_seconds": self._pending_wave_seconds,
                    "drain_start_spare": self._drain_start_spare,
                    "engine_spare": getattr(self._engine, "_spare_time_s", None),
                    "wave_time": self._wave_time_s, "wave_samples": self._wave_samples,
                    "previous_capture": self._previous_capture,
                    "last_waveform_start_s": self.last_waveform_start_s,
                    "phase": self._cardiac_phase, "ecg": self._ecg.snapshot_state(),
                    "pressure": self._pressure.snapshot_state()}
        return json.dumps(snapshot, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def load_state(self, blob: bytes) -> None:
        from pulse.cdm.engine import eSerializationFormat

        self._require_engine()
        try:
            state = json.loads(blob)
            if state["version"] != 1 or tuple(state["channels"]) != self.waveform_channels:
                raise ValueError("incompatible snapshot version or waveform channels")
            pulse = base64.b64decode(state["pulse"], validate=True)
            capture = _WaveformCapture(**state["capture"])
            # Validate synthesizer state before changing the live engine.
            ecg = ECGSynthesizer(self.seed, self._ecg.fs, self._ecg.noise_mv, self._ecg.wander_mv)
            pressure = PressureSynthesizer(self.seed + 1, self._pressure.fs, self._pressure.art_noise_mmhg, self._pressure.pleth_noise)
            ecg.restore_state(state["ecg"])
            pressure.restore_state(state["pressure"])
            timestep, spare = float(state["timestep"]), float(state["spare"])
            wave_time, phase = float(state["wave_time"]), float(state["phase"])
            pending = float(state["pending_wave_seconds"])
            drain_spare = float(state["drain_start_spare"])
            engine_spare = state["engine_spare"]
            if engine_spare is not None:
                engine_spare = float(engine_spare)
                if not math.isfinite(engine_spare) or engine_spare < 0:
                    raise ValueError("invalid engine spare time")
            samples = {c: int(state["wave_samples"][c]) for c in SYNTHESIZED_CHANNELS}
            starts = {c: float(value) for c, value in state["last_waveform_start_s"].items()}
            if any(c not in self.waveform_channels or not math.isfinite(value) or value < 0
                   for c, value in starts.items()):
                raise ValueError("invalid waveform start times")
            previous = state["previous_capture"]
            if previous is not None:
                previous = {name: float(previous[name]) for name in ("hr", "sbp", "dbp", "spo2")}
                if not all(math.isfinite(v) for v in previous.values()):
                    raise ValueError("invalid previous capture")
            if not all(math.isfinite(v) for v in (timestep, spare, wave_time, phase, pending, drain_spare)) or timestep <= 0 or not 0 <= spare < timestep or not 0 <= drain_spare < timestep or pending < 0 or wave_time < 0 or not 0 <= phase < 1 or any(v < 0 for v in samples.values()):
                raise ValueError("invalid timing state")
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            raise ValueError("invalid adapter snapshot") from exc
        if not self._engine.serialize_from_string(pulse, self._drm, eSerializationFormat.BINARY):
            raise RuntimeError("Pulse refused to load state")
        if engine_spare is not None:
            self._engine._spare_time_s = engine_spare
        self._pending_wave_seconds, self._drain_start_spare = pending, drain_spare
        self._capture, self._ecg, self._pressure = capture, ecg, pressure
        self._previous_capture = previous
        self.last_waveform_start_s = starts
        self._timestep_s, self._spare_time_s = timestep, spare
        self._wave_time_s, self._cardiac_phase, self._wave_samples = wave_time, phase, samples

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_engine(self):
        if self._engine is None:
            raise RuntimeError("PulseAdapter.initialize() has not been called")
        return self._engine

    def _capture_timestep(self) -> None:
        results = self._engine.pull_data()

        def _value(code: str) -> float:
            idx = self._index.get(code)
            if idx is None or idx >= len(results):
                raise RuntimeError(f"missing Pulse capture value: {code}")
            value = float(results[idx])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite Pulse capture value: {code}")
            return value

        # Read all values first so a missing field cannot leave partial buffers.
        hr, sbp, dbp, spo2 = (_value(c) for c in ("hr", "sbp", "dbp", "spo2"))
        native = {c: _value(f"__wave_{c}") for c in self.waveform_channels
                  if c in NATIVE_WAVEFORM_REQUESTS}
        self._capture.hr.append(hr)
        self._capture.sbp.append(sbp)
        self._capture.dbp.append(dbp)
        self._capture.spo2.append(spo2 * 100.0)
        for channel, value in native.items():
            self._capture.native.setdefault(channel, []).append(value)

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
