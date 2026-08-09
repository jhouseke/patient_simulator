# Pulse Physiology Engine — API & Build Notes

> **Status:** captured 2026-08-08 by reading the Pulse source at tag `REL_4_3_2`
> (gitlab.kitware.com/physiology/engine) — **not** yet verified against a local build.
> This is the authoritative reference for `src/engine/physiology/pulse_adapter.py`
> (Phase 0 Task 0.2 Step 6 deliverable). Re-verify each `CONFIRM` item once the
> engine image builds.

## 1. Distribution: source build only

**Pulse is NOT on PyPI.** A full scan of the PyPI simple index found no Kitware
physiology package (`pulse-physiology-engine`, `PulsePhysiologyEngine`,
`pulse-cdm`, … all absent). It must be compiled from source.

- Repo: `https://gitlab.kitware.com/physiology/engine.git`
- **Tag format is `REL_<major>_<minor>_<patch>`**, e.g. `REL_4_3_2` — *not* `v4.4.0`.
- Latest release tags: `REL_4_3_2` (2025-08-12), `REL_4_3_1`, `REL_4_3_0`.

### Build recipe (from Pulse's own `Dockerfile` at `REL_4_3_2`)

CMake **superbuild from the repo root** (it downloads and builds its own
dependencies — Eigen3, protobuf, abseil, pybind11):

```bash
cmake -DPulse_PYTHON_API=ON \
      -DPulse_JAVA_API=OFF \
      -DCMAKE_INSTALL_PREFIX=/pulse \
      -DCMAKE_BUILD_TYPE:STRING=Release \
      /source
make -j"$(nproc)"
```

- The correct flag is **`Pulse_PYTHON_API`** (ON by default). The names
  `Pulse_BUILD_PYTHON` / `Pulse_PYTHON_BINDINGS` do not exist.
- `Pulse_JAVA_API` defaults **ON** and requires a JDK — set it `OFF` unless you
  want to install Corretto in the image.
- Configure from the **repo root**, not `src/`.
- Pulse's upstream Dockerfile uses clang + libc++ on `debian:bullseye`. GCC 9+ is
  also supported per the ReadMe. `CONFIRM` which toolchain builds cleanly for us.
- Pulse's own `requirements.txt`: `numpy pandas matplotlib openpyxl pycel`.

### Runtime path

```
ENV PYTHONPATH=/pulse/bin:/pulse/python
```

**Both** directories are required — `/pulse/python` holds the `pulse` package,
`/pulse/bin` holds the compiled `PyPulse` extension plus the data files the
engine loads at runtime (`data_root_dir`).

### Python version ceiling: 3.13

`cmake/external/External_pybind11.cmake` pins **pybind11 v2.13.1**, which supports
Python ≤ 3.13. **Python 3.14 will not build PyPulse.** This is why the project
targets 3.13 (also the ceiling for torch / `sentence-transformers`, Phase 4).

## 2. Module layout

```
pulse/
├── cdm/                 # common data model
│   ├── engine.py            SEAction, SEDataRequest, SEDataRequestManager,
│   │                        eSerializationFormat, IEventHandler, ILoggerForward
│   ├── patient.py           SEPatient, SEPatientConfiguration
│   ├── patient_actions.py   all SEPatientAction subclasses (see §4)
│   ├── scalars.py           unit objects (see §3)
│   ├── mechanical_ventilator_actions.py
│   ├── equipment_actions.py, environment_actions.py, ecmo_actions.py
│   └── io/                  serialize_*_to_string / _from_string
└── engine/
    ├── PulseEngine.py       PulseEngine, eModelType, version(), hash()
    ├── PulseEnginePool.py   multi-engine pool (relevant to Phase 5 scale-out)
    ├── PulseConfiguration.py
    └── PulseScenarioExec.py
```

`import PyPulse` is the compiled extension; `pulse.engine.PulseEngine` wraps it.
Our adapter imports only the `pulse.*` Python layer.

## 3. `PulseEngine` — the methods we use

```python
from pulse.engine.PulseEngine import PulseEngine, eModelType
from pulse.cdm.engine import (SEDataRequest, SEDataRequestManager,
                              eSerializationFormat)
from pulse.cdm.patient import SEPatient, SEPatientConfiguration
from pulse.cdm.scalars import (FrequencyUnit, PressureUnit, TemperatureUnit,
                               VolumeUnit, VolumePerTimeUnit,
                               ElectricPotentialUnit, MassPerVolumeUnit,
                               LengthUnit, MassUnit, TimeUnit)

engine = PulseEngine(eModelType.HumanAdultWholeBody, data_root_dir="/pulse/bin/")
engine.initialize_engine(patient_configuration, data_request_mgr)  # -> bool
engine.advance_time_s(30.0)          # -> bool
engine.advance_time()                # ONE timestep -> bool
engine.pull_data()                   # -> list[float]
engine.process_action(action)        # SEAction
engine.process_actions([a1, a2])
engine.serialize_to_string(eSerializationFormat.BINARY)          # save state
engine.serialize_from_string(state, data_request_mgr, fmt)       # load state
engine.serialize_to_file(path) / serialize_from_file(path, drm)
engine.get_initial_patient(SEPatient())
engine.pull_active_events()
engine.log_to_console(False)
```

`eModelType`: `HumanAdultWholeBody` | `HumanAdultVentilationMechanics`.

### ⚠️ `advance_time_s()` samples only ONCE, at the end

```python
def advance_time_s(self, duration_s):
    num_steps = int(total_time / self._dt_s)
    for n in range(num_steps):
        if not self._advance_time(): return False
    self._pull()          # <-- single pull AFTER the whole loop
```

`pull_data()` returns the **instantaneous state**, not a time series. Therefore:

- **Scalar charting** (Arch Decision 8, ~5 s cadence): call `advance_time_s(5.0)`
  then `pull_data()`, repeatedly. Fine.
- **Waveform capture** (Arch Decision 9, 50 Hz): `advance_time_s()` is unusable —
  it discards every intermediate timestep. Must loop `advance_time()` (one
  timestep) and `pull_data()` each iteration.

The engine timestep is read via `self._dt_s = __pulse.get_timestep("s")`
(nominally 0.02 s = 50 Hz). `PulseEngine` also tracks `_spare_time_s` so
successive `advance_time_s` calls don't drift — a per-timestep loop must not
re-implement that carelessly.

### `pull_data()` result ordering

Values come back as a flat list, ordered by the data requests as registered, with
**simulation time always first** (`"Simulation time is always the first result"`).
The adapter must build its `metric_code -> index` map accordingly: index 0 is
sim time, our requests start at index 1.

## 4. Data requests

```python
drm = SEDataRequestManager()
drm.set_data_requests([
    SEDataRequest.create_physiology_request("HeartRate", unit=FrequencyUnit.Per_min),
    SEDataRequest.create_physiology_request("MeanArterialPressure", unit=PressureUnit.mmHg),
    SEDataRequest.create_physiology_request("OxygenSaturation"),   # unitless 0..1
    SEDataRequest.create_ecg_request("Lead3ElectricPotential", unit=ElectricPotentialUnit.mV),
    SEDataRequest.create_gas_compartment_substance_request(
        "Carina", "CarbonDioxide", "PartialPressure", unit=PressureUnit.mmHg),
])
```

Factory methods seen in use: `create_physiology_request`, `create_ecg_request`,
`create_gas_compartment_substance_request`. Others exist (`CONFIRM` the full set
in `pulse/cdm/engine.py` once built) — likely `create_liquid_compartment_*`,
`create_substance_request`, `create_mechanical_ventilator_request`,
`create_action_request`.

Names confirmed present in Pulse's default request set:
`HeartRate`, `ArterialPressure`, `MeanArterialPressure`,
`SystolicArterialPressure`, `DiastolicArterialPressure`, `OxygenSaturation`,
`EndTidalCarbonDioxidePressure`, `RespirationRate`, `SkinTemperature`,
`CardiacOutput`, `BloodVolume`.

- **`OxygenSaturation` is unitless 0..1** — multiply by 100 for our `spo2` (%).
- Pulse's default uses **`SkinTemperature`**; `CoreTemperature` is the name we
  want for `temp_c` — `CONFIRM` it is a valid physiology request name.
- `CardiacOutput` is directly available in `VolumePerTimeUnit.L_Per_min`, so no
  mL/min→L/min conversion is needed (the plan's `convert=lambda x: x/1000` is
  unnecessary if we request L/min).
- `CentralVenousPressure` — `CONFIRM`.

### Pulse DOES produce a native ECG

`create_ecg_request("Lead3ElectricPotential", ...)` exists — Pulse emits ECG
Lead 3 electric potential, driven by a canned waveform, at the engine timestep
(50 Hz). It is **not** a substitute for our 250 Hz synthesized Lead II, but it is
a useful cross-check and calibration reference for the synthesizer. Update the
Phase 2 wording: Pulse lacks *monitor-rate, multi-lead* ECG morphology, not ECG
entirely.

## 5. Patient actions (`pulse.cdm.patient_actions`)

Relevant classes confirmed present:

| Need | Pulse class | Notes |
|---|---|---|
| IV bolus | `SESubstanceBolus` | ⚠️ see dose warning below |
| IV infusion | `SESubstanceInfusion` | rate + concentration |
| Compound infusion (fluids) | `SESubstanceCompoundInfusion` | bag volume + rate |
| Intubate / extubate | `SEIntubation` | `eIntubationType` |
| Hemorrhage | `SEHemorrhage` | severity **or** flow rate, + compartment |
| Vent settings | `SEMechanicalVentilation`, `SERespiratoryMechanicsConfiguration` | `CONFIRM` which fits |
| Supplemental O2 | `SESupplementalOxygen` | `eDevice` |
| Arrhythmia | `SEArrhythmia` | `eHeartRhythm` |
| Exercise / stress | `SEExercise`, `SEAcuteStress` | |

Also available and useful for scenario events: `SEAirwayObstruction`,
`SEAsthmaAttack`, `SEBronchoconstriction`, `SETensionPneumothorax`,
`SEHemothorax`, `SEPericardialEffusion`, `SEChestCompression`,
`SEPneumoniaExacerbation`, `SEBrainInjury`, `SEDyspnea`, `SEUrinate`.

### ⚠️ `SESubstanceBolus` dose is a VOLUME, not a mass

```python
class SESubstanceBolus(SEPatientAction):
    def set_substance(self, substance: str)
    def set_admin_route(self, route: eSubstance_Administration)
    def get_dose(self)           -> SEScalarVolume          # mL !!
    def get_concentration(self)  -> SEScalarMassPerVolume   # e.g. mg/mL
    def get_admin_duration(self) -> SEScalarTime
```

A clinical order of "propofol 100 mg IV" must be expressed as a
**concentration + volume**: 10 mg/mL at 10 mL. Our `actions.py` therefore has to
carry a per-drug concentration table and compute
`volume_ml = dose_mg / concentration_mg_per_ml`. The plan's
`SubstanceBolus(substance, dose_mg)` shape does not map directly to Pulse.

`eSubstance_Administration`: `Intravenous` (default), plus oral/intramuscular/
etc. — `CONFIRM` full member list.

### `SEHemorrhage`

```python
eHemorrhage_Type: External | Internal
eHemorrhage_Compartment: Aorta, Brain, Muscle, LargeIntestine, LeftArm,
    LeftKidney, LeftLeg, Liver, RightArm, RightKidney, RightLeg, Skin,
    SmallIntestine, Splanchnic, Spleen, VenaCava
```
Set `severity` (0..1) **or** `flow_rate`. Compartments are anatomic — there is no
`"Abdomen"`. For an intra-abdominal trauma bleed use `Liver`, `Spleen`,
`Splanchnic`, or `VenaCava`.

### `SEIntubation`

```python
eIntubationType: Off | Esophageal | LeftMainstem | RightMainstem |
                 Tracheal | Oropharyngeal | Nasopharyngeal
```
Intubate → `Tracheal`; extubate → `Off`. `Esophageal` / mainstem values are a
free source of realistic airway-misadventure scenarios.

## 6. Open questions to resolve at build time

1. **No anesthesia machine.** `equipment_actions.py` defines only
   `SEEquipmentAction`; Pulse 4.x appears to have dropped the anesthesia machine
   that older versions had. **Inhaled volatile agents (sevoflurane MAC) may have
   no delivery path.** This directly threatens Phase 3 `MAINTENANCE`. Fallbacks:
   drive maintenance with TIVA (propofol/remifentanil infusions, which Pulse
   supports) instead of a volatile, or model the volatile via the SciPy
   surrogate. **Decide before Phase 3.**
2. **Substance coverage** (Risk 2): confirm which of Propofol, Fentanyl,
   Rocuronium, Sugammadex, Succinylcholine, Norepinephrine, Phenylephrine,
   Sevoflurane ship as Pulse substances (`data/` is generated from `Data.xlsx`,
   so this can only be enumerated after a build).
3. **`SERespiratoryMechanicsConfiguration` vs `SEMechanicalVentilation`** — which
   one expresses PEEP / tidal volume / rate / FiO2 for our `ADJUST_VENT` intent.
4. Whether a **pre-stabilized patient state file** ships with the build (would
   make `initialize()` fast — see Task 1.1 Step 4).
5. `PulseEnginePool` — may offer in-process multi-patient parallelism, which
   would change the Phase 5 "one OS process per patient" decision.
