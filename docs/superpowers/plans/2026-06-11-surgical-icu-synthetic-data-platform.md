# Surgical-ICU Synthetic Data Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute phases **in order** — each phase depends on interfaces defined in the previous one. Do **not** skip the TDD steps (write failing test → see it fail → implement → see it pass → commit).

**Goal:** Turn the existing ICU PDES scaffold into a system that generates large, validated, multi-modal synthetic datasets (scalar vitals + high-frequency waveforms + clinical narratives + medication/event logs) for surgical-ICU and OR patients, suitable for anesthesiology safety research.

**Architecture:** The hand-rolled SciPy PK/PD code and parametric disease injector are **demoted from "physiology truth" to fast surrogates**. The [Pulse Physiology Engine](https://pulse.kitware.com) (Kitware, Apache-2.0, C++ with Python bindings) becomes the single source of physiological truth: one Pulse engine instance per simulated patient, advanced in lockstep with the PDES barrier loop. Pulse emits scalar vitals **and** continuous waveforms at its native 50 Hz timestep (airway pressure, capnography, flow/volume); seeded morphology synthesizers driven by Pulse's beat-level outputs generate the monitor-rate channels (ECG 250 Hz, arterial pressure and pleth 125 Hz) that Pulse cannot natively produce at clinical sample rates. A scenario state machine adds OR phases (preop → induction → intubation → maintenance → emergence → extubation → PACU/ICU) as a *policy* over the same engine loop. LLM clinical agents (OpenAI-compatible endpoint — local vLLM by default, cloud **only with Zero Data Retention enabled**, see Architecture Decision 10; temperature 0.3) drive care decisions and generate narrative notes. A cohort sampler + batch runner + multi-format exporter (Parquet / WFDB / FHIR) produce datasets at scale. Two real datasets calibrate the system: **VitalDB** (intraoperative high-resolution vitals + waveforms + infusion rates) for physiologic/waveform fidelity, and **MIMIC-IV** (ICU EHR + MIMIC-IV-Note + MIMIC-IV waveforms) for clinical-course and narrative calibration.

**Tech Stack:** Python 3.11+, Pulse Physiology Engine (Python bindings), TimescaleDB + pgvector + Apache AGE (PostgreSQL 16), gRPC/protobuf, asyncpg, NumPy/SciPy, vLLM (OpenAI-compatible server) + an open-weights instruct model, PyArrow/Parquet, `wfdb` (WFDB I/O), `vitaldb` (VitalDB API), `sentence-transformers` (384-dim embeddings for agent memory), `pytest`/`pytest-asyncio`, Docker Compose.

---

## How This Plan Is Organized

The work is split into **six phases**, each independently testable and each producing working software:

| Phase | Layer | Outcome |
|---|---|---|
| **Phase 0** | Prereqs | Repo is git-tracked, Pulse builds and runs from Python in the container, datasets are accessible |
| **Phase 1** | **L1** | Pulse drives scalar vitals into TimescaleDB every barrier; rollback reverts physiology |
| **Phase 2** | **L2** | Continuous waveforms (ART/ECG/PLETH/CO2/AWP/flow/vol) generated, stored, and exportable |
| **Phase 3** | **L4** | OR/anesthesia phase state machine (induction, intubation, NMB, emergence) on the same engine |
| **Phase 4** | **L3** | vLLM clinical agents make decisions + write narrative notes; pgvector agent memory |
| **Phase 5** | **L5** | Cohort sampler + parallel batch runner + Parquet/WFDB/FHIR export |
| **Cross-cutting** | Calibration | VitalDB + MIMIC-IV calibration harness, run as a gate at the end of Phases 1, 2, and 4 |

> **Ordering rationale:** L4 (OR phases) comes **before** L3 (LLM agents) because agents need a well-defined phase/action vocabulary to act against, and the phase state machine can be driven by deterministic scripted "agents" for testing before the LLM exists. Calibration is woven in rather than left to the end so fidelity problems surface early.

> **Numbering caution:** the "L*n*" labels come from the v4 spec's layer numbering and intentionally do **not** match phase numbers (Phase 3 builds L4; Phase 4 builds L3). In commit messages, gate names, and discussion, always reference the **phase** number; treat L-labels as spec cross-references only.

**Conventions used throughout:**
- All new engine code lives under `src/engine/`. Run tests with `PYTHONPATH=src python -m pytest`.
- `metric_code` strings extend the existing set in `src/engine/pdes/snapshot.py::_metric_unit`.
- Pulse Python API symbol names below follow the Pulse 4.x `PyPulse`/`pulse.cdm` conventions. **Pulse IS published on PyPI as `pulse-physiology-engine`** — Phase 0 Task 0.2 tries the wheel first (this is an Apple-Silicon host, so verify a linux/arm64 wheel exists for the container) and falls back to a source build only if no usable wheel is found. Exact symbol names MUST be confirmed against the version installed in Phase 0 Task 0.2 (see that task for the verification command). Where a name is uncertain it is wrapped behind our own adapter (`src/engine/physiology/pulse_adapter.py`) so the rest of the codebase never imports Pulse directly.
- **Anchor edits to function/class names, never to line numbers** — line references drift as tasks land. Where this plan says "in `_commit_single`" etc., locate the symbol, don't count lines.

---

## Architecture Decisions (read before starting)

1. **One Pulse engine instance per patient.** Pulse is single-patient and stateful. A simulation of patient `P` owns exactly one `PulseAdapter`. Cohort scale (Phase 5) is achieved by running many independent processes, not many patients in one engine.

2. **Pulse is advanced by the engine, never by agents.** Agents submit `AgentIntent`s (give drug, change vent, intubate). The engine translates approved intents into Pulse *actions*, advances Pulse to the next barrier, then samples Pulse outputs into TimescaleDB. This preserves the existing "engine-driven physiology, not LLM-driven vitals" principle from `v4_final_evaluation.md`.

3. **The existing SciPy `PKEngine` is kept as a surrogate**, used (a) by the `ConstraintValidator` for fast dose-safety math and (b) as a fallback when Pulse lacks a substance. It is no longer written to `vital_signs` as truth.

4. **Waveforms are stored as compressed chunks, not one row per sample.** A 125 Hz arterial line over a 6-hour case is 2.7M samples/channel — row-per-sample is untenable. We store fixed-duration windows (default 10 s) as `REAL[]` arrays in a TimescaleDB hypertable with native compression.

5. **Determinism:** every simulation takes a `seed`. Pulse, the ECG synthesizer, the cohort sampler, and (where supported) the LLM sampler are all seeded so runs are reproducible.

6. **The `internal/` and `cmd/` Go directories are abandoned scaffolding** (the proto has a `go_package` option from an earlier design). Do not build on them. Leave them; optionally delete in Phase 0 Task 0.1 if the user approves.

7. **One orchestration loop.** `PDES_Engine` is the ONLY loop that advances physiology, validates intents, commits, and persists. The Phase 3 `ScenarioStateMachine` is a *policy object* plugged into the engine (per-barrier cadence, phase-entry actions, scheduled event injection) — it must NOT own its own advance/sample/persist cycle. This keeps validation, rollback, and the `agent_actions`/`mar` audit trail uniform across OR and ICU, and means Phase 4 LLM agents simply register on the same engine with no mode switch.

8. **Charting cadence is decoupled from deliberation cadence.** Barriers control when agents think (15 s–15 min depending on phase); scalar vitals are sampled from Pulse every `charting_interval_s` (default 5 s) *during* each barrier advance and batch-inserted. One data point per barrier is useless for anesthesia safety research — VitalDB, the calibration reference, records at 1–2 s.

9. **Waveform fidelity strategy.** Pulse advances at a fixed 0.02 s timestep, so 50 Hz is the highest rate it can natively emit. Gas/ventilator channels (`CO2`/`AWP`/`FLOW`/`VOL`) are sampled at 50 Hz directly from Pulse. Monitor-rate channels (`ART`/`PLETH` at 125 Hz, `ECG_II` at 250 Hz) are **synthesized** by seeded, stateful morphology generators driven by Pulse's beat-to-beat parameters (HR, SBP/DBP/MAP, SpO2), calibrated against VitalDB in the Phase 2 gate. Never upsample a 50 Hz Pulse trace and label it 125 Hz.

10. **LLM data policy (PhysioNet DUA).** The default endpoint is local vLLM. Sending MIMIC-derived text (notes, prompts containing note excerpts, calibration samples) to a cloud LLM is permitted **if and only if the provider endpoint has Zero Data Retention (ZDR) enabled**; anything else violates the MIMIC DUA. Enforced in code: `LLMClient` refuses non-local `base_url`s unless `LLM_ZDR_CONFIRMED=1` is set (Task 4.1), and that flag must only be set on accounts with ZDR actually enabled.

---

## Target File Structure

```
src/engine/
├── physiology/                      # NEW — Pulse integration (the L1/L2 core)
│   ├── __init__.py
│   ├── pulse_adapter.py             # Wraps the Pulse Python engine; the ONLY file importing pulse
│   ├── data_requests.py             # Declares which Pulse outputs we sample + their metric_codes
│   ├── actions.py                   # AgentIntent/phase -> Pulse action translation
│   ├── waveforms.py                 # Waveform sampling + windowing (Phase 2)
│   ├── ecg_synth.py                 # Synthetic ECG morphology from Pulse HR (Phase 2)
│   └── pressure_synth.py            # Synthetic ART/PLETH morphology from Pulse beat params (Phase 2)
├── scenario/                        # NEW — OR/anesthesia phase machine (Phase 3 / L4)
│   ├── __init__.py
│   ├── phases.py                    # Phase enum + transition rules + per-phase Pulse config
│   ├── machine.py                   # ScenarioStateMachine driving barrier cadence + actions
│   └── library.py                   # Built-in scenarios (elective lap chole, trauma laparotomy, ...)
├── agents/                          # NEW — LLM clinical agents (Phase 4 / L3)
│   ├── __init__.py
│   ├── client.py                    # vLLM OpenAI-compatible client (temp 0.3, JSON mode)
│   ├── personas.py                  # Role prompts: Nurse, RT, Resident, Attending, Anesthesiologist, CRNA
│   ├── deliberator.py               # snapshot(+context) -> AgentIntent ; implements the Callable the PDES engine expects
│   ├── memory.py                    # pgvector-backed episodic memory
│   └── notes.py                     # Narrative note generation (calibrated to MIMIC-IV-Note)
├── cohort/                          # NEW — scale-out (Phase 5 / L5)
│   ├── __init__.py
│   ├── sampler.py                   # Patient/procedure/event-schedule sampler (seeded)
│   ├── runner.py                    # Parallel batch orchestration (one process per patient)
│   └── export.py                    # Parquet / WFDB / FHIR writers
├── calibration/                     # NEW — cross-cutting fidelity gates
│   ├── __init__.py
│   ├── vitaldb_ref.py               # Load + summarize VitalDB reference cohorts
│   ├── mimic_ref.py                 # Load + summarize MIMIC-IV (EHR, notes, waveforms)
│   └── compare.py                   # Distribution/spectral comparison + pass/fail report
├── pdes/engine.py                   # MODIFY — call physiology on commit + advance + rollback revert
├── pkpd/models.py                   # KEEP — surrogate only
├── disease/injector.py             # KEEP — surrogate; superseded by Pulse pathophysiology where available
├── validators/constraint.py        # MODIFY — add OR-phase drugs (NMB, volatiles), keep dose math
├── store/postgres.py               # MODIFY — add waveform chunk, MAR, note + pgvector helpers
└── store/age_graph.py              # NEW — Cypher write/read layer for Apache AGE (Task 1.7)

db/init/
├── 04_waveform_schema.sql          # NEW — waveform_chunks hypertable + compression (Phase 2)
├── 05_agent_memory_schema.sql      # NEW — pgvector episodic memory table (Phase 4)
├── 06_cohort_schema.sql            # NEW — simulation_runs registry + patient dim (Phase 5)
└── 07_clinical_notes_schema.sql    # NEW — clinical_notes table (Phase 4)

proto/simulation.proto              # MODIFY — add waveform, phase, and note messages (Phases 2-4)

tests/
├── test_pulse_adapter.py           # Phase 1
├── test_physiology_loop.py         # Phase 1
├── test_waveforms.py               # Phase 2
├── test_ecg_synth.py               # Phase 2
├── test_scenario_machine.py        # Phase 3
├── test_agents.py                  # Phase 4
├── test_cohort.py                  # Phase 5
└── test_calibration.py             # cross-cutting

scripts/
├── build_pulse.sh                  # Phase 0 — source-build fallback if no PyPI wheel works
├── run_single_sim.py               # Phase 1 — CLI to run one patient end-to-end
├── bench_pulse.py                  # Phase 1 — sim-speed + save_state cost benchmark (Task 1.8)
└── run_cohort.py                   # Phase 5 — CLI to launch a cohort
```

---

# Phase 0 — Prerequisites

**Goal:** Git-track the repo, get Pulse building and callable from Python inside the dev container, and confirm both calibration datasets are reachable. Nothing in later phases can be tested without this.

### Task 0.1: Initialize git and commit the current scaffold

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
*.egg-info/
build/
dist/
# Pulse build artifacts
third_party/pulse/builds/
*.so
# Large data — never commit datasets
data/vitaldb/
data/mimic/
data/exports/
# Secrets
.env
```

- [ ] **Step 2: Initialize the repo and make the baseline commit**

Run:
```bash
cd /Volumes/Jeremy/Users/jeremy/Development/patient_simulator
git init
git add -A
git commit -m "chore: baseline ICU PDES scaffold under version control"
```
Expected: a commit succeeds containing the existing `src/`, `db/`, `proto/`, `tests/`, etc.

- [ ] **Step 3: (Optional, ask user first) remove abandoned Go scaffolding**

Only if the user confirms `internal/` and `cmd/` are dead. If yes:
```bash
git rm -r internal cmd && git commit -m "chore: remove abandoned Go engine scaffolding"
```

### Task 0.2: Install Pulse with Python bindings in the container

Kitware publishes Pulse Python wheels on PyPI as **`pulse-physiology-engine`** — try that FIRST; it reduces this task to a pip install. Compile from source (gitlab.kitware.com/physiology/engine) only if no wheel works for the container platform. Note: this is an Apple-Silicon host, so the container needs a **linux/arm64** wheel; if only amd64 wheels exist, weigh Rosetta emulation (`platform: linux/amd64`, slower for a CPU-bound physiology engine) against the source build. Either way the install is isolated in an `engine` Docker image so contributors never build anything on the host.

**Files:**
- Create: `scripts/build_pulse.sh` (fallback path only)
- Create: `engine.Dockerfile`
- Modify: `docker-compose.yml` (add an `engine` service)
- Modify: `requirements.txt` (wheel path only)

- [ ] **Step 1: Try the PyPI wheel first**

Run:
```bash
docker run --rm python:3.11-bookworm bash -c \
  "pip install --no-cache-dir pulse-physiology-engine && python -c 'import pulse; print(\"pulse import OK\")'"
```
Expected: `pulse import OK`. If it prints OK: add `pulse-physiology-engine` (pinned to the resolved version) to `requirements.txt`, **skip Step 2 entirely**, use the simple pip-based `engine.Dockerfile` in Step 3, and record in `data/README.md` that the wheel path was taken. If pip finds no compatible linux/arm64 wheel, retry with `docker run --platform linux/amd64 ...`; only if neither works, fall back to Step 2.

- [ ] **Step 2 (fallback — ONLY if Step 1 found no usable wheel): Write `scripts/build_pulse.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
# Builds the Pulse Physiology Engine with Python bindings.
# Pin to a known-good tag; bump deliberately, never float on main.
PULSE_TAG="${PULSE_TAG:-4.4.0}"
PULSE_SRC="/opt/pulse/engine"
PULSE_BUILD="/opt/pulse/build"

git clone --depth 1 --branch "v${PULSE_TAG}" \
    https://gitlab.kitware.com/physiology/engine.git "${PULSE_SRC}" || \
  (cd "${PULSE_SRC}" && git fetch --depth 1 origin "v${PULSE_TAG}" && git checkout "v${PULSE_TAG}")

cmake -S "${PULSE_SRC}/src" -B "${PULSE_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPulse_BUILD_PYTHON=ON \
    -DPulse_PYTHON_BINDINGS=ON
cmake --build "${PULSE_BUILD}" --config Release -j "$(nproc)"

# Expose the built python package + data files on PYTHONPATH
# (python:3.11-bookworm keeps site-packages under /usr/local/lib, not /usr/lib)
echo "${PULSE_BUILD}/install/python" > /usr/local/lib/python3.11/site-packages/pulse.pth
```

> NOTE FOR IMPLEMENTER: Pulse's exact CMake flag names and the install layout vary by version. Consult the ReadMe of the pinned tag (`https://gitlab.kitware.com/physiology/engine/-/blob/v${PULSE_TAG}/ReadMe.md`). The goal is: after this script, `python -c "import pulse"` works and Pulse's `bin`/data directory is locatable. Record the working flags in this file.

- [ ] **Step 3: Write `engine.Dockerfile`**

Wheel path (preferred — `pulse-physiology-engine` is in `requirements.txt`):
```dockerfile
FROM python:3.11-bookworm
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt -e ".[dev]"
COPY . .
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "engine.main"]
```

Source-build fallback (only if Step 2 was needed):
```dockerfile
FROM python:3.11-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY scripts/build_pulse.sh /tmp/build_pulse.sh
RUN bash /tmp/build_pulse.sh
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt -e ".[dev]"
COPY . .
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "engine.main"]
```

- [ ] **Step 4: Add the `engine` service to `docker-compose.yml`**

Add under `services:` (alongside the existing `db`):
```yaml
  engine:
    build:
      context: .
      dockerfile: engine.Dockerfile
    depends_on:
      - db
    environment:
      PGHOST: db
      PGUSER: psim
      PGPASSWORD: psim_dev
      PGDATABASE: patient_simulator
    volumes:
      - ./src:/app/src
      - ./data:/app/data
    # platform: linux/amd64   # uncomment ONLY if forced onto the amd64 wheel via emulation (Step 1)
```

- [ ] **Step 5: Build the image and verify Pulse imports**

Run:
```bash
docker compose build engine
docker compose run --rm engine python -c "import pulse; print('pulse import OK')"
```
Expected: prints `pulse import OK` with no traceback. If it fails, fix the install path (wheel pin or `build_pulse.sh` flags) before proceeding — **do not** continue to Phase 1 until this passes.

- [ ] **Step 6: Capture the real Pulse API surface for the adapter**

Run and save the output into a scratch note (this is the authoritative reference for Task 1.1):
```bash
docker compose run --rm engine python - <<'PY'
import pulse, inspect
import pulse.cdm.engine as eng  # adjust module path per version
print([n for n in dir(pulse) if not n.startswith('_')])
PY
```
Expected: a list of public symbols (engine factory, patient initialization, action classes, data-request classes). Use these exact names in Task 1.1.

- [ ] **Step 7: Commit**

```bash
git add engine.Dockerfile docker-compose.yml requirements.txt   # + scripts/build_pulse.sh if Step 2 was needed
git commit -m "build: install Pulse Physiology Engine with Python bindings in engine image"
```

### Task 0.3: Confirm dataset access (VitalDB + MIMIC-IV)

**Files:**
- Create: `requirements.txt` additions (append, do not rewrite)
- Create: `data/README.md`

- [ ] **Step 1: Add calibration/export dependencies to `requirements.txt`**

Append:
```
vitaldb>=1.4.0
wfdb>=4.1.0
pyarrow>=15.0.0
pandas>=2.2.0
openai>=1.30.0                 # vLLM OpenAI-compatible client (Phase 4)
sentence-transformers>=2.7.0   # 384-dim agent-memory embeddings (Phase 4, matches VECTOR(384))
```

- [ ] **Step 2: Write `data/README.md` documenting expected layout**

```markdown
# Calibration & export data (gitignored)

data/
├── vitaldb/      # VitalDB cached cases (downloaded via the `vitaldb` package or local copy)
├── mimic/        # MIMIC-IV: hosp/, icu/, note/, and waveform (wfdb) subsets (credentialed)
└── exports/      # Generated synthetic datasets (Parquet/WFDB/FHIR)

VitalDB: the user has the full VitalDB dataset downloaded locally. Point
VITALDB_LOCAL_DIR at it, or let the `vitaldb` package stream from the API.

MIMIC-IV: requires PhysioNet credentialing. Place the extracted CSV.gz tables
under data/mimic/{hosp,icu}, notes under data/mimic/note, and the matched
waveform records under data/mimic/wdb.
```

- [ ] **Step 3: Verify VitalDB reachability**

Run:
```bash
docker compose run --rm engine python - <<'PY'
import vitaldb
caseids = vitaldb.find_cases(['ART'])[:3]   # cases with an arterial waveform track
print("VitalDB OK, sample case ids:", caseids)
PY
```
Expected: prints a short list of integer case ids. If offline, set `VITALDB_LOCAL_DIR` and load from disk instead.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt data/README.md
git commit -m "chore: add calibration/export deps and document dataset layout"
```

---

# Phase 1 — L1: Pulse-Driven Scalar Vitals

**Goal:** Replace stub physiology with Pulse. Each PDES barrier advances a real Pulse engine, applies approved drug/vent intents as Pulse actions, and writes engine-sourced scalar vitals to `vital_signs`. Rollback reverts physiology to the checkpoint.

**Key interfaces created in this phase (used by all later phases):**

```python
# src/engine/physiology/pulse_adapter.py
class PulseAdapter:
    def __init__(self, seed: int, patient_cfg: "PatientConfig") -> None: ...
    def initialize(self) -> None: ...                      # stabilize to a starting state
    def advance(self, seconds: float) -> None: ...          # run physiology forward
    def apply_action(self, action: "PulseAction") -> None: ...
    def sample_scalars(self) -> dict[str, float]: ...       # {metric_code: value}
    def save_state(self) -> bytes: ...                      # for rollback
    def load_state(self, blob: bytes) -> None: ...
    def close(self) -> None: ...
```

### Task 1.1: `PatientConfig` + `PulseAdapter` wrapping the Pulse engine

**Files:**
- Create: `src/engine/physiology/__init__.py` (empty)
- Create: `src/engine/physiology/pulse_adapter.py`
- Test: `tests/test_pulse_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pulse_adapter.py
import pytest
from engine.physiology.pulse_adapter import PulseAdapter, PatientConfig

@pytest.fixture
def adapter():
    cfg = PatientConfig(sex="MALE", age_yr=65, height_cm=175, weight_kg=80)
    a = PulseAdapter(seed=42, patient_cfg=cfg)
    a.initialize()
    yield a
    a.close()

def test_initial_vitals_are_physiologic(adapter):
    v = adapter.sample_scalars()
    assert 50 <= v["hr"] <= 100, f"resting HR out of range: {v['hr']}"
    assert 60 <= v["map"] <= 110, f"resting MAP out of range: {v['map']}"
    assert 90 <= v["spo2"] <= 100, f"SpO2 out of range: {v['spo2']}"
    assert 8 <= v["rr"] <= 24, f"RR out of range: {v['rr']}"

def test_advance_progresses_time(adapter):
    before = adapter.sample_scalars()
    adapter.advance(30.0)
    after = adapter.sample_scalars()
    assert before.keys() == after.keys()
    # vitals should stay physiologic after 30s of quiet advancement
    assert 50 <= after["hr"] <= 100
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_pulse_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: engine.physiology.pulse_adapter`.

- [ ] **Step 3: Implement `pulse_adapter.py`**

> Use the exact Pulse symbol names captured in Phase 0 Task 0.2 Step 5. The structure below is the contract; fill the `# PULSE:` lines with the real API of the pinned version. All Pulse imports stay inside this file.

```python
"""Adapter around the Pulse Physiology Engine.

This is the ONLY module permitted to import `pulse`. Everything else in the
codebase depends on PulseAdapter's stable interface so that a Pulse version
bump (or a swap to a different physiology engine) is contained here.
"""
from __future__ import annotations

from dataclasses import dataclass

# PULSE: import the engine factory, patient/init classes, and data-request enums.
import pulse  # noqa: F401  (concrete imports filled per pinned version)


@dataclass
class PatientConfig:
    sex: str            # "MALE" | "FEMALE"
    age_yr: int
    height_cm: float
    weight_kg: float
    hr_baseline: float | None = None     # optional overrides for calibration
    map_baseline: float | None = None


# metric_code -> Pulse data request descriptor. Filled in data_requests.py (Task 1.2),
# imported here to keep sampling declarative.
from engine.physiology.data_requests import SCALAR_REQUESTS


class PulseAdapter:
    def __init__(self, seed: int, patient_cfg: PatientConfig) -> None:
        self.seed = seed
        self.cfg = patient_cfg
        self._engine = None      # PULSE: created in initialize()

    def initialize(self) -> None:
        # PULSE: create engine, build a SEPatient from self.cfg, set data requests
        # from SCALAR_REQUESTS, and run to a stable resting state (or load a
        # pre-stabilized patient state file for speed).
        raise NotImplementedError

    def advance(self, seconds: float) -> None:
        if seconds <= 0:
            return
        # PULSE: engine.advance_time_s(seconds)
        raise NotImplementedError

    def apply_action(self, action) -> None:  # action: PulseAction (Task 1.4)
        # PULSE: engine.process_action(action.to_pulse())
        raise NotImplementedError

    def sample_scalars(self) -> dict[str, float]:
        # PULSE: read the current values for each request in SCALAR_REQUESTS,
        # converting units to our canonical metric units.
        raise NotImplementedError

    def save_state(self) -> bytes:
        # PULSE: serialize engine state (Pulse supports state save/load to file/string)
        raise NotImplementedError

    def load_state(self, blob: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self._engine = None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_pulse_adapter.py -v`
Expected: PASS. If `initialize` is slow (>5s), in Step 3 load a pre-stabilized Pulse state file for the standard patient instead of stabilizing live.

- [ ] **Step 5: Commit**

```bash
git add src/engine/physiology/__init__.py src/engine/physiology/pulse_adapter.py tests/test_pulse_adapter.py
git commit -m "feat(physiology): PulseAdapter wrapping the Pulse engine with scalar sampling"
```

### Task 1.2: Declarative scalar data requests

**Files:**
- Create: `src/engine/physiology/data_requests.py`
- Test: `tests/test_pulse_adapter.py` (add a case)

- [ ] **Step 1: Write the failing test (append to `tests/test_pulse_adapter.py`)**

```python
def test_scalar_requests_cover_core_metrics():
    from engine.physiology.data_requests import SCALAR_REQUESTS
    codes = {r.metric_code for r in SCALAR_REQUESTS}
    assert {"hr", "map", "rr", "spo2", "temp_c", "co", "cvp", "etco2"} <= codes
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_pulse_adapter.py::test_scalar_requests_cover_core_metrics -v`
Expected: FAIL — module/attribute missing.

- [ ] **Step 3: Implement `data_requests.py`**

```python
"""Declarative mapping from Pulse outputs to our canonical metric_codes + units.

Each ScalarRequest records: our metric_code, the Pulse physiology data name,
the Pulse unit string, and a converter to our canonical unit (see
engine/pdes/snapshot.py::_metric_unit for the canonical units).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ScalarRequest:
    metric_code: str
    pulse_name: str          # PULSE: physiology data name, e.g. "HeartRate"
    pulse_unit: str          # PULSE: e.g. "1/min", "mmHg", "mL/min", "%"
    convert: Callable[[float], float] = lambda x: x


SCALAR_REQUESTS: list[ScalarRequest] = [
    ScalarRequest("hr",     "HeartRate",                "1/min"),
    ScalarRequest("map",    "MeanArterialPressure",     "mmHg"),
    ScalarRequest("sbp",    "SystolicArterialPressure", "mmHg"),
    ScalarRequest("dbp",    "DiastolicArterialPressure","mmHg"),
    ScalarRequest("rr",     "RespirationRate",          "1/min"),
    ScalarRequest("spo2",   "OxygenSaturation",         "unitless", convert=lambda x: x * 100.0),
    ScalarRequest("etco2",  "EndTidalCarbonDioxidePressure", "mmHg"),
    ScalarRequest("temp_c", "CoreTemperature",          "degC"),
    ScalarRequest("co",     "CardiacOutput",            "mL/min",  convert=lambda x: x / 1000.0),  # -> L/min
    ScalarRequest("cvp",    "CentralVenousPressure",    "mmHg"),
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_pulse_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Extend `_metric_unit` in `src/engine/pdes/snapshot.py`**

Add the new codes to the `units` dict in `_metric_unit`:
```python
    units = {
        "hr": "bpm", "map": "mmHg", "sbp": "mmHg", "dbp": "mmHg",
        "rr": "bpm", "spo2": "%", "temp_c": "°C", "cvp": "mmHg",
        "co": "L/min", "etco2": "mmHg",
        "levo_rate": "mcg/min", "prop_rate": "mg/hr", "fent_rate": "mcg/hr",
    }
```

- [ ] **Step 6: Commit**

```bash
git add src/engine/physiology/data_requests.py src/engine/pdes/snapshot.py tests/test_pulse_adapter.py
git commit -m "feat(physiology): declarative Pulse scalar requests + canonical units"
```

### Task 1.3: Persist sampled scalars to TimescaleDB

**Files:**
- Modify: `src/engine/store/postgres.py` (add `insert_vitals` helper)
- Test: `tests/test_physiology_loop.py`

- [ ] **Step 1: Write the failing test (uses a transaction-rolled-back DB fixture)**

```python
# tests/test_physiology_loop.py
import os
from datetime import datetime, timezone
import pytest
from engine.store.postgres import PostgresStore

pytestmark = pytest.mark.skipif(
    os.getenv("PGHOST") is None, reason="requires Postgres (run via docker compose)"
)

@pytest.fixture
async def store():
    s = PostgresStore()
    await s.connect()
    yield s
    await s.execute("DELETE FROM vital_signs WHERE patient_id = 9999")
    await s.close()

async def test_insert_and_read_back_vitals(store):
    t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    await store.insert_vitals(9999, t, {"hr": 72.0, "map": 85.0}, source="engine")
    rows = await store.fetch(
        "SELECT metric_code, value FROM vital_signs WHERE patient_id=9999 ORDER BY metric_code"
    )
    got = {r["metric_code"]: r["value"] for r in rows}
    assert got == {"hr": 72.0, "map": 85.0}
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose up -d db && docker compose run --rm engine python -m pytest tests/test_physiology_loop.py -v`
Expected: FAIL — `PostgresStore` has no `insert_vitals`.

- [ ] **Step 3: Add `insert_vitals` + `insert_vitals_batch` to `PostgresStore`**

```python
    async def insert_vitals(
        self, patient_id: int, time, vitals: dict[str, float], source: str = "engine"
    ) -> None:
        await self.insert_vitals_batch(patient_id, [(time, vitals)], source=source)

    async def insert_vitals_batch(
        self, patient_id: int, samples: list, source: str = "engine"
    ) -> None:
        """samples: [(time, {metric_code: value}), ...] — one executemany round-trip.

        The batched form exists because charting runs at ~5 s cadence (Arch
        Decision 8): a 15-min barrier yields 180 sample dicts, and per-sample
        round-trips would dominate barrier wall time.
        """
        records = [
            (t, patient_id, code, float(val), source)
            for t, vitals in samples for code, val in vitals.items()
        ]
        if not records:
            return
        if not self._pool:
            raise RuntimeError("PostgresStore not connected")
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO vital_signs (time, patient_id, metric_code, value, source)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (time, patient_id, metric_code, source) DO UPDATE SET value = EXCLUDED.value
                """,
                records,
            )
```
(The `ON CONFLICT` target matches the existing `UNIQUE (time, patient_id, metric_code, source)` constraint in `db/init/02_timescale_schema.sql` — verified.)

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_physiology_loop.py::test_insert_and_read_back_vitals -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine/store/postgres.py tests/test_physiology_loop.py
git commit -m "feat(store): batched vital_signs upsert helper"
```

### Task 1.4: Intent → Pulse action translation

**Files:**
- Create: `src/engine/physiology/actions.py`
- Test: `tests/test_physiology_loop.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_administer_med_maps_to_substance_bolus():
    from engine.physiology.actions import intent_to_pulse_action, SubstanceBolus
    action = intent_to_pulse_action(
        action_type="ADMINISTER_MED",
        payload={"drug_name": "Propofol", "dose": 100.0, "unit": "mg", "route": "IV_bolus"},
    )
    assert isinstance(action, SubstanceBolus)
    assert action.substance == "Propofol"
    assert action.dose_mg == 100.0

def test_unknown_action_returns_none():
    from engine.physiology.actions import intent_to_pulse_action
    assert intent_to_pulse_action("CALL_ATTENDING", {}) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_physiology_loop.py -k action -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `actions.py`**

```python
"""Translate approved AgentIntents (and scenario phase steps) into Pulse actions.

Our intent vocabulary -> Pulse action classes. Each PulseAction dataclass has a
`to_pulse()` method (filled per Pulse version) that PulseAdapter.apply_action
hands to the engine. Drug names must match Pulse substance names; the
SUBSTANCE_MAP handles aliasing (e.g. 'Levophed' -> 'Norepinephrine').
"""
from __future__ import annotations

from dataclasses import dataclass

SUBSTANCE_MAP = {
    "Levophed": "Norepinephrine",
    "Norepi": "Norepinephrine",
}


def _canon(name: str) -> str:
    return SUBSTANCE_MAP.get(name, name)


class PulseAction:
    def to_pulse(self):  # PULSE: return the concrete pulse action object
        raise NotImplementedError


@dataclass
class SubstanceBolus(PulseAction):
    substance: str
    dose_mg: float
    def to_pulse(self):  # PULSE: SESubstanceBolus(IV route, mass dose)
        raise NotImplementedError


@dataclass
class SubstanceInfusion(PulseAction):
    substance: str
    rate_mg_per_min: float
    concentration_mg_per_ml: float = 10.0
    def to_pulse(self):  # PULSE: SESubstanceInfusion(rate, concentration)
        raise NotImplementedError


@dataclass
class VentilatorSettings(PulseAction):
    peep_cmh2o: float
    tidal_volume_ml: float
    rate_per_min: float
    fio2: float = 0.5
    def to_pulse(self):  # PULSE: SEMechanicalVentilatorConfiguration
        raise NotImplementedError


def intent_to_pulse_action(action_type: str, payload: dict) -> PulseAction | None:
    if action_type == "ADMINISTER_MED":
        sub = _canon(payload.get("drug_name", payload.get("drug", "")))
        route = payload.get("route", "")
        if route.endswith("bolus"):
            return SubstanceBolus(substance=sub, dose_mg=float(payload.get("dose", 0.0)))
        return SubstanceInfusion(
            substance=sub, rate_mg_per_min=float(payload.get("rate", payload.get("rate_mg_per_min", 0.0)))
        )
    if action_type == "ADJUST_VENT":
        return VentilatorSettings(
            peep_cmh2o=float(payload.get("peep", 5)),
            tidal_volume_ml=float(payload.get("tidal_volume", 500)),
            rate_per_min=float(payload.get("rate", 12)),
            fio2=float(payload.get("fio2", 0.5)),
        )
    return None  # CALL_ATTENDING, CHECK_VITALS, etc. have no physiologic effect
```

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_physiology_loop.py -k action -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine/physiology/actions.py tests/test_physiology_loop.py
git commit -m "feat(physiology): AgentIntent -> Pulse action translation"
```

### Task 1.5: Wire physiology into the PDES barrier loop

This is the core L1 change: `PDES_Engine` owns a `PulseAdapter`, applies committed intents as actions, advances Pulse one barrier interval, samples scalars, and persists them. Rollback uses Pulse state save/load.

**Files:**
- Modify: `src/engine/pdes/engine.py`
- Test: `tests/test_physiology_loop.py` (append)

- [ ] **Step 1: Write the failing test (offline, store=None, with a fake adapter)**

```python
def test_engine_advances_pulse_and_samples_each_barrier():
    from datetime import datetime, timezone
    import asyncio
    from engine.pdes.engine import PDES_Engine

    class FakeAdapter:
        def __init__(self): self.t = 0.0; self.advances = 0; self.actions = []
        def initialize(self): pass
        def advance(self, s): self.t += s; self.advances += 1
        def apply_action(self, a): self.actions.append(a)
        def sample_scalars(self): return {"hr": 70 + self.advances, "map": 80.0}
        def save_state(self): return b"state"
        def load_state(self, b): pass
        def close(self): pass

    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    fake = FakeAdapter()
    captured = []
    engine = PDES_Engine(store=None, start_time=start, end_time=end, physiology=fake)
    engine.on_vitals_sampled = lambda t, v: captured.append((t, v))
    asyncio.run(engine.run())

    assert fake.t == 3600.0, "Pulse must advance exactly the simulated hour"
    assert len(captured) == 720, "1h at the default 5s charting cadence => 720 samples"
    assert captured[-1][1]["hr"] > captured[0][1]["hr"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_physiology_loop.py -k advances_pulse -v`
Expected: FAIL — `PDES_Engine.__init__` has no `physiology` parameter.

- [ ] **Step 3: Modify `PDES_Engine`**

In `src/engine/pdes/engine.py`:

(a) Extend `PDES_Engine.__init__` to accept and store the adapter, a charting interval, and a sampling hook:
```python
    def __init__(
        self,
        store: PostgresStore | None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        patient_id: int = 1,
        physiology=None,                 # PulseAdapter | None
        barrier_seconds: float = 15 * 60,
        charting_interval_s: float = 5.0,   # scalar sampling cadence WITHIN a barrier (Arch Decision 8)
    ) -> None:
        self.store = store
        self.clock = PDESClock(start_time, end_time)
        self.patient_id = patient_id
        self.physiology = physiology
        self.barrier_seconds = barrier_seconds
        self.charting_interval_s = charting_interval_s
        self.on_vitals_sampled = None    # optional Callable[[datetime, dict], None]
        self._agents = {}
        self._checkpoints = []
        self._current_version_hash = ""
        self._rollback_count = 0
        self._sequential_mode = False
        if self.physiology is not None:
            self.physiology.initialize()
```

(b) Save Pulse state in the checkpoint. Extend the `RollbackCheckpoint` dataclass:
```python
@dataclass
class RollbackCheckpoint:
    barrier_time: datetime
    version_hash: str
    committed_intents: list["IntentRecord"] = field(default_factory=list)
    physiology_state: bytes | None = None
```
and in `_process_barrier`, where the checkpoint is constructed, capture it:
```python
        checkpoint = RollbackCheckpoint(
            barrier_time=barrier.sim_time,
            version_hash=snapshot.version_hash,
            physiology_state=self.physiology.save_state() if self.physiology else None,
        )
```

(c) After `_deliberate_and_commit` returns in `_process_barrier`, advance + sample:
```python
        await self._deliberate_and_commit(snapshot, barrier, depth=0)
        await self._advance_physiology(barrier)
```
and add the method:
```python
    async def _advance_physiology(self, barrier: TimeBarrier) -> None:
        """Advance Pulse across the barrier, charting scalars every charting_interval_s.

        Barriers control deliberation cadence; charting_interval_s controls output
        resolution — VitalDB, the calibration reference, records at 1-2 s, so one
        sample per 15-min barrier would be useless for research (Arch Decision 8).
        """
        if self.physiology is None:
            return
        elapsed = 0.0
        batch: list[tuple[datetime, dict[str, float]]] = []
        while elapsed < self.barrier_seconds:
            step = min(self.charting_interval_s, self.barrier_seconds - elapsed)
            self.physiology.advance(step)
            elapsed += step
            sample_time = barrier.sim_time + timedelta(seconds=elapsed)
            batch.append((sample_time, self.physiology.sample_scalars()))
        if self.store:
            await self.store.insert_vitals_batch(self.patient_id, batch, source="engine")
        if self.on_vitals_sampled:
            for sample_time, vitals in batch:
                self.on_vitals_sampled(sample_time, vitals)
```

(d) Apply approved intents to Pulse inside `_commit_single`, immediately after the `agent_actions` insert:
```python
        if self.physiology is not None:
            from engine.physiology.actions import intent_to_pulse_action
            pulse_action = intent_to_pulse_action(intent.action_type, intent.payload)
            if pulse_action is not None:
                self.physiology.apply_action(pulse_action)
```

(e) Revert Pulse on rollback, at the end of `_rollback`:
```python
        if self.physiology is not None and checkpoint.physiology_state is not None:
            self.physiology.load_state(checkpoint.physiology_state)
```

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_physiology_loop.py -k advances_pulse -v`
Expected: PASS.

- [ ] **Step 5: Run the full existing suite to confirm no regression**

Run: `docker compose run --rm engine python -m pytest tests/ -v`
Expected: all prior tests still PASS (the new `physiology` arg defaults to `None`, preserving old behavior).

- [ ] **Step 6: Commit**

```bash
git add src/engine/pdes/engine.py tests/test_physiology_loop.py
git commit -m "feat(pdes): drive Pulse physiology from the barrier loop with rollback revert"
```

### Task 1.6: End-to-end single-sim CLI + real-Pulse smoke test

**Files:**
- Create: `scripts/run_single_sim.py`
- Test: `tests/test_physiology_loop.py` (append a slow real-Pulse test marked `slow`)

- [ ] **Step 1: Write the failing slow test**

```python
@pytest.mark.slow
def test_one_hour_sim_with_real_pulse_writes_trajectory():
    import asyncio
    from datetime import datetime, timezone
    from engine.pdes.engine import PDES_Engine
    from engine.physiology.pulse_adapter import PulseAdapter, PatientConfig

    cfg = PatientConfig(sex="MALE", age_yr=65, height_cm=175, weight_kg=80)
    adapter = PulseAdapter(seed=1, patient_cfg=cfg)
    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    captured = []
    eng = PDES_Engine(store=None, start_time=start, end_time=end, physiology=adapter)
    eng.on_vitals_sampled = lambda t, v: captured.append(v)

    def resident(snapshot):
        from engine.proto.v1 import simulation_pb2 as pb
        return pb.AgentIntent(agent_id="R1", action_type="ADMINISTER_MED",
            payload=b'{"drug_name":"Propofol","dose":100,"unit":"mg","route":"IV_bolus"}',
            based_on_hash=snapshot.version_hash)
    eng.register_agent("R1", resident)
    asyncio.run(eng.run())
    adapter.close()

    hrs = [v["hr"] for v in captured]
    assert len(hrs) >= 4
    assert all(30 < h < 200 for h in hrs), "HR trajectory must remain physiologic"
```

- [ ] **Step 2: Run to verify it fails (or errors) before implementation is complete**

Run: `docker compose run --rm engine python -m pytest tests/test_physiology_loop.py -k real_pulse -v -m slow`
Expected: FAIL until `PulseAdapter` methods (Task 1.1 Step 3) are fully filled with real Pulse calls. **This test is the acceptance gate for Phase 1.**

- [ ] **Step 3: Complete the real Pulse calls in `pulse_adapter.py`**

Fill every `# PULSE:` line using the API surface from Phase 0 Task 0.2 Step 5. Verify a propofol bolus visibly affects physiology (RR/MAP dip) in the captured trajectory.

- [ ] **Step 4: Write `scripts/run_single_sim.py`**

```python
"""Run one patient end-to-end and print the vitals trajectory (or write to DB)."""
import argparse, asyncio, json
from datetime import datetime, timedelta, timezone
from engine.pdes.engine import PDES_Engine
from engine.physiology.pulse_adapter import PulseAdapter, PatientConfig

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    cfg = PatientConfig(sex="MALE", age_yr=65, height_cm=175, weight_kg=80)
    adapter = PulseAdapter(seed=args.seed, patient_cfg=cfg)
    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    eng = PDES_Engine(store=None, start_time=start,
                      end_time=start + timedelta(hours=args.hours), physiology=adapter)
    eng.on_vitals_sampled = lambda t, v: print(t.isoformat(), json.dumps(v))
    asyncio.run(eng.run())
    adapter.close()

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the gate test and the CLI**

Run:
```bash
docker compose run --rm engine python -m pytest tests/test_physiology_loop.py -k real_pulse -m slow -v
docker compose run --rm engine python scripts/run_single_sim.py --hours 1
```
Expected: gate test PASS; CLI prints a physiologic vitals trajectory.

- [ ] **Step 6: Commit**

```bash
git add src/engine/physiology/pulse_adapter.py scripts/run_single_sim.py tests/test_physiology_loop.py
git commit -m "feat(physiology): complete real Pulse integration + single-sim CLI (L1 done)"
```

### Task 1.7: MAR + Apache AGE graph writes on commit

The schema already has a `mar` table (`db/init/02_timescale_schema.sql`) and an AGE graph (`db/init/03_age_graph_schema.sql`) — nothing writes to either. Per `v4_final_evaluation.md` Risk 3, the engine writes numerical state (Timescale) and clinical relationships (AGE) **synchronously in one PostgreSQL transaction**. This task moves the commit-path writes into a single store helper so `agent_actions` + `mar` + graph edge land atomically.

**Files:**
- Create: `src/engine/store/age_graph.py`
- Modify: `src/engine/store/postgres.py` (add `record_committed_intent`)
- Modify: `src/engine/pdes/engine.py` (`_commit_single` calls the combined helper)
- Test: `tests/test_physiology_loop.py` (append, DB-gated)

- [ ] **Step 1: Write the failing DB-gated test**

```python
async def test_committed_med_intent_writes_mar_and_graph(store):
    from datetime import datetime, timezone
    t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    await store.record_committed_intent(
        patient_id=9999, time=t, agent_id="R1", action_type="ADMINISTER_MED",
        payload={"drug_name": "Propofol", "dose": 100.0, "unit": "mg", "route": "IV_bolus"},
        based_on_hash="abc123",
    )
    mar = await store.fetch("SELECT drug_name, dose FROM mar WHERE patient_id=9999")
    assert mar and mar[0]["drug_name"] == "Propofol"
    acts = await store.fetch("SELECT action_type FROM agent_actions WHERE agent_id='R1'")
    assert acts and acts[0]["action_type"] == "ADMINISTER_MED"
    from engine.store.age_graph import fetch_orders
    edges = await fetch_orders(store, agent_id="R1")
    assert any("Propofol" in e for e in edges)
    # cleanup: delete mar/agent_actions rows and graph edge for R1/9999
```
Adjust the `mar` column names to the actual schema in `02_timescale_schema.sql` before running.

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_physiology_loop.py -k mar_and_graph -v`
Expected: FAIL — helper missing.

- [ ] **Step 3: Implement**

`age_graph.py` — the ONLY module emitting Cypher:
```python
"""Thin Cypher layer for Apache AGE. Translates committed intents into graph
mutations, e.g. (Agent)-[:ORDERED {time}]->(Drug). Executed inside the SAME
transaction as the Timescale writes (v4 eval Risk 3: single ACID transaction)."""

async def record_med_order(conn, agent_id: str, drug_name: str, dose: float, time) -> None:
    ...  # SELECT * FROM cypher('clinical_graph', $$ MERGE agent/drug nodes, CREATE ORDERED edge $$)

async def fetch_orders(store, agent_id: str) -> list[str]:
    ...  # read helper; reused by Phase 4 deliberator context
```
`PostgresStore.record_committed_intent(...)`: open one transaction (`async with conn.transaction():`), insert into `agent_actions`, insert into `mar` when `action_type == "ADMINISTER_MED"`, and call `age_graph.record_med_order` on the same connection. Then change `PDES_Engine._commit_single` to call `store.record_committed_intent(...)` instead of the bare `agent_actions` insert (keep the Pulse `apply_action` call from Task 1.5(d) after it).

- [ ] **Step 4: Run to verify it passes; run the full suite; commit**

Run: `docker compose run --rm engine python -m pytest tests/ -v`
```bash
git add src/engine/store/age_graph.py src/engine/store/postgres.py src/engine/pdes/engine.py tests/test_physiology_loop.py
git commit -m "feat(store): atomic agent_actions + MAR + AGE graph writes on intent commit"
```

### Task 1.8: Performance budget benchmark

Pulse runs roughly 5–10× real time single-threaded, and `save_state()` (taken every barrier for rollback) is not free — at 15 s intubation barriers it could dominate wall time. Measure BEFORE Phase 5 commits to cohort scale.

**Files:**
- Create: `scripts/bench_pulse.py`
- Create: `docs/perf.md` (results)

- [ ] **Step 1: Write `scripts/bench_pulse.py`** measuring, for the standard 65 M patient:
  1. sim-seconds per wall-second of bare `advance()` (10 sim-min run);
  2. `save_state()`/`load_state()` latency and blob size;
  3. charting overhead: advance loop at `charting_interval_s` of 1 s vs 5 s vs one-shot.

- [ ] **Step 2: Run it and record results in `docs/perf.md`**

Run: `docker compose run --rm engine python scripts/bench_pulse.py`
Decision rules (record which applied):
  - If `save_state` exceeds ~20% of wall time at 30 s barriers, add a `physiology_checkpoint_every: int` engine option — checkpoint physiology every N barriers; rollback then reverts to the latest physiology checkpoint and replays the intervening committed intents.
  - If 1 s charting costs >2× the 5 s cadence, keep 5 s as the default and note the tradeoff in `docs/perf.md`.
  - Extrapolate a 1000-patient × 6 h cohort wall-clock + storage estimate; if unacceptable, revisit before Phase 5, not during it.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_pulse.py docs/perf.md
git commit -m "perf: Pulse advance/save_state benchmark + recorded budget"
```

### Phase 1 Calibration Gate (VitalDB scalar distributions)

- [ ] Implement `src/engine/calibration/vitaldb_ref.py` (load N VitalDB cases, extract resting/intraop HR, MAP, SpO2, RR distributions) and `src/engine/calibration/compare.py` (`ks_compare(sim_values, ref_values) -> {statistic, pvalue, pass}`). Test in `tests/test_calibration.py` with a tiny in-repo fixture (do not require network in unit tests; gate the real comparison behind `-m slow`). Acceptance: simulated resting vital distributions overlap VitalDB reference within agreed tolerances (KS statistic < 0.2 for HR/MAP/SpO2). Document tolerances in the file. Commit.

---

# Phase 2 — L2: High-Frequency Waveforms

**Goal:** Sample Pulse's continuous outputs at its native rate between barriers, synthesize monitor-rate channels (ECG/ART/PLETH) from Pulse beat parameters, store windows as compressed array chunks, and surface a recent-waveform summary for agents/snapshots.

**Channels (`channel` codes):** `ART` (arterial pressure, mmHg, 125 Hz, **synthesized**), `PLETH` (pleth, unitless, 125 Hz, **synthesized**), `CO2` (capnography, mmHg, 50 Hz, Pulse-native), `AWP` (airway pressure, cmH2O, 50 Hz, Pulse-native), `FLOW` (mL/s, 50 Hz, Pulse-native), `VOL` (mL, 50 Hz, Pulse-native), `ECG_II` (mV, 250 Hz, **synthesized**).

> **Why synthesized (Arch Decision 9):** Pulse advances at a fixed 0.02 s timestep — 50 Hz is the maximum rate it can natively emit, so 125/250 Hz channels cannot come straight from Pulse. CO2/AWP/FLOW/VOL are read directly at 50 Hz during `advance()`. ART/PLETH/ECG_II morphology is generated per-beat from Pulse's beat-level outputs (HR, SBP/DBP/MAP, SpO2) by seeded, **stateful** generators (beat phase carries across windows — no seams), and calibrated against VitalDB in the Phase 2 gate. Never upsample a 50 Hz Pulse trace and label it 125 Hz.

### Task 2.1: Waveform chunk schema + store helper

**Files:**
- Create: `db/init/04_waveform_schema.sql`
- Modify: `src/engine/store/postgres.py`
- Test: `tests/test_waveforms.py`

- [ ] **Step 1: Write `db/init/04_waveform_schema.sql`**

```sql
-- Waveform storage: one row per (patient, channel, window). samples is a
-- fixed-duration array of REAL to avoid row-per-sample explosion.
CREATE TABLE IF NOT EXISTS waveform_chunks (
    time            TIMESTAMPTZ NOT NULL,   -- window start
    patient_id      INT NOT NULL,
    channel         TEXT NOT NULL,          -- ART | PLETH | CO2 | AWP | FLOW | VOL | ECG_II
    sample_rate_hz  REAL NOT NULL,
    duration_s      REAL NOT NULL,
    samples         REAL[] NOT NULL,
    UNIQUE (time, patient_id, channel)
);
SELECT create_hypertable('waveform_chunks', 'time', if_not_exists => TRUE);
ALTER TABLE waveform_chunks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'patient_id, channel'
);
SELECT add_compression_policy('waveform_chunks', INTERVAL '7 days', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_waveform_patient_channel
    ON waveform_chunks (patient_id, channel, time DESC);
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_waveforms.py
import os
from datetime import datetime, timezone
import pytest
from engine.store.postgres import PostgresStore

pytestmark = pytest.mark.skipif(os.getenv("PGHOST") is None, reason="requires Postgres")

@pytest.fixture
async def store():
    s = PostgresStore(); await s.connect()
    yield s
    await s.execute("DELETE FROM waveform_chunks WHERE patient_id = 9999")
    await s.close()

async def test_waveform_chunk_roundtrip(store):
    t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    samples = [float(i % 10) for i in range(1250)]  # 10s @ 125Hz
    await store.insert_waveform_chunk(9999, t, "ART", 125.0, 10.0, samples)
    rows = await store.fetch(
        "SELECT channel, sample_rate_hz, array_length(samples,1) AS n "
        "FROM waveform_chunks WHERE patient_id=9999"
    )
    assert rows[0]["channel"] == "ART"
    assert rows[0]["n"] == 1250
```

- [ ] **Step 3: Run to verify it fails**

Run: `docker compose down && docker compose up -d --build db && docker compose run --rm engine python -m pytest tests/test_waveforms.py -v`
Expected: FAIL — table/helper missing (rebuild `db` so the new init SQL runs).

- [ ] **Step 4: Add `insert_waveform_chunk` to `PostgresStore`**

```python
    async def insert_waveform_chunk(
        self, patient_id: int, time, channel: str,
        sample_rate_hz: float, duration_s: float, samples: list[float],
    ) -> None:
        if not self._pool:
            raise RuntimeError("PostgresStore not connected")
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO waveform_chunks (time, patient_id, channel, sample_rate_hz, duration_s, samples)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (time, patient_id, channel) DO NOTHING
                """,
                time, patient_id, channel, sample_rate_hz, duration_s, samples,
            )
```

- [ ] **Step 5: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_waveforms.py::test_waveform_chunk_roundtrip -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add db/init/04_waveform_schema.sql src/engine/store/postgres.py tests/test_waveforms.py
git commit -m "feat(store): compressed waveform_chunks hypertable + insert helper"
```

### Task 2.2: Synthetic ECG + ART/PLETH morphology generators

Pulse cannot produce monitor-rate morphology (Arch Decision 9). Synthesize a Lead-II waveform from Pulse's instantaneous heart rate using a sum-of-Gaussians PQRST model (McSharry/ECGSYN-style), and ART/PLETH beat templates scaled by Pulse's SBP/DBP. All generators are seeded AND **stateful**: beat phase, wander time, and the noise RNG carry across successive `generate` calls so consecutive 10 s windows join without seams.

**Files:**
- Create: `src/engine/physiology/ecg_synth.py`
- Create: `src/engine/physiology/pressure_synth.py`
- Test: `tests/test_ecg_synth.py`, `tests/test_waveforms.py` (pressure cases)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ecg_synth.py
import numpy as np
from engine.physiology.ecg_synth import ECGSynthesizer

def test_ecg_has_expected_rate():
    ecg = ECGSynthesizer(seed=1, sample_rate_hz=250.0)
    sig = ecg.generate(duration_s=10.0, heart_rate_bpm=60.0)
    assert len(sig) == 2500
    # ~1 R-peak per second at 60 bpm => ~10 peaks in 10s
    peaks = ecg.detect_r_peaks(sig)
    assert 8 <= len(peaks) <= 12, f"expected ~10 R-peaks, got {len(peaks)}"

def test_ecg_is_deterministic():
    a = ECGSynthesizer(seed=7, sample_rate_hz=250.0).generate(5.0, 75.0)
    b = ECGSynthesizer(seed=7, sample_rate_hz=250.0).generate(5.0, 75.0)
    assert np.allclose(a, b)

def test_ecg_is_continuous_across_windows():
    ecg = ECGSynthesizer(seed=3, sample_rate_hz=250.0)
    joined = np.concatenate([ecg.generate(10.0, 72.0), ecg.generate(10.0, 72.0)])
    whole = ECGSynthesizer(seed=3, sample_rate_hz=250.0).generate(20.0, 72.0)
    assert np.allclose(joined, whole), "generator must be stateful: no seam at window edges"

def test_ecg_accepts_instantaneous_hr_series():
    ecg = ECGSynthesizer(seed=5, sample_rate_hz=250.0)
    hr = np.linspace(60.0, 90.0, 2500)   # HR ramp over 10s => beat-to-beat variability
    sig = ecg.generate(10.0, hr)
    assert len(sig) == 2500
```
```python
# append to tests/test_waveforms.py
def test_art_waveform_tracks_pressures():
    from engine.physiology.pressure_synth import PressureSynthesizer
    ps = PressureSynthesizer(seed=1, sample_rate_hz=125.0)
    sig = ps.generate_art(duration_s=10.0, heart_rate_bpm=60.0, sbp_mmhg=120.0, dbp_mmhg=80.0)
    assert len(sig) == 1250
    assert 114.0 <= sig.max() <= 126.0, "systolic peaks must track SBP"
    assert 74.0 <= sig.min() <= 84.0, "diastolic troughs must track DBP"

def test_pleth_is_normalized_and_periodic():
    from engine.physiology.pressure_synth import PressureSynthesizer
    ps = PressureSynthesizer(seed=1, sample_rate_hz=125.0)
    sig = ps.generate_pleth(duration_s=10.0, heart_rate_bpm=60.0, spo2_pct=98.0)
    assert len(sig) == 1250 and 0.0 <= sig.min() and sig.max() <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_ecg_synth.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `ecg_synth.py` and `pressure_synth.py`**

```python
"""Synthetic Lead-II ECG via a sum-of-Gaussians PQRST model.

Driven by an instantaneous heart rate (scalar or per-sample series from Pulse).
STATEFUL: beat phase, wander time, and the noise RNG carry across generate()
calls so successive windows are seamless. Seeded for reproducibility.
Calibrated against VitalDB ECG_II spectra in the Phase 2 calibration gate.
"""
from __future__ import annotations

import numpy as np

# (center_angle_rad, gaussian_width, amplitude_mV) for P,Q,R,S,T
_WAVES = [
    (-np.pi/3,  0.25,  0.08),   # P
    (-np.pi/12, 0.10, -0.12),   # Q
    (0.0,       0.10,  1.10),   # R
    (np.pi/12,  0.10, -0.25),   # S
    (np.pi/2,   0.40,  0.30),   # T
]


class ECGSynthesizer:
    def __init__(self, seed: int, sample_rate_hz: float = 250.0,
                 noise_mv: float = 0.02, wander_mv: float = 0.05) -> None:
        self.fs = sample_rate_hz
        self.rng = np.random.default_rng(seed)
        self.noise_mv = noise_mv
        self.wander_mv = wander_mv
        self._t = 0.0            # continuous time across calls (wander phase)
        self._phase = -np.pi     # beat phase, carried across calls (no seams)

    def generate(self, duration_s: float, heart_rate_bpm) -> np.ndarray:
        """heart_rate_bpm: scalar OR per-sample array — pass Pulse's instantaneous
        HR series for beat-to-beat variability; a scalar gives a fixed rate."""
        n = int(round(duration_s * self.fs))
        hr = np.broadcast_to(np.asarray(heart_rate_bpm, dtype=float), (n,))
        dphi = 2 * np.pi * np.maximum(hr, 1.0) / 60.0 / self.fs
        phase = self._phase + np.cumsum(dphi)
        self._phase = float(phase[-1])
        wrapped = np.angle(np.exp(1j * phase))
        sig = np.zeros(n)
        for center, width, amp in _WAVES:
            d = np.angle(np.exp(1j * (wrapped - center)))  # wrapped distance
            sig += amp * np.exp(-(d ** 2) / (2 * width ** 2))
        t = self._t + np.arange(n) / self.fs
        self._t = float(t[-1] + 1.0 / self.fs)
        wander = self.wander_mv * np.sin(2 * np.pi * 0.15 * t)
        noise = self.rng.normal(0.0, self.noise_mv, n)
        return sig + wander + noise

    def detect_r_peaks(self, sig: np.ndarray) -> np.ndarray:
        thr = 0.6 * np.max(sig)
        above = sig > thr
        edges = np.where((~above[:-1]) & (above[1:]))[0]
        return edges
```

`pressure_synth.py` follows the same stateful pattern: a `PressureSynthesizer(seed, sample_rate_hz=125.0)` with `generate_art(duration_s, heart_rate_bpm, sbp_mmhg, dbp_mmhg)` (per-beat template: fast systolic upstroke, exponential diastolic decay, dicrotic notch — amplitudes scaled so peaks≈SBP, troughs≈DBP; scalar-or-series params like the ECG) and `generate_pleth(duration_s, heart_rate_bpm, spo2_pct)` (normalized 0–1 pulse contour; amplitude modulated by perfusion). Carry beat phase and RNG across calls exactly as in `ECGSynthesizer`; calibrate shapes against VitalDB `SNUADC/ART` and `SNUADC/PLETH` in the Phase 2 gate.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm engine python -m pytest tests/test_ecg_synth.py tests/test_waveforms.py -k "ecg or art or pleth" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine/physiology/ecg_synth.py src/engine/physiology/pressure_synth.py tests/test_ecg_synth.py tests/test_waveforms.py
git commit -m "feat(physiology): seeded stateful ECG + ART/PLETH morphology generators"
```

### Task 2.3: Waveform sampling driver + barrier integration

Add high-frequency sampling to `PulseAdapter` (request waveform channels at their native rates) and a `WaveformBuffer` that accumulates windows; have `_advance_physiology` flush completed windows to `waveform_chunks`.

**Files:**
- Create: `src/engine/physiology/waveforms.py`
- Modify: `src/engine/physiology/pulse_adapter.py` (add `sample_waveforms`)
- Modify: `src/engine/pdes/engine.py` (`_advance_physiology` flushes windows)
- Test: `tests/test_waveforms.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_waveform_buffer_emits_fixed_windows():
    from engine.physiology.waveforms import WaveformBuffer
    buf = WaveformBuffer(channel="ART", sample_rate_hz=125.0, window_s=10.0)
    # feed 25s of data in 5s increments
    emitted = []
    for _ in range(5):
        for w in buf.add([0.0] * 625):   # 5s @125Hz
            emitted.append(w)
    assert len(emitted) == 2            # two complete 10s windows (20s), 5s remains buffered
    assert len(emitted[0].samples) == 1250
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_waveforms.py -k buffer -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `waveforms.py`**

```python
"""Window accumulation for high-frequency waveform channels."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WaveformWindow:
    channel: str
    sample_rate_hz: float
    duration_s: float
    samples: list[float]


class WaveformBuffer:
    def __init__(self, channel: str, sample_rate_hz: float, window_s: float = 10.0) -> None:
        self.channel = channel
        self.fs = sample_rate_hz
        self.window_s = window_s
        self.window_n = int(round(sample_rate_hz * window_s))
        self._buf: list[float] = []

    def add(self, samples: list[float]):
        self._buf.extend(samples)
        while len(self._buf) >= self.window_n:
            chunk = self._buf[: self.window_n]
            self._buf = self._buf[self.window_n :]
            yield WaveformWindow(self.channel, self.fs, self.window_s, chunk)
```

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_waveforms.py -k buffer -v`
Expected: PASS.

- [ ] **Step 5: Add `sample_waveforms` to `PulseAdapter` and flush in the engine**

In `pulse_adapter.py`, add:
```python
    def sample_waveforms(self, seconds: float) -> dict[str, list[float]]:
        """Return per-channel sample arrays for the last `seconds` of advancement.

        CO2/AWP/FLOW/VOL come from Pulse's native 50 Hz (0.02 s timestep) outputs
        accumulated during advance(). ART/PLETH/ECG_II are SYNTHESIZED at monitor
        rates (125/125/250 Hz) from the beat-level HR/SBP/DBP/SpO2 series sampled
        over the same interval, via the stateful generators owned by this adapter
        (one ECGSynthesizer + one PressureSynthesizer per patient, seeded from
        self.seed) so consecutive windows join without seams. Never upsample the
        50 Hz Pulse traces to fake higher rates (Arch Decision 9).
        """
        # PULSE: collect the 50 Hz data accumulated during advance();
        # drive self._ecg / self._pressure synths from the beat-param series.
        raise NotImplementedError
```
In `engine.py::_advance_physiology`, after sampling scalars, drive buffers and flush:
```python
        if self.physiology is not None and hasattr(self.physiology, "sample_waveforms") and self.store:
            chans = self.physiology.sample_waveforms(self.barrier_seconds)
            for channel, samples in chans.items():
                buf = self._wave_buffers.setdefault(
                    channel, WaveformBuffer(channel, WAVEFORM_RATES[channel])
                )
                for window in buf.add(samples):
                    await self.store.insert_waveform_chunk(
                        self.patient_id, sample_time, window.channel,
                        window.sample_rate_hz, window.duration_s, window.samples,
                    )
```
Add module-level `WAVEFORM_RATES = {"ART":125.0,"PLETH":125.0,"CO2":50.0,"AWP":50.0,"FLOW":50.0,"VOL":50.0,"ECG_II":250.0}`, import `WaveformBuffer`, and initialize `self._wave_buffers = {}` in `__init__`.

- [ ] **Step 6: Run full suite + commit**

Run: `docker compose run --rm engine python -m pytest tests/ -v`
Expected: PASS (waveform DB writes exercised by the real-Pulse slow test).
```bash
git add src/engine/physiology/waveforms.py src/engine/physiology/pulse_adapter.py src/engine/pdes/engine.py tests/test_waveforms.py
git commit -m "feat(physiology): high-frequency waveform sampling + chunked persistence (L2)"
```

### Phase 2 Calibration Gate (VitalDB waveform fidelity)

- [ ] Extend `vitaldb_ref.py` to load matched VitalDB tracks (`SNUADC/ART`, `SNUADC/ECG_II`, `SNUADC/PLETH`, `Primus/CO2`) and compute reference power spectra + amplitude ranges. In `compare.py` add `spectral_compare(sim, ref) -> {dominant_freq_err, band_power_err, pass}`. Acceptance (document exact thresholds in code): synthesized ART/ECG/CO2 dominant frequencies track simulated HR/RR within 5%, amplitude ranges within VitalDB IQR. Mark the real comparison `-m slow`. Commit.

---

# Phase 3 — L4: OR / Anesthesia Phase State Machine

**Goal:** Model a full surgical course as an ordered set of phases, each configuring Pulse (anesthesia machine, intubation, NMB), adjusting barrier cadence (fine during induction/intubation, coarse in maintenance/ICU), and defining the expected agent roster. Drive it first with deterministic scripted actions (no LLM yet).

**Phases:** `PREOP → INDUCTION → INTUBATION → MAINTENANCE → EMERGENCE → EXTUBATION → PACU → ICU`. Events (e.g. hemorrhage, hypotension, laryngospasm) can be injected within any phase.

### Task 3.1: Phase definitions + transition rules

**Files:**
- Create: `src/engine/scenario/__init__.py`, `src/engine/scenario/phases.py`
- Test: `tests/test_scenario_machine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_machine.py
from engine.scenario.phases import Phase, PHASE_ORDER, phase_config

def test_phase_order_is_complete():
    assert PHASE_ORDER[0] == Phase.PREOP
    assert Phase.INDUCTION in PHASE_ORDER
    assert Phase.INTUBATION in PHASE_ORDER
    assert PHASE_ORDER[-1] == Phase.ICU

def test_induction_uses_fine_barriers():
    assert phase_config(Phase.INDUCTION).barrier_seconds <= 60
    assert phase_config(Phase.MAINTENANCE).barrier_seconds >= 300
    assert phase_config(Phase.ICU).barrier_seconds == 15 * 60
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_scenario_machine.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `phases.py`**

```python
"""OR/ICU phase model: ordering, per-phase Pulse configuration, barrier cadence."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    PREOP = "PREOP"
    INDUCTION = "INDUCTION"
    INTUBATION = "INTUBATION"
    MAINTENANCE = "MAINTENANCE"
    EMERGENCE = "EMERGENCE"
    EXTUBATION = "EXTUBATION"
    PACU = "PACU"
    ICU = "ICU"


PHASE_ORDER = [
    Phase.PREOP, Phase.INDUCTION, Phase.INTUBATION, Phase.MAINTENANCE,
    Phase.EMERGENCE, Phase.EXTUBATION, Phase.PACU, Phase.ICU,
]


@dataclass
class PhaseConfig:
    phase: Phase
    barrier_seconds: float
    expected_roles: list[str] = field(default_factory=list)
    # Scripted Pulse actions applied on entry (drug/vent/airway), as
    # engine.physiology.actions.PulseAction descriptors built lazily by the machine.
    entry_action_specs: list[dict] = field(default_factory=list)


_CONFIGS: dict[Phase, PhaseConfig] = {
    Phase.PREOP:       PhaseConfig(Phase.PREOP, 300, ["Anesthesiologist", "Nurse"]),
    Phase.INDUCTION:   PhaseConfig(Phase.INDUCTION, 30, ["Anesthesiologist", "CRNA", "Nurse"],
        entry_action_specs=[
            {"action_type": "ADMINISTER_MED", "drug_name": "Propofol", "dose": 150, "route": "IV_bolus"},
            {"action_type": "ADMINISTER_MED", "drug_name": "Fentanyl", "dose": 0.1, "route": "IV_bolus"},
            {"action_type": "ADMINISTER_MED", "drug_name": "Rocuronium", "dose": 50, "route": "IV_bolus"},
        ]),
    Phase.INTUBATION:  PhaseConfig(Phase.INTUBATION, 15, ["Anesthesiologist", "CRNA"],
        entry_action_specs=[{"action_type": "INTUBATE", "airway": "ETT"},
                            {"action_type": "ADJUST_VENT", "peep": 5, "tidal_volume": 480, "rate": 12, "fio2": 0.5}]),
    Phase.MAINTENANCE: PhaseConfig(Phase.MAINTENANCE, 300, ["Anesthesiologist", "CRNA", "Nurse"],
        entry_action_specs=[{"action_type": "ADMINISTER_MED", "drug_name": "Sevoflurane", "route": "inhaled", "mac": 1.0}]),
    Phase.EMERGENCE:   PhaseConfig(Phase.EMERGENCE, 60, ["Anesthesiologist", "CRNA"],
        entry_action_specs=[{"action_type": "ADMINISTER_MED", "drug_name": "Sugammadex", "dose": 200, "route": "IV_bolus"}]),
    Phase.EXTUBATION:  PhaseConfig(Phase.EXTUBATION, 30, ["Anesthesiologist", "CRNA"],
        entry_action_specs=[{"action_type": "EXTUBATE"}]),
    Phase.PACU:        PhaseConfig(Phase.PACU, 300, ["Nurse", "Anesthesiologist"]),
    Phase.ICU:         PhaseConfig(Phase.ICU, 15 * 60, ["Nurse", "Resident", "RespiratoryTherapist"]),
}


def phase_config(phase: Phase) -> PhaseConfig:
    return _CONFIGS[phase]
```

> NOTE: `INTUBATE`, `EXTUBATE`, inhaled volatile (`Sevoflurane` MAC), `Rocuronium`, and `Sugammadex` require new branches in `engine/physiology/actions.py` and new `ConstraintValidator` dose limits. Add those in Task 3.2 — do not leave them unmapped.

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_scenario_machine.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine/scenario/__init__.py src/engine/scenario/phases.py tests/test_scenario_machine.py
git commit -m "feat(scenario): OR/ICU phase definitions with per-phase cadence and actions"
```

### Task 3.2: New Pulse actions + validator limits for anesthesia

**Files:**
- Modify: `src/engine/physiology/actions.py` (add `Intubation`, `Extubation`, `InhaledAnesthetic`, NMB via `SubstanceBolus`)
- Modify: `src/engine/validators/constraint.py` (add `Rocuronium`, `Sugammadex`, `Sevoflurane` limits + `INTUBATE/EXTUBATE` pass-through)
- Test: `tests/test_scenario_machine.py` (append) and `tests/test_constraint_validator.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_scenario_machine.py
def test_intubate_maps_to_pulse_action():
    from engine.physiology.actions import intent_to_pulse_action, Intubation
    a = intent_to_pulse_action("INTUBATE", {"airway": "ETT"})
    assert isinstance(a, Intubation)

def test_inhaled_anesthetic_maps():
    from engine.physiology.actions import intent_to_pulse_action, InhaledAnesthetic
    a = intent_to_pulse_action("ADMINISTER_MED",
        {"drug_name": "Sevoflurane", "route": "inhaled", "mac": 1.0})
    assert isinstance(a, InhaledAnesthetic) and a.mac == 1.0

def test_hemorrhage_event_maps():
    from engine.physiology.actions import intent_to_pulse_action, Hemorrhage
    a = intent_to_pulse_action("INJECT_EVENT",
        {"disease_type": "OCCULT_HEMORRHAGE", "params": {"severity": 0.7}})
    assert isinstance(a, Hemorrhage) and a.severity == 0.7
```
```python
# append to tests/test_constraint_validator.py
def test_rocuronium_dose_limit():
    from engine.validators.constraint import ConstraintValidator, ValidationError
    v = ConstraintValidator()
    v.validate({"action_type": "ADMINISTER_MED", "drug_name": "Rocuronium",
                "dose": 50, "route": "IV_bolus", "is_bolus": True})  # ok
    import pytest
    with pytest.raises(ValidationError):
        v.validate({"action_type": "ADMINISTER_MED", "drug_name": "Rocuronium",
                    "dose": 500, "route": "IV_bolus", "is_bolus": True})
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm engine python -m pytest tests/test_scenario_machine.py tests/test_constraint_validator.py -k "intubate or inhaled or rocuronium" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `actions.py` add dataclasses + branches:
```python
@dataclass
class Intubation(PulseAction):
    airway: str = "ETT"
    def to_pulse(self):  # PULSE: SEIntubation(type=Tracheal)
        raise NotImplementedError

@dataclass
class Extubation(PulseAction):
    def to_pulse(self):  # PULSE: SEIntubation(type=Off) / airway action
        raise NotImplementedError

@dataclass
class InhaledAnesthetic(PulseAction):
    substance: str
    mac: float
    def to_pulse(self):  # PULSE: anesthesia machine inhaled agent + fraction from MAC
        raise NotImplementedError

@dataclass
class Hemorrhage(PulseAction):
    severity: float                 # 0..1 -> Pulse hemorrhage severity/flow
    compartment: str = "Abdomen"
    def to_pulse(self):  # PULSE: SEHemorrhage(compartment, severity)
        raise NotImplementedError
```
and in `intent_to_pulse_action`, before the final `return None`:
```python
    if action_type == "INTUBATE":
        return Intubation(airway=payload.get("airway", "ETT"))
    if action_type == "EXTUBATE":
        return Extubation()
    if action_type == "ADMINISTER_MED" and payload.get("route") == "inhaled":
        return InhaledAnesthetic(substance=_canon(payload.get("drug_name", "")),
                                 mac=float(payload.get("mac", 1.0)))
    if action_type == "INJECT_EVENT":
        if payload.get("disease_type") == "OCCULT_HEMORRHAGE":
            params = payload.get("params", {})
            return Hemorrhage(severity=float(params.get("severity", 0.5)),
                              compartment=params.get("compartment", "Abdomen"))
        # Disease types Pulse lacks fall back to the SciPy disease injector —
        # log the gap loudly; NEVER silently no-op an event (Risk 2).
        return None
```
In `constraint.py` add to `DOSE_LIMITS`:
```python
    DoseLimit("Rocuronium", "IV_bolus", max_bolus_mg=100.0, max_single_dose_mg=100.0),
    DoseLimit("Sugammadex", "IV_bolus", max_bolus_mg=800.0, max_single_dose_mg=800.0),
```
and in `validate`, allow `INTUBATE`/`EXTUBATE` to pass through (treat like `CALL_ATTENDING`).

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm engine python -m pytest tests/test_scenario_machine.py tests/test_constraint_validator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine/physiology/actions.py src/engine/validators/constraint.py tests/test_scenario_machine.py tests/test_constraint_validator.py
git commit -m "feat(scenario): anesthesia actions (intubation, NMB, volatile) + dose limits"
```

### Task 3.3: `ScenarioStateMachine` as a policy over `PDES_Engine`

> **Architecture (Decision 7):** the scenario machine must NOT be a second orchestration loop. `PDES_Engine` stays the ONLY loop that advances physiology, validates, commits, and persists. The machine is a *policy object* the engine consults each barrier for (a) the current phase's barrier cadence, (b) phase-entry and scheduled-event action specs — which flow through the SAME validate → commit → `apply_action` → MAR/`agent_actions` path as agent intents (agent_id `"SCENARIO"`). This gives scripted OR actions the identical audit trail, rollback behavior, and safety validation as LLM agents, and makes Phase 4 a pure addition (register deliberators on the same engine) instead of a mode switch.

**Files:**
- Create: `src/engine/scenario/machine.py`, `src/engine/scenario/library.py`
- Modify: `src/engine/pdes/engine.py` (accept a `scenario` policy; commit its action specs each barrier)
- Modify: `src/engine/pdes/clock.py` (variable barrier interval)
- Test: `tests/test_scenario_machine.py` (append integration tests with fake adapter)

- [ ] **Step 1: Write the failing integration tests**

```python
def test_full_case_visits_all_phases_with_fake_adapter():
    import asyncio
    from engine.scenario.machine import ScenarioStateMachine
    from engine.scenario.library import elective_lap_chole

    class FakeAdapter:
        def __init__(self): self.actions = []; self.t = 0.0
        def initialize(self): pass
        def advance(self, s): self.t += s
        def apply_action(self, a): self.actions.append(type(a).__name__)
        def sample_scalars(self): return {"hr": 75.0, "map": 80.0}
        def save_state(self): return b""
        def load_state(self, b): pass
        def close(self): pass

    fake = FakeAdapter()
    sm = ScenarioStateMachine(scenario=elective_lap_chole(), physiology=fake, store=None)
    visited = []
    sm.on_phase_enter = lambda p: visited.append(p)
    asyncio.run(sm.run())
    from engine.scenario.phases import PHASE_ORDER
    assert visited == PHASE_ORDER
    assert "Intubation" in fake.actions and "SubstanceBolus" in fake.actions
    total = sum(elective_lap_chole().phase_durations_s.values())
    assert fake.t == float(total), "engine (not the machine) must advance exactly the case duration"

def test_scenario_events_reach_physiology():
    import asyncio
    from engine.scenario.machine import ScenarioStateMachine
    from engine.scenario.library import trauma_laparotomy

    class FakeAdapter:
        def __init__(self): self.actions = []; self.t = 0.0
        def initialize(self): pass
        def advance(self, s): self.t += s
        def apply_action(self, a): self.actions.append(type(a).__name__)
        def sample_scalars(self): return {"hr": 75.0, "map": 80.0}
        def save_state(self): return b""
        def load_state(self, b): pass
        def close(self): pass

    fake = FakeAdapter()
    sm = ScenarioStateMachine(scenario=trauma_laparotomy(), physiology=fake, store=None)
    asyncio.run(sm.run())
    assert "Hemorrhage" in fake.actions, "scenario.events must be consumed, not dead data"
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_scenario_machine.py -k full_case -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement `library.py` and `machine.py`**

`library.py`:
```python
"""Built-in surgical scenarios: phase durations + injected events."""
from __future__ import annotations
from dataclasses import dataclass, field
from engine.scenario.phases import Phase, PHASE_ORDER

@dataclass
class Scenario:
    name: str
    phase_durations_s: dict          # Phase -> seconds spent in phase
    events: list[dict] = field(default_factory=list)   # [{phase, at_s, disease_type, params}]

def elective_lap_chole() -> Scenario:
    return Scenario(
        name="elective_laparoscopic_cholecystectomy",
        phase_durations_s={
            Phase.PREOP: 600, Phase.INDUCTION: 120, Phase.INTUBATION: 60,
            Phase.MAINTENANCE: 3600, Phase.EMERGENCE: 300, Phase.EXTUBATION: 120,
            Phase.PACU: 1800, Phase.ICU: 3600,
        },
    )

def trauma_laparotomy() -> Scenario:
    s = elective_lap_chole()
    s.name = "trauma_exploratory_laparotomy"
    s.phase_durations_s[Phase.MAINTENANCE] = 7200
    s.events = [{"phase": Phase.MAINTENANCE, "at_s": 1800,
                 "disease_type": "OCCULT_HEMORRHAGE", "params": {"severity": 0.7}}]
    return s
```

`machine.py` — a policy plus a thin `run()` convenience wrapper; it owns NO advance/sample/persist loop:
```python
"""Scenario as a *policy* over PDES_Engine — one orchestration loop (Arch Decision 7).

ScenarioStateMachine never advances physiology or writes vitals itself. It:
  1. tells the engine the barrier cadence for the current phase,
  2. emits phase-entry and scheduled-event action specs, which the engine
     validates and commits as agent_id "SCENARIO" intents (same audit/MAR/
     Pulse path as agent intents),
  3. reports phase transitions via on_phase_enter.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from engine.scenario.phases import PHASE_ORDER, phase_config


class ScenarioStateMachine:
    def __init__(self, scenario, physiology, store, start_time=None, patient_id: int = 1):
        self.scenario = scenario
        self.physiology = physiology
        self.store = store
        self.patient_id = patient_id
        self.start = start_time or datetime(2026, 1, 1, 7, 0, tzinfo=timezone.utc)
        self.on_phase_enter = None
        self._entered_idx = -1
        self._fired_events: set[int] = set()

    # ---- policy interface consumed by PDES_Engine ----

    def phase_at(self, elapsed_s: float):
        acc = 0.0
        for ph in PHASE_ORDER:
            acc += self.scenario.phase_durations_s[ph]
            if elapsed_s < acc:
                return ph
        return PHASE_ORDER[-1]

    def _phase_start_s(self, phase) -> float:
        acc = 0.0
        for ph in PHASE_ORDER:
            if ph is phase:
                return acc
            acc += self.scenario.phase_durations_s[ph]
        return acc

    def barrier_seconds(self, elapsed_s: float) -> float:
        return phase_config(self.phase_at(elapsed_s)).barrier_seconds

    def pending_action_specs(self, elapsed_s: float) -> list[dict]:
        """Phase-entry actions on transition + scenario events that came due."""
        specs: list[dict] = []
        ph = self.phase_at(elapsed_s)
        idx = PHASE_ORDER.index(ph)
        if idx != self._entered_idx:
            # fire entry actions for every phase crossed since the last barrier
            for i in range(self._entered_idx + 1, idx + 1):
                if self.on_phase_enter:
                    self.on_phase_enter(PHASE_ORDER[i])
                specs += phase_config(PHASE_ORDER[i]).entry_action_specs
            self._entered_idx = idx
        for i, ev in enumerate(self.scenario.events):
            due_at = self._phase_start_s(ev["phase"]) + ev["at_s"]
            if i not in self._fired_events and elapsed_s >= due_at:
                self._fired_events.add(i)
                specs.append({"action_type": "INJECT_EVENT",
                              "disease_type": ev["disease_type"], "params": ev["params"]})
        return specs

    # ---- convenience wrapper: build the engine and run the whole case ----

    async def run(self) -> None:
        from engine.pdes.engine import PDES_Engine
        total = sum(self.scenario.phase_durations_s.values())
        eng = PDES_Engine(store=self.store, start_time=self.start,
                          end_time=self.start + timedelta(seconds=total),
                          patient_id=self.patient_id, physiology=self.physiology,
                          scenario=self)
        await eng.run()
```

Engine + clock modifications:
- `pdes/clock.py`: make the barrier interval mutable — `PDESClock.barriers()` reads `self.interval_s` (default `BARRIER_MINUTES * 60`) at each iteration instead of the fixed class constant, so cadence can change between barriers.
- `pdes/engine.py`: `__init__` gains `scenario=None`. At the top of `_process_barrier`, when a scenario is set:
```python
        if self.scenario is not None:
            elapsed = (barrier.sim_time - self.clock.start_time).total_seconds()
            self.barrier_seconds = self.scenario.barrier_seconds(elapsed)
            self.clock.interval_s = self.barrier_seconds
```
  and, after the snapshot is built (so `based_on_hash` is current), validate + commit each scripted spec through the normal path:
```python
        if self.scenario is not None:
            for spec in self.scenario.pending_action_specs(elapsed):
                payload = {k: v for k, v in spec.items() if k != "action_type"}
                try:
                    self._validator.validate({"action_type": spec["action_type"], **payload})
                except ValidationError:
                    logger.error("scenario_action_failed_validation", spec=spec)  # scripts must be safe
                    continue
                record = IntentRecord(agent_id="SCENARIO", action_type=spec["action_type"],
                                      payload=payload, proposed_time=barrier.sim_time,
                                      based_on_hash=snapshot.version_hash, raw=None)
                await self._commit_single(record, snapshot.version_hash)
```
  (add `self._validator = ConstraintValidator()` in `__init__`; `_commit_single` already writes `agent_actions`/MAR/graph via Task 1.7 and applies the Pulse action via Task 1.5(d) — scripted actions get the full treatment for free).

> NOTE: In Phase 4, LLM deliberators simply `register_agent` on the same engine; scripted scenario actions and agent intents coexist on the one loop — there is no scripted-vs-agent mode switch to remove.

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_scenario_machine.py -v`
Expected: PASS.

- [ ] **Step 5: Add a real-Pulse slow end-to-end scenario test + commit**

Append a `@pytest.mark.slow` test running `elective_lap_chole()` with a real `PulseAdapter`, asserting MAP drops after induction and recovers by PACU. Then:
```bash
git add src/engine/scenario/machine.py src/engine/scenario/library.py src/engine/pdes/engine.py src/engine/pdes/clock.py tests/test_scenario_machine.py
git commit -m "feat(scenario): scenario policy over PDES_Engine + built-in surgical cases (Phase 3)"
```

---

# Phase 4 — L3: LLM Clinical Agents

**Goal:** Add LLM-driven clinicians as engine agents alongside the scripted scenario policy (Arch Decision 7 — no mode switch), served by an OpenAI-compatible endpoint: local vLLM by default; a cloud endpoint is allowed **iff Zero Data Retention is enabled** (Arch Decision 10, enforced in Task 4.1). Temperature 0.3, JSON-structured output. Agents read the snapshot + recent waveform/vitals summary + AGE context (Task 1.7's `fetch_orders`) + episodic memory (pgvector), and emit validated `AgentIntent`s. They also write narrative notes calibrated to MIMIC-IV-Note, persisted to `clinical_notes` (Task 4.4).

### Task 4.1: vLLM client + persona prompts

**Files:**
- Create: `src/engine/agents/__init__.py`, `src/engine/agents/client.py`, `src/engine/agents/personas.py`
- Test: `tests/test_agents.py` (mock the HTTP client — no live model in unit tests)

- [ ] **Step 1: Write the failing test (mocked client)**

```python
# tests/test_agents.py
import json
from engine.agents.client import LLMClient
from engine.agents.personas import persona_prompt

def test_persona_prompt_includes_role_and_constraints():
    p = persona_prompt("Anesthesiologist")
    assert "Anesthesiologist" in p
    assert "JSON" in p  # must instruct structured output

def test_cloud_endpoint_requires_zdr(monkeypatch):
    import pytest
    monkeypatch.delenv("LLM_ZDR_CONFIRMED", raising=False)
    with pytest.raises(RuntimeError, match="Zero Data Retention"):
        LLMClient(base_url="https://api.example.com/v1", model="m")
    monkeypatch.setenv("LLM_ZDR_CONFIRMED", "1")
    LLMClient(base_url="https://api.example.com/v1", model="m")   # ZDR confirmed -> ok
    LLMClient(base_url="http://localhost:8000/v1", model="m")     # local never needs the flag

def test_llm_client_parses_json_intent(monkeypatch):
    client = LLMClient(base_url="http://localhost:8000/v1", model="m", temperature=0.3)
    def fake_complete(system, user):
        return json.dumps({"action_type": "ADMINISTER_MED",
            "payload": {"drug_name": "Propofol", "dose": 100, "route": "IV_bolus"},
            "rationale": "induction"})
    monkeypatch.setattr(client, "_complete", fake_complete)
    out = client.decide(system="s", user="u")
    assert out["action_type"] == "ADMINISTER_MED"
    assert out["payload"]["drug_name"] == "Propofol"
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_agents.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement `client.py` and `personas.py`**

`client.py`:
```python
"""OpenAI-compatible client — local vLLM by default. Temperature locked low for
clinical determinism; clinical inexperience is expressed via the persona prompt,
not via sampling temperature (per v4 spec).

DUA guard (Arch Decision 10): MIMIC-derived text may only leave the machine to
an endpoint with Zero Data Retention. Cloud base_urls therefore require
LLM_ZDR_CONFIRMED=1 — set that flag ONLY on accounts where ZDR is actually
enabled with the provider."""
from __future__ import annotations
import json
import os
from urllib.parse import urlparse
from openai import OpenAI

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")


def _is_local(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in _LOCAL_HOSTS or host.endswith(".local") or host.startswith(("10.", "192.168."))


class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.3) -> None:
        if not _is_local(base_url) and os.getenv("LLM_ZDR_CONFIRMED") != "1":
            raise RuntimeError(
                f"Cloud LLM endpoint {base_url!r} requires Zero Data Retention: "
                "set LLM_ZDR_CONFIRMED=1 only if ZDR is enabled on the provider account "
                "(PhysioNet DUA — see Architecture Decision 10)."
            )
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "not-needed"))

    def _complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content

    def decide(self, system: str, user: str) -> dict:
        raw = self._complete(system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"action_type": "CALL_ATTENDING", "payload": {}, "rationale": "unparseable LLM output"}
```

`personas.py`:
```python
"""Role-specific system prompts. Each must (a) define scope of practice,
(b) enumerate the allowed action_type vocabulary, (c) require strict JSON output."""
from __future__ import annotations

_ALLOWED = ("ADMINISTER_MED, ADJUST_VENT, INTUBATE, EXTUBATE, CALL_ATTENDING, CHECK_VITALS")

_BASE = (
    "You are a {role} in a {setting}. Decide the single next clinical action.\n"
    "Allowed action_type values: " + _ALLOWED + ".\n"
    "Respond ONLY with a JSON object: "
    '{{"action_type": <str>, "payload": <object>, "rationale": <str>}}.\n'
    "Doses must be safe and route-appropriate. If unsure or out of scope, "
    'use action_type "CALL_ATTENDING".'
)

_ROLES = {
    "Anesthesiologist": "managing induction, maintenance, and emergence in the OR",
    "CRNA": "delivering anesthesia under supervision in the OR",
    "Nurse": "monitoring and administering ordered medications",
    "Resident": "managing the patient in the surgical ICU",
    "RespiratoryTherapist": "managing the ventilator and airway",
    "Attending": "supervising and handling escalations",
}

def persona_prompt(role: str, setting: str = "surgical ICU") -> str:
    return _BASE.format(role=role, setting=setting) + f"\nRole focus: {_ROLES.get(role, 'clinician')}."
```

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_agents.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine/agents/__init__.py src/engine/agents/client.py src/engine/agents/personas.py tests/test_agents.py
git commit -m "feat(agents): vLLM client + clinician persona prompts"
```

### Task 4.2: Deliberator producing validated AgentIntents

**Files:**
- Create: `src/engine/agents/deliberator.py`
- Test: `tests/test_agents.py` (append)

- [ ] **Step 1: Write the failing test (mock LLMClient + ConstraintValidator)**

```python
def test_deliberator_returns_agent_intent_and_rejects_unsafe(monkeypatch):
    from engine.agents.deliberator import LLMDeliberator
    from engine.proto.v1 import simulation_pb2 as pb

    class StubClient:
        def __init__(self, decision): self.decision = decision
        def decide(self, system, user): return self.decision

    snap = pb.StateSnapshot(version_hash="abc123")
    # Unsafe dose -> deliberator must downgrade to CALL_ATTENDING after validation fails
    delib = LLMDeliberator(agent_id="R1", role="Resident",
        client=StubClient({"action_type": "ADMINISTER_MED",
            "payload": {"drug_name": "Propofol", "dose": 5000, "route": "IV_bolus"},
            "rationale": "x"}))
    intent = delib(snap)
    assert isinstance(intent, pb.AgentIntent)
    assert intent.based_on_hash == "abc123"
    assert intent.action_type == "CALL_ATTENDING"  # unsafe action rejected
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_agents.py -k deliberator -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `deliberator.py`**

```python
"""Turns a StateSnapshot into a validated AgentIntent via the LLM client.

Implements the Callable[[StateSnapshot], AgentIntent] contract that
PDES_Engine.register_agent expects, so LLM agents are drop-in replacements
for the stub Python agents."""
from __future__ import annotations
import json
from engine.agents.personas import persona_prompt
from engine.proto.v1 import simulation_pb2 as pb
from engine.validators.constraint import ConstraintValidator, ValidationError


def _snapshot_to_text(snapshot: pb.StateSnapshot) -> str:
    vitals = ", ".join(f"{v.metric_code}={v.value:.1f}{v.unit}" for v in snapshot.vitals)
    drugs = ", ".join(f"{d.drug_name}@{d.current_rate}{d.unit}" for d in snapshot.active_drugs)
    return f"SDI={snapshot.sdi_score:.2f}. Vitals: {vitals or 'n/a'}. Active drugs: {drugs or 'none'}."


class LLMDeliberator:
    def __init__(self, agent_id: str, role: str, client, setting: str = "surgical ICU",
                 validator: ConstraintValidator | None = None, memory=None) -> None:
        self.agent_id = agent_id
        self.role = role
        self.client = client
        self.setting = setting
        self.validator = validator or ConstraintValidator()
        self.memory = memory

    def __call__(self, snapshot: pb.StateSnapshot) -> pb.AgentIntent:
        system = persona_prompt(self.role, self.setting)
        context = _snapshot_to_text(snapshot)
        if self.memory is not None:
            context += "\nRecent: " + self.memory.recall(self.agent_id, context)
        decision = self.client.decide(system=system, user=context)
        action_type = decision.get("action_type", "CALL_ATTENDING")
        payload = decision.get("payload", {})
        try:
            self.validator.validate({"action_type": action_type, **payload})
        except ValidationError:
            action_type, payload = "CALL_ATTENDING", {"reason": "validation_failed"}
        intent = pb.AgentIntent(agent_id=self.agent_id, action_type=action_type,
            payload=json.dumps(payload).encode("utf-8"), based_on_hash=snapshot.version_hash)
        intent.proposed_time.CopyFrom(snapshot.sim_time)
        if self.memory is not None:
            self.memory.remember(self.agent_id, context, decision.get("rationale", ""))
        return intent
```

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm engine python -m pytest tests/test_agents.py -k deliberator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/engine/agents/deliberator.py tests/test_agents.py
git commit -m "feat(agents): LLM deliberator emitting validated AgentIntents"
```

### Task 4.3: pgvector episodic memory

**Files:**
- Create: `db/init/05_agent_memory_schema.sql`, `src/engine/agents/memory.py`
- Modify: `src/engine/store/postgres.py` (vector insert/search helpers)
- Test: `tests/test_agents.py` (append, DB-gated)

- [ ] **Step 1: Write `db/init/05_agent_memory_schema.sql`**

```sql
CREATE TABLE IF NOT EXISTS agent_memory (
    id          BIGSERIAL PRIMARY KEY,
    time        TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent_id    TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL          -- match the embedding model dim
);
CREATE INDEX IF NOT EXISTS idx_agent_memory_ivfflat
    ON agent_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_agent_memory_agent ON agent_memory (agent_id, time DESC);
```

- [ ] **Step 2: Write the failing DB-gated test**

```python
import os, pytest
pytestmark = pytest.mark.skipif(os.getenv("PGHOST") is None, reason="requires Postgres")

async def test_memory_recall_returns_relevant(monkeypatch):
    from engine.store.postgres import PostgresStore
    from engine.agents.memory import EpisodicMemory
    store = PostgresStore(); await store.connect()
    # deterministic fake embedder so the test is stable
    mem = EpisodicMemory(store, embed=lambda s: [float(len(s) % 10)] * 384)
    await mem.aremember("R1", "MAP dropping, gave fluids")
    hits = await mem.arecall("R1", "MAP dropping, gave fluids")
    assert any("fluids" in h for h in hits)
    await store.execute("DELETE FROM agent_memory WHERE agent_id='R1'")
    await store.close()
```

- [ ] **Step 3: Run to verify it fails**

Run: `docker compose down && docker compose up -d --build db && docker compose run --rm engine python -m pytest tests/test_agents.py -k memory -v`
Expected: FAIL — module/helpers missing.

- [ ] **Step 4: Implement `memory.py` + store helpers**

Add to `PostgresStore`:
```python
    async def insert_embedding(self, agent_id: str, content: str, embedding: list[float]) -> None:
        if not self._pool: raise RuntimeError("PostgresStore not connected")
        vec = "[" + ",".join(str(x) for x in embedding) + "]"
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO agent_memory (agent_id, content, embedding) VALUES ($1,$2,$3::vector)",
                agent_id, content, vec)

    async def search_embeddings(self, agent_id: str, embedding: list[float], k: int = 3) -> list[str]:
        if not self._pool: raise RuntimeError("PostgresStore not connected")
        vec = "[" + ",".join(str(x) for x in embedding) + "]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT content FROM agent_memory WHERE agent_id=$1 "
                "ORDER BY embedding <=> $2::vector LIMIT $3", agent_id, vec, k)
        return [r["content"] for r in rows]
```
`memory.py`:
```python
"""pgvector-backed episodic memory. `embed` is injectable so tests can use a
deterministic stub; production uses default_embedder() (384-dim, matches the
VECTOR(384) schema)."""
from __future__ import annotations


def default_embedder():
    """Production embedder: all-MiniLM-L6-v2 (384-dim). Loaded lazily and once —
    the model download happens at image build / first use, never per call.
    `sentence-transformers` is already in requirements.txt (Task 0.3)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return lambda text: model.encode([text], normalize_embeddings=True)[0].tolist()


class EpisodicMemory:
    def __init__(self, store, embed) -> None:
        self.store = store
        self.embed = embed

    async def aremember(self, agent_id: str, content: str) -> None:
        await self.store.insert_embedding(agent_id, content, self.embed(content))

    async def arecall(self, agent_id: str, query: str, k: int = 3) -> list[str]:
        return await self.store.search_embeddings(agent_id, self.embed(query), k)
```
Add a fast sanity test (no DB): `default_embedder()("MAP dropping")` returns a list of 384 floats — mark it `-m slow` if model download is unacceptable in the unit suite.

> NOTE: `LLMDeliberator` uses the synchronous `recall`/`remember` names. Provide thin sync wrappers in production (run the coroutine on the engine loop) or refactor the deliberator to async when integrating; keep the embedding dim (384) consistent with the schema and chosen model.

- [ ] **Step 5: Run to verify it passes + commit**

Run: `docker compose run --rm engine python -m pytest tests/test_agents.py -k memory -v`
Expected: PASS.
```bash
git add db/init/05_agent_memory_schema.sql src/engine/agents/memory.py src/engine/store/postgres.py tests/test_agents.py
git commit -m "feat(agents): pgvector episodic memory"
```

### Task 4.4: Narrative note generation, persistence, and trigger wiring

Notes must actually be produced during runs and land in the database — a `NoteWriter` nobody calls and notes nobody stores would make the Phase 5 "notes exported" claim false.

**Files:**
- Create: `src/engine/agents/notes.py`
- Create: `db/init/07_clinical_notes_schema.sql`
- Modify: `src/engine/store/postgres.py` (add `insert_note`)
- Modify: `src/engine/scenario/machine.py` (note triggers on phase transitions)
- Modify: `proto/simulation.proto` (add `ClinicalNote` message), regenerate stubs via `make proto`
- Test: `tests/test_agents.py` (append, mocked client)

- [ ] **Step 0: Write `db/init/07_clinical_notes_schema.sql`**

```sql
CREATE TABLE IF NOT EXISTS clinical_notes (
    id          BIGSERIAL PRIMARY KEY,
    time        TIMESTAMPTZ NOT NULL,
    patient_id  INT NOT NULL,
    author_role TEXT NOT NULL,
    note_type   TEXT NOT NULL,    -- PREOP | INTRAOP | PROGRESS | HANDOFF
    body        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_patient_time ON clinical_notes (patient_id, time DESC);
```
Add a matching `PostgresStore.insert_note(patient_id, time, author_role, note_type, body)` helper.

- [ ] **Step 1: Add `ClinicalNote` to `proto/simulation.proto`** (after `PatientEvent`)

```protobuf
message ClinicalNote {
  string author_role = 1;     // Anesthesiologist | Nurse | Resident | ...
  string note_type = 2;       // PREOP | INTRAOP | PROGRESS | HANDOFF
  string body = 3;            // free text, calibrated to MIMIC-IV-Note style
  google.protobuf.Timestamp authored_at = 4;
  string patient_id = 5;
}
```

- [ ] **Step 2: Regenerate stubs**

Run: `docker compose run --rm engine make proto`
Expected: `src/engine/proto/v1/simulation_pb2.py` now exposes `ClinicalNote`.

- [ ] **Step 3: Write the failing test**

```python
def test_note_generation(monkeypatch):
    from engine.agents.notes import NoteWriter
    class StubClient:
        def _complete(self, system, user): return "65M POD0 s/p lap chole, hemodynamically stable..."
        def decide(self, system, user): return {}
    nw = NoteWriter(client=StubClient())
    note = nw.write(role="Resident", note_type="PROGRESS",
                    snapshot_text="HR 80, MAP 78", patient_id="P1")
    assert note.author_role == "Resident"
    assert "stable" in note.body
```

- [ ] **Step 4: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_agents.py -k note -v`
Expected: FAIL — module missing.

- [ ] **Step 5: Implement `notes.py`**

```python
"""Generate free-text clinical notes. Prompt style is calibrated against
MIMIC-IV-Note (see calibration/mimic_ref.py): section structure, abbreviation
density, and length distribution should match real notes of the same type."""
from __future__ import annotations
from datetime import datetime, timezone
from engine.proto.v1 import simulation_pb2 as pb

_NOTE_SYSTEM = (
    "You are a {role} writing a {note_type} note. Use realistic clinical style, "
    "standard abbreviations, and a problem-oriented structure. Do not invent "
    "values; use only the provided data."
)

class NoteWriter:
    def __init__(self, client) -> None:
        self.client = client

    def write(self, role: str, note_type: str, snapshot_text: str, patient_id: str) -> pb.ClinicalNote:
        system = _NOTE_SYSTEM.format(role=role, note_type=note_type)
        body = self.client._complete(system=system, user=snapshot_text)
        note = pb.ClinicalNote(author_role=role, note_type=note_type, body=body, patient_id=patient_id)
        note.authored_at.FromDatetime(datetime.now(timezone.utc))
        return note
```

- [ ] **Step 6: Wire note triggers into the scenario policy**

Notes are generated at clinically meaningful moments, driven by the scenario policy (NOT free-running):

```python
# in src/engine/scenario/machine.py
NOTE_TRIGGERS = {
    Phase.PREOP:     ("Anesthesiologist", "PREOP"),     # pre-anesthesia assessment on entry
    Phase.EMERGENCE: ("Anesthesiologist", "INTRAOP"),   # intraop summary as the case ends
    Phase.PACU:      ("Anesthesiologist", "HANDOFF"),   # OR -> PACU handoff
    Phase.ICU:       ("Resident", "HANDOFF"),           # PACU -> ICU handoff
}
PROGRESS_NOTE_EVERY_S = 12 * 3600                        # Resident PROGRESS note per 12 ICU hours
```

Give `ScenarioStateMachine` an optional `note_writer` (a `NoteWriter` or `None`). Extend `pending_action_specs` (or a parallel `pending_notes(elapsed_s, snapshot_text)` hook the engine calls at the same point) to return due notes; the engine persists each via `store.insert_note(...)` at the barrier's sim time. With `note_writer=None` (all Phase 3 tests), nothing changes — note generation is a pure Phase 4 addition on the same loop.

- [ ] **Step 7: Run to verify it passes + commit**

Run: `docker compose run --rm engine python -m pytest tests/test_agents.py tests/test_scenario_machine.py -v`
Expected: PASS (add a mocked-client test asserting a HANDOFF note is emitted when the scenario crosses into PACU, and a DB-gated test that `insert_note` round-trips).
```bash
git add proto/simulation.proto src/engine/proto/v1/ src/engine/agents/notes.py db/init/07_clinical_notes_schema.sql src/engine/store/postgres.py src/engine/scenario/machine.py tests/
git commit -m "feat(agents): clinical note generation, persistence, and scenario-driven triggers"
```

### Phase 4 Calibration Gate (MIMIC-IV-Note + decision realism)

> **DUA constraint (Arch Decision 10):** every step in this gate handles MIMIC-derived text. It runs against the local vLLM endpoint by default; a cloud endpoint may be substituted **only** with ZDR enabled on the provider account (`LLM_ZDR_CONFIRMED=1`). The `LLMClient` guard enforces this, but the calibration scripts must also never write MIMIC text into logs, fixtures, or committed files.

- [ ] Implement `src/engine/calibration/mimic_ref.py`: load a sample of MIMIC-IV-Note records by note type; compute length distribution, abbreviation density, and section-header frequency. In `compare.py` add `note_style_compare(sim_notes, ref_notes) -> {length_ks, abbrev_err, pass}`. Acceptance (document thresholds): generated PROGRESS/INTRAOP notes match MIMIC length distribution (KS < 0.25) and contain expected section headers. Also calibrate disease/event prevalence and order patterns from MIMIC-IV `hosp`/`icu` tables to set cohort sampling priors (feeds Phase 5). Commit.

---

# Phase 5 — L5: Cohort Generation, Batch Runner, Export

**Goal:** Generate diverse seeded patient/procedure/event configurations, run many independent simulations in parallel (one process per patient), and export the multi-modal output to Parquet (tabular), WFDB (waveforms — compatible with VitalDB/MIMIC tooling), and FHIR (interoperable EHR).

### Task 5.1: Seeded cohort sampler

**Files:**
- Create: `src/engine/cohort/__init__.py`, `src/engine/cohort/sampler.py`
- Test: `tests/test_cohort.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cohort.py
from engine.cohort.sampler import sample_cohort

def test_cohort_is_deterministic_and_diverse():
    a = sample_cohort(n=50, seed=123)
    b = sample_cohort(n=50, seed=123)
    assert [p.__dict__ for p in a] == [p.__dict__ for p in b], "same seed => identical cohort"
    ages = {p.patient.age_yr for p in a}
    procs = {p.scenario_name for p in a}
    assert len(ages) > 5 and len(procs) >= 2, "cohort should be diverse"
    assert all(18 <= p.patient.age_yr <= 90 for p in a)
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_cohort.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `sampler.py`**

```python
"""Seeded sampler producing diverse patient + procedure + event specs.
Priors (age/sex/comorbidity/procedure mix, event rates) are calibrated from
MIMIC-IV and VitalDB demographics (see calibration/)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from engine.physiology.pulse_adapter import PatientConfig

_PROCEDURES = ["elective_laparoscopic_cholecystectomy", "trauma_exploratory_laparotomy"]

@dataclass
class CohortMember:
    patient_id: int
    seed: int
    patient: PatientConfig
    scenario_name: str

def sample_cohort(n: int, seed: int) -> list[CohortMember]:
    rng = np.random.default_rng(seed)
    members = []
    for i in range(n):
        sex = "MALE" if rng.random() < 0.5 else "FEMALE"
        age = int(rng.integers(18, 91))
        height = float(rng.normal(178 if sex == "MALE" else 164, 7))
        weight = float(rng.normal(85 if sex == "MALE" else 72, 15))
        proc = _PROCEDURES[int(rng.integers(0, len(_PROCEDURES)))]
        members.append(CohortMember(
            patient_id=1000 + i, seed=int(rng.integers(0, 2**31)),
            patient=PatientConfig(sex=sex, age_yr=age,
                                  height_cm=round(height, 1), weight_kg=round(max(weight, 40), 1)),
            scenario_name=proc))
    return members
```

- [ ] **Step 4: Run to verify it passes + commit**

Run: `docker compose run --rm engine python -m pytest tests/test_cohort.py -v`
Expected: PASS.
```bash
git add src/engine/cohort/__init__.py src/engine/cohort/sampler.py tests/test_cohort.py
git commit -m "feat(cohort): seeded diverse patient/procedure sampler"
```

### Task 5.2: Run registry schema + parallel batch runner

**Files:**
- Create: `db/init/06_cohort_schema.sql`, `src/engine/cohort/runner.py`, `scripts/run_cohort.py`
- Test: `tests/test_cohort.py` (append; use fake adapter + multiprocessing-safe stub)

- [ ] **Step 1: Write `db/init/06_cohort_schema.sql`**

```sql
CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id      TEXT PRIMARY KEY,
    patient_id  INT NOT NULL,
    seed        INT NOT NULL,
    scenario    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending|running|done|failed
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON simulation_runs (status);
```

- [ ] **Step 2: Write the failing test (runner orchestrates members with an injected sim function)**

```python
def test_runner_executes_all_members_in_parallel():
    from engine.cohort.runner import run_batch
    from engine.cohort.sampler import sample_cohort
    members = sample_cohort(n=8, seed=5)
    done = run_batch(members, max_workers=4, simulate=lambda m: ("ok", m.patient_id))
    assert sorted(pid for _, pid in done) == sorted(m.patient_id for m in members)
    assert all(status == "ok" for status, _ in done)
```

- [ ] **Step 3: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_cohort.py -k runner -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement `runner.py`** (process pool; `simulate` injectable for tests)

```python
"""Parallel batch runner: one OS process per patient (Pulse is single-patient
and CPU-bound, so processes — not threads — give real parallelism)."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed

def _default_simulate(member):
    # Runs a full scenario for one member with its own PulseAdapter + DB writes.
    # Records lifecycle in the simulation_runs registry — the registry, not logs,
    # is how failures are triaged and how interrupted cohorts resume.
    import asyncio
    from engine.physiology.pulse_adapter import PulseAdapter
    from engine.scenario.machine import ScenarioStateMachine
    from engine.scenario.library import elective_lap_chole, trauma_laparotomy
    from engine.store.postgres import PostgresStore

    scenario = {"elective_laparoscopic_cholecystectomy": elective_lap_chole,
                "trauma_exploratory_laparotomy": trauma_laparotomy}[member.scenario_name]()
    run_id = f"run-{member.seed}-{member.patient_id}"

    async def _go():
        store = PostgresStore(); await store.connect()
        await store.execute(
            "INSERT INTO simulation_runs (run_id, patient_id, seed, scenario, status, started_at) "
            "VALUES ($1,$2,$3,$4,'running',now()) "
            "ON CONFLICT (run_id) DO UPDATE SET status='running', started_at=now(), error=NULL",
            run_id, member.patient_id, member.seed, member.scenario_name)
        try:
            adapter = PulseAdapter(seed=member.seed, patient_cfg=member.patient)
            sm = ScenarioStateMachine(scenario=scenario, physiology=adapter, store=store,
                                      patient_id=member.patient_id)
            await sm.run(); adapter.close()
            await store.execute(
                "UPDATE simulation_runs SET status='done', finished_at=now() WHERE run_id=$1", run_id)
        except Exception as e:
            await store.execute(
                "UPDATE simulation_runs SET status='failed', finished_at=now(), error=$2 WHERE run_id=$1",
                run_id, str(e))
            raise
        finally:
            await store.close()
    asyncio.run(_go())
    return ("ok", member.patient_id)

def run_batch(members, max_workers: int = 4, simulate=None):
    simulate = simulate or _default_simulate
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(simulate, m): m for m in members}
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # record failure, keep going
                results.append(("failed:" + str(e), m.patient_id))
    return results
```

`scripts/run_cohort.py`:
```python
import argparse
from engine.cohort.sampler import sample_cohort
from engine.cohort.runner import run_batch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    members = sample_cohort(n=args.n, seed=args.seed)
    results = run_batch(members, max_workers=args.workers)
    ok = sum(1 for s, _ in results if s == "ok")
    print(f"cohort complete: {ok}/{len(results)} succeeded")

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add a DB-gated registry test, run, and commit**

Append a DB-gated test: call `_default_simulate` with a fake-adapter member (monkeypatch `PulseAdapter`) and assert the `simulation_runs` row transitions to `done` (and to `failed` with a populated `error` column when the scenario raises).

Run: `docker compose run --rm engine python -m pytest tests/test_cohort.py -v`
Expected: PASS.
```bash
git add db/init/06_cohort_schema.sql src/engine/cohort/runner.py scripts/run_cohort.py tests/test_cohort.py
git commit -m "feat(cohort): run registry (with status tracking) + parallel process-pool batch runner"
```

### Task 5.3: Multi-format exporter (Parquet / WFDB / FHIR)

**Files:**
- Create: `src/engine/cohort/export.py`
- Test: `tests/test_cohort.py` (append)

- [ ] **Step 1: Write the failing test (export from in-memory rows to a temp dir)**

```python
def test_export_vitals_to_parquet(tmp_path):
    import pyarrow.parquet as pq
    from engine.cohort.export import export_vitals_parquet
    rows = [{"time": "2026-01-01T12:00:00Z", "patient_id": 1000, "metric_code": "hr", "value": 80.0},
            {"time": "2026-01-01T12:00:00Z", "patient_id": 1000, "metric_code": "map", "value": 78.0}]
    path = tmp_path / "vitals.parquet"
    export_vitals_parquet(rows, str(path))
    table = pq.read_table(str(path))
    assert table.num_rows == 2
    assert set(table.column_names) == {"time", "patient_id", "metric_code", "value"}

def test_export_waveform_to_wfdb(tmp_path):
    from engine.cohort.export import export_waveform_wfdb
    import numpy as np
    sig = np.sin(np.linspace(0, 20*np.pi, 1250)).astype("float32")
    record = export_waveform_wfdb(str(tmp_path / "p1000_art"), {"ART": (sig, 125.0, "mmHg")})
    assert (tmp_path / "p1000_art.hea").exists()
    assert (tmp_path / "p1000_art.dat").exists()

def test_wfdb_rejects_mixed_rates_and_uses_per_channel_units(tmp_path):
    from engine.cohort.export import export_waveform_wfdb
    import numpy as np, pytest, wfdb, os
    sig = np.zeros(500, dtype="float32")
    with pytest.raises(AssertionError):
        export_waveform_wfdb(str(tmp_path / "bad"), {"ART": (sig, 125.0, "mmHg"),
                                                     "CO2": (sig, 50.0, "mmHg")})
    export_waveform_wfdb(str(tmp_path / "p1000_ecg"), {"ECG_II": (sig, 250.0, "mV")})
    rec = wfdb.rdheader(str(tmp_path / "p1000_ecg"))
    assert rec.units == ["mV"], "units must come from the channel, not a hardcoded list"

def test_export_notes_and_mar_to_parquet(tmp_path):
    import pyarrow.parquet as pq
    from engine.cohort.export import export_notes_parquet, export_mar_parquet
    notes = [{"time": "2026-01-01T14:00:00Z", "patient_id": 1000,
              "author_role": "Resident", "note_type": "HANDOFF", "body": "..."}]
    export_notes_parquet(notes, str(tmp_path / "notes.parquet"))
    assert pq.read_table(str(tmp_path / "notes.parquet")).num_rows == 1
    mar = [{"time": "2026-01-01T12:00:00Z", "patient_id": 1000,
            "drug_name": "Propofol", "dose": 100.0, "unit": "mg", "route": "IV_bolus"}]
    export_mar_parquet(mar, str(tmp_path / "mar.parquet"))
    assert pq.read_table(str(tmp_path / "mar.parquet")).num_rows == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm engine python -m pytest tests/test_cohort.py -k export -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `export.py`**

```python
"""Export synthetic data to research-standard formats:
- Parquet: tabular vitals/MAR/notes (analysis-friendly, columnar)
- WFDB: waveforms (PhysioNet standard; same toolchain as MIMIC-IV waveforms)
- FHIR: EHR interoperability (Observation + MedicationAdministration)"""
from __future__ import annotations
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import wfdb


def _rows_to_parquet(rows: list[dict], path: str, columns: tuple[str, ...]) -> None:
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in columns}), path)

def export_vitals_parquet(rows: list[dict], path: str) -> None:
    _rows_to_parquet(rows, path, ("time", "patient_id", "metric_code", "value"))

def export_mar_parquet(rows: list[dict], path: str) -> None:
    _rows_to_parquet(rows, path, ("time", "patient_id", "drug_name", "dose", "unit", "route"))

def export_notes_parquet(rows: list[dict], path: str) -> None:
    _rows_to_parquet(rows, path, ("time", "patient_id", "author_role", "note_type", "body"))

def export_waveform_wfdb(record_basepath: str, channels: dict) -> str:
    """channels: {name: (signal, fs, unit)}. One WFDB record = ONE sample rate;
    group channels by rate before calling (ART/PLETH 125 Hz, gas/vent 50 Hz,
    ECG_II 250 Hz — see the note below)."""
    names = list(channels.keys())
    rates = {channels[c][1] for c in names}
    assert len(rates) == 1, f"one WFDB record = one sample rate, got {rates}"
    sigs = np.column_stack([channels[c][0] for c in names])
    units = [channels[c][2] for c in names]
    import os
    wfdb.wrsamp(record_name=os.path.basename(record_basepath),
                fs=rates.pop(), units=units, sig_name=names,
                p_signal=sigs, write_dir=os.path.dirname(record_basepath) or ".")
    return record_basepath

def export_fhir_observation(patient_id: int, metric_code: str, value: float, when: str) -> dict:
    return {"resourceType": "Observation", "status": "final",
            "subject": {"reference": f"Patient/{patient_id}"},
            "code": {"text": metric_code}, "valueQuantity": {"value": value},
            "effectiveDateTime": when}

def export_fhir_medication_administration(
    patient_id: int, drug_name: str, dose: float, unit: str, route: str, when: str
) -> dict:
    return {"resourceType": "MedicationAdministration", "status": "completed",
            "subject": {"reference": f"Patient/{patient_id}"},
            "medicationCodeableConcept": {"text": drug_name},
            "dosage": {"dose": {"value": dose, "unit": unit}, "route": {"text": route}},
            "effectiveDateTime": when}
```

> NOTE: WFDB requires equal-length, equal-rate signals per record. Group channels by sample rate (ART/PLETH at 125 Hz in one record, CO2/AWP/FLOW/VOL at 50 Hz in another, ECG_II at 250 Hz in its own) — the assertion above enforces this. Units come from the per-channel tuple: mmHg (ART/CO2), cmH2O (AWP), mL/s (FLOW), mL (VOL), mV (ECG_II), unitless (PLETH).

- [ ] **Step 4: Run to verify it passes + commit**

Run: `docker compose run --rm engine python -m pytest tests/test_cohort.py -v`
Expected: PASS.
```bash
git add src/engine/cohort/export.py tests/test_cohort.py
git commit -m "feat(cohort): Parquet (vitals/MAR/notes) + WFDB + FHIR exporters (Phase 5)"
```

### Task 5.4: Cohort-scale acceptance run

- [ ] **Step 1: Run a small real cohort end-to-end**

Run:
```bash
docker compose up -d db
docker compose run --rm engine python scripts/run_cohort.py --n 8 --seed 1 --workers 4
```
Expected: `cohort complete: 8/8 succeeded`, with `vital_signs`, `waveform_chunks`, `mar`, and `clinical_notes` populated for `patient_id` 1000–1007, and `simulation_runs` showing 8 `done` rows (zero `failed`).

- [ ] **Step 2: Export the cohort and verify artifacts**

Add a `--export DIR` flag to `scripts/run_cohort.py` that, after the batch, pulls each patient's vitals/waveforms/MAR/notes from the DB and writes Parquet + WFDB (+ optional FHIR NDJSON) under `data/exports/<run>/`. Run it and confirm files exist and re-read cleanly with `pyarrow`/`wfdb`. Compare actual waveform storage + wall-clock against the `docs/perf.md` extrapolation from Task 1.8 and note any surprise. Commit.

---

# Cross-Cutting: Calibration Harness Summary

The calibration code (`src/engine/calibration/`) is built incrementally across the phase gates above. By the end it provides:

- `vitaldb_ref.py` — VitalDB reference distributions (scalar vitals) + waveform spectra/amplitudes, from cases selected by available tracks. The user has VitalDB downloaded locally; point `VITALDB_LOCAL_DIR` at it.
- `mimic_ref.py` — MIMIC-IV reference: demographics/comorbidity/procedure mix and event prevalence (hosp/icu tables) → cohort sampling priors; MIMIC-IV-Note style metrics → note-generation calibration.
- `compare.py` — `ks_compare`, `spectral_compare`, `note_style_compare`, each returning a `{metric..., pass: bool}` dict with documented thresholds, plus a `report(results) -> str` that prints a pass/fail table.

Calibration runs as `make calibrate` (add this target) and as `-m slow` tests, not in the fast unit suite. Each phase gate must pass before that layer is considered done.

**MIMIC handling rule (Arch Decision 10):** calibration code touching MIMIC data defaults to the local vLLM endpoint; cloud LLMs only with ZDR enabled (`LLM_ZDR_CONFIRMED=1`). MIMIC text must never land in logs, committed fixtures, or exported artifacts.

- [ ] Add a `calibrate` target to the `Makefile`:
```makefile
calibrate: ## Run the calibration harness against VitalDB + MIMIC-IV references
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_calibration.py -m slow -v
```

---

# Testing Strategy

- **Fast unit suite** (`make test`, no markers): everything that does not need Pulse, a live model, or large datasets. Uses fakes/stubs/monkeypatch. Must stay green at every commit.
- **DB-gated tests**: skipped unless `PGHOST` is set; run inside `docker compose run --rm engine`.
- **`-m slow` tests**: real Pulse, real cohorts, calibration against datasets. Run before merging a phase, not on every commit.
- **Phase acceptance gates** (must pass to call a layer done): Task 1.6 (L1), Task 2.3 + Phase 2 calibration (L2), Task 3.3 slow test (L4), Phase 4 calibration (L3), Task 5.4 (L5).
- Add `slow` to pytest markers in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = ["slow: requires Pulse, a live model, or large datasets"]
```

---

# Risk Register (monitor during execution)

1. **Pulse API drift / platform availability (MED — was HIGH).** Downgraded: Pulse ships PyPI wheels (`pulse-physiology-engine`), so Phase 0 tries pip first. Residual risk is wheel availability for linux/arm64 (Apple-Silicon host) — fallbacks in order: amd64 wheel under emulation, source build (`build_pulse.sh`), and finally driving Pulse as a subprocess over its C++ executable + protobuf scenario files. All Pulse calls stay isolated in `pulse_adapter.py`, so any fallback preserves the same interface.
2. **Pulse substance coverage (MED).** Rocuronium, sugammadex, sevoflurane, and some pressors may not ship as Pulse substances. Where missing, define custom Pulse substances from literature PK/PD, or fall back to the existing SciPy surrogate for that drug's effect while logging the gap. Never silently no-op a drug (this applies to `INJECT_EVENT` disease types too).
3. **Waveform volume at cohort scale (MED).** Chunked arrays + TimescaleDB compression keep this bounded, but a 1000-patient × 6-hour cohort with 7 channels is large. Task 1.8 produces the storage/wall-clock extrapolation BEFORE Phase 5; validate the compression ratio on the Task 5.4 run; consider exporting waveforms straight to WFDB files and storing only pointers in the DB if storage balloons.
4. **Wall-clock cost of Pulse + state checkpointing (MED).** Pulse runs ~5–10× real time single-threaded, and `save_state()` every barrier may dominate short-barrier phases (15 s intubation barriers). Task 1.8 measures both; the `physiology_checkpoint_every` escape hatch (checkpoint every N barriers + intent replay on rollback) is specified there.
5. **Synthesized waveform realism (MED).** ART/PLETH/ECG morphology is generated, not simulated (Pulse's 0.02 s timestep caps native output at 50 Hz — Arch Decision 9). The VitalDB spectral/amplitude gate in Phase 2 is the guard; document the synthesis provenance in the dataset datasheet so downstream users don't mistake generated morphology for modeled physiology.
6. **vLLM throughput (MED).** Per the v4 spec, parallel agent deliberation serializes on one GPU. Start with serial deliberation; the PDES engine already supports it. Batch agent calls per barrier through vLLM continuous batching before scaling cohorts with LLM agents enabled. If local GPU throughput blocks cohort scale, a cloud endpoint MAY be substituted **only with ZDR enabled** (Arch Decision 10).
7. **MIMIC-IV credentialing (LOW/process).** Note/waveform calibration is blocked until PhysioNet access is in place. Phases 1–3 and 5 do not depend on MIMIC; only the Phase 4 calibration gate and cohort priors do. Sequence accordingly.
8. **Determinism vs LLM (LOW).** LLM output is only weakly seedable. Treat agent decisions as a stochastic layer; keep physiology/cohort/waveform synthesis fully deterministic so a run is reproducible given the same recorded agent transcript.

---

# Definition of Done (whole program)

- `make test` green; each phase's `-m slow` acceptance gate passes.
- `scripts/run_single_sim.py` produces a physiologic multi-modal trajectory for a single patient through a full OR→ICU course, with scalar vitals at charting resolution (≤5 s), MAR rows for every administered medication, and clinical notes at the scenario's trigger points.
- `scripts/run_cohort.py --n N --export DIR` produces N reproducible patients with vitals + waveforms + MAR + notes exported to Parquet/WFDB/FHIR, and a `simulation_runs` registry row per patient reflecting the true final status.
- `make calibrate` reports PASS for VitalDB scalar + waveform gates and MIMIC-IV note-style gate within documented tolerances.
- `docs/perf.md` records measured sim-speed, `save_state` cost, and the cohort-scale extrapolation (Task 1.8).
- All Pulse access is confined to `src/engine/physiology/pulse_adapter.py`; all Cypher is confined to `src/engine/store/age_graph.py`.
- No cloud LLM call is possible without `LLM_ZDR_CONFIRMED=1` (ZDR enabled on the provider account); MIMIC text appears in no logs, fixtures, or exports.
