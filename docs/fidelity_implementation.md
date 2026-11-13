# Fidelity foundations

Implemented in the sequence: precise event timing and replay; observed versus true state; clinical challenge definitions; correlated example patients; synchronized waveforms. Sol 6.1 coordinated integration and review, with three Terra 6.1 workers assigned disjoint implementation areas. These changes provide testable simulation mechanics. Clinical thresholds, population distributions and waveform morphology still require the planned clinical review and empirical calibration.

## Timing and replay

The engine queues future intents and advances physiology to each administration time. Scenario phase boundaries and insult times cap the next barrier. Charting uses a global cadence independent of these interruptions, and simulation termination is exact even when the last interval is partial. Actions scheduled at a chart timestamp and scenario events at that boundary precede the chart sample; agent decisions based on that published observation follow it.

Checkpoint state includes the pending intent queue, observation model, scenario event flags, unreported phase transitions, state revision and output cadence, together with adapter state. The adapter preserves the Pulse state plus Python waveform buffers, RNGs, beat phase and fractional timestep carry. Sequential fallback applies each action once. Speculative audit rows and phase callbacks are staged until causal deliberation succeeds. This is an in-barrier causal rollback mechanism; it does not rewind previously published database history or arbitrary external callback side effects. A process crash during persistence is not a transactional resume protocol.

Pulse uses a finite solver timestep. Arbitrary event timestamps are honored by the scheduler, while the underlying physiology resolves changes at its native timestep. This is not sub-timestep physiological resolution. Waveform sample counts follow the solved time and sampling clocks, with fractional carry preserved.

## Truth and observations

`ConfigurableObservationModel` supports seeded noise, trailing averaging, lag, intermittent measurements and dropout. Defaults are ideal measurements. Missing observations stay missing. Agents receive observed values; truth is retained for evaluation and is excluded from database snapshot retrieval. Snapshot hashes include visible contents and committed state revision. Measurement timestamps preserve observation publication time; these are not laboratory collection timestamps.

With a store, numerical trajectories use distinct `truth` and `observed` sources. With the CLI, both trajectories are exported separately in JSON. This observation layer does not yet implement device-specific signal processing, arterial-line mechanics, asynchronous lab workflows or alarm-driven agent interruption.

## Clinical challenges and example patients

See [fidelity_challenges.md](fidelity_challenges.md) for definitions, review metadata, paired comparisons, coverage requirements and the evaluator CLI. Draft examples can pass numeric checks but cannot produce clinical approval. Approved metadata records a clinician's declaration; it is not an authentication system.

`sample_patient` produces bounded, reproducible adult, older-adult and larger-adult examples with correlated baseline inputs that are actually applied by the adapter. These distributions are examples awaiting review, rather than calibrated populations. The current sampler does not change renal clearance, frailty, lung mechanics or drug sensitivity.

## Waveforms

ART, PLETH and ECG use the same integrated cardiac phase and evolving captured values instead of interval means. Sample count carry and interpolation preserve continuity across drains. Adapter snapshots include generator state so replay reproduces noise as well as physiology. Morphology uses templates with fixed cycle-relative delays. Patient-specific conduction, pulse transit time, arrhythmias, respiratory modulation and arterial waveform MAP calibration remain future work. Native respiratory request names and units still require verification against the built Pulse version.

## Running and review

Use Python 3.13. On this checkout `.venv-fidelity` is the native verification environment; the old `.venv` contains an incompatible Intel interpreter.

```sh
PYTHONPATH=src .venv-fidelity/bin/python -m pytest -q
PYTHONPATH=src python scripts/run_single_sim.py --phenotype older_adult --seed 42 \
  --scenario elective_laparoscopic_cholecystectomy --output trajectory.json --quiet
PYTHONPATH=src python scripts/run_single_sim.py --hours 0.02 --waveforms ART PLETH ECG_II \
  --output waveforms.json --quiet
```

For monitor effects, pass `--observation-config monitor.json`, for example:

```json
{"hr": {"lag_s": 5, "averaging_s": 5}, "map": {"interval_s": 60, "noise_std": 2}}
```

The simulation commands require compiled Pulse. On the testing machine, run the opt-in real-engine replay test:

```sh
PSIM_RUN_PULSE_TESTS=1 PYTHONPATH=src python -m pytest tests/test_pulse_acceptance.py -m slow -v
```

Offline tests use fake physiology and establish software behavior, not clinical validity. The engine remains pinned to the source API described in `pulse_api_notes.md`; no Pulse upgrade is included. Live PostgreSQL and real-Pulse verification are separate acceptance gates.
