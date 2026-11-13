import numpy as np

from engine.physiology.ecg_synth import ECGSynthesizer
from engine.physiology.pressure_synth import PressureSynthesizer
from test_adapter_replay import adapter  # shared fake Pulse fixture


def test_shared_beat_timing_and_evolving_pressures(adapter):
    adapter._ecg.noise_mv = adapter._ecg.wander_mv = 0
    adapter._pressure.art_noise_mmhg = adapter._pressure.pleth_noise = 0
    # Replace fake data with two distinct regimes, so averaging cannot pass.
    adapter._engine.pull_data = lambda: [0, 60, 100 if adapter._engine.step <= 100 else 180,
                                         60 if adapter._engine.step <= 100 else 120, .98, 0]
    adapter.advance(4)
    out = adapter.sample_waveforms(4)
    art, pleth, ecg = (np.asarray(out[c]) for c in ('ART', 'PLETH', 'ECG_II'))
    assert art[:200].max() < 110
    assert art[300:].max() > 170
    # At 60 bpm the R, arterial and pleth peaks occur at fixed offsets.
    r = np.argmax(ecg[:250]) / 250
    arterial = np.argmax(art[:125]) / 125
    peripheral = np.argmax(pleth[:125]) / 125
    assert .10 < arterial - r < .16
    assert .17 < peripheral - r < .23


def test_synthesizer_snapshots_restore_noise_and_phase():
    ecg = ECGSynthesizer(1)
    pressure = PressureSynthesizer(2)
    ecg.generate(.2, 75)
    pressure.generate_art(.2, 75, 120, 80)
    e, p = ecg.snapshot_state(), pressure.snapshot_state()
    expected_ecg = ecg.generate(.4, 80)
    expected_art = pressure.generate_art(.4, 80, 110, 70)
    ecg.restore_state(e)
    pressure.restore_state(p)
    np.testing.assert_array_equal(ecg.generate(.4, 80), expected_ecg)
    np.testing.assert_array_equal(pressure.generate_art(.4, 80, 110, 70), expected_art)


def test_sample_count_carry_across_short_drains(adapter):
    total = 0
    for _ in range(10):
        adapter.advance(.02)
        total += len(adapter.sample_waveforms(.02)['ART'])
    assert total == 25


def test_variable_hr_interpolation_is_independent_of_drain_boundaries(adapter):
    import copy
    whole = copy.deepcopy(adapter)
    for obj in (adapter, whole):
        obj._ecg.noise_mv = obj._ecg.wander_mv = 0
        obj._pressure.art_noise_mmhg = obj._pressure.pleth_noise = 0
    pieces = {c: [] for c in adapter.waveform_channels}
    for _ in range(20):
        adapter.advance(.1)
        for channel, values in adapter.sample_waveforms(.1).items():
            pieces[channel].extend(values)
    whole.advance(2)
    expected = whole.sample_waveforms(2)
    for channel in pieces:
        np.testing.assert_allclose(pieces[channel], expected[channel], atol=1e-10)


def test_varying_hr_and_pressure_reach_each_synthesizer(adapter, monkeypatch):
    received = {}
    ecg = adapter._ecg.generate
    art = adapter._pressure.generate_art
    pleth = adapter._pressure.generate_pleth

    def record_ecg(duration, hr, **kwargs):
        received['ecg_hr'] = hr.copy()
        received['ecg_phase'] = kwargs['phase_cycles'].copy()
        return ecg(duration, hr, **kwargs)

    def record_art(duration, hr, sbp, dbp, **kwargs):
        received.update(art_hr=hr.copy(), sbp=sbp.copy(), dbp=dbp.copy(),
                        art_phase=kwargs['phase_cycles'].copy())
        return art(duration, hr, sbp, dbp, **kwargs)

    def record_pleth(duration, hr, spo2, **kwargs):
        received['pleth_phase'] = kwargs['phase_cycles'].copy()
        return pleth(duration, hr, spo2, **kwargs)

    monkeypatch.setattr(adapter._ecg, 'generate', record_ecg)
    monkeypatch.setattr(adapter._pressure, 'generate_art', record_art)
    monkeypatch.setattr(adapter._pressure, 'generate_pleth', record_pleth)
    adapter.advance(1)
    adapter.sample_waveforms(1)
    for name in ('ecg_hr', 'art_hr', 'sbp', 'dbp'):
        assert np.ptp(received[name]) > 1
    np.testing.assert_allclose(received['art_phase'], received['pleth_phase'])
    np.testing.assert_allclose(received['art_phase'], received['ecg_phase'][1::2])


def test_nonaligned_drains_preserve_global_sample_grid(adapter):
    import copy
    whole = copy.deepcopy(adapter)
    for obj in (adapter, whole):
        obj._ecg.noise_mv = obj._ecg.wander_mv = 0
        obj._pressure.art_noise_mmhg = obj._pressure.pleth_noise = 0
    pieces = {c: [] for c in adapter.waveform_channels}
    lengths = []
    for duration in [.01, .01, .015, .025] * 10:
        adapter.advance(duration)
        chunk = adapter.sample_waveforms(duration)
        lengths.append(len(chunk['CO2']))
        for channel, values in chunk.items():
            pieces[channel].extend(values)
    whole.advance(.6)
    expected = whole.sample_waveforms(.6)
    assert lengths[:4] == [0, 1, 0, 2]
    assert {c: len(v) for c, v in pieces.items()} == {
        'CO2': 30, 'ART': 75, 'PLETH': 75, 'ECG_II': 150}
    for channel in pieces:
        np.testing.assert_allclose(pieces[channel], expected[channel], atol=1e-10)


def test_nonaligned_drain_start_times_follow_sample_grid(adapter):
    emitted = {channel: 0 for channel in adapter.waveform_channels}
    rates = {'ART': 125, 'PLETH': 125, 'ECG_II': 250, 'CO2': 50}
    for duration in [.01, .01, .015, .025, .007, .013, .02]:
        adapter.advance(duration)
        waves = adapter.sample_waveforms(duration)
        assert set(adapter.last_waveform_start_s) == {c for c, data in waves.items() if data}
        for channel, data in waves.items():
            if data:
                assert np.isclose(adapter.last_waveform_start_s[channel],
                                  (emitted[channel] + 1) / rates[channel])
                emitted[channel] += len(data)
