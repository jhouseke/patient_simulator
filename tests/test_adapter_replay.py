import json
import sys
import types

import pytest

from engine.physiology.pulse_adapter import PatientConfig, PulseAdapter


class FakePulse:
    def __init__(self):
        self.step = 0

    def advance_time(self):
        self.step += 1
        return True

    def pull_data(self):
        return [self.step * .02, 60 + self.step, 120 + self.step,
                80 + self.step / 2, .98, self.step * 2]

    def serialize_to_string(self, fmt):
        return str(self.step).encode()

    def serialize_from_string(self, blob, drm, fmt):
        self.step = int(blob)
        return True


@pytest.fixture
def adapter(monkeypatch):
    pulse = types.ModuleType('pulse')
    cdm = types.ModuleType('pulse.cdm')
    engine = types.ModuleType('pulse.cdm.engine')
    engine.eSerializationFormat = types.SimpleNamespace(BINARY=1)
    for name, module in [('pulse', pulse), ('pulse.cdm', cdm), ('pulse.cdm.engine', engine)]:
        monkeypatch.setitem(sys.modules, name, module)
    obj = PulseAdapter(3, PatientConfig('MALE', 30, 170, 70),
                       waveform_channels=('ART', 'PLETH', 'ECG_II', 'CO2'))
    obj._engine = FakePulse()
    obj._index = {'hr': 1, 'sbp': 2, 'dbp': 3, 'spo2': 4, '__wave_CO2': 5}
    return obj


def test_replay_includes_pending_capture_rng_and_fractional_time(adapter):
    adapter.advance(.11)
    adapter.sample_waveforms(.11)
    adapter.advance(.07)  # pending samples and a fractional timestep
    snapshot = adapter.save_state()
    def run():
        adapter.advance(.12)
        first = adapter.sample_waveforms(.19)
        adapter.advance(.1)
        return first, adapter.sample_waveforms(.1), adapter._engine.step
    expected = run()
    adapter.load_state(snapshot)
    assert run() == expected


def test_invalid_snapshot_rejected_without_pickle(adapter):
    with pytest.raises(ValueError):
        adapter.load_state(b'not json')
    state = json.loads(adapter.save_state())
    state['version'] = 99
    with pytest.raises(ValueError):
        adapter.load_state(json.dumps(state).encode())


def test_missing_or_incomplete_capture_is_rejected(adapter):
    with pytest.raises(ValueError, match='advanced'):
        adapter.sample_waveforms(1)
    adapter._index.pop('sbp')
    with pytest.raises(RuntimeError, match='sbp'):
        adapter.advance(.02)
    assert not adapter._capture.hr
    adapter._index['sbp'] = 2
    adapter.advance(.02)
    adapter._capture.native['CO2'].clear()
    with pytest.raises(ValueError, match='CO2'):
        adapter.sample_waveforms(.02)
    assert len(adapter._capture.hr) == 1


def test_engine_spare_time_replayed_without_waveform_channels(adapter):
    class SparePulse(FakePulse):
        def __init__(self):
            super().__init__()
            self._spare_time_s = 0.0

        def advance_time_s(self, seconds):
            total = seconds + self._spare_time_s
            steps = int((total + 1e-12) / .02)
            self.step += steps
            self._spare_time_s = total - steps * .02
            return True

    adapter.waveform_channels = ()
    adapter._engine = SparePulse()
    adapter.advance(.01)
    snapshot = adapter.save_state()
    adapter.advance(.03)
    expected = (adapter._engine.step, adapter._engine._spare_time_s)
    adapter.load_state(snapshot)
    assert adapter._engine._spare_time_s == pytest.approx(.01)
    adapter.advance(.03)
    assert (adapter._engine.step, adapter._engine._spare_time_s) == expected


def test_nonaligned_drain_replay_and_length_validation(adapter):
    adapter.advance(.01)
    assert all(not values for values in adapter.sample_waveforms(.01).values())
    snapshot = adapter.save_state()
    def run():
        adapter.advance(.01)
        first = adapter.sample_waveforms(.01)
        adapter.advance(.015)
        second = adapter.sample_waveforms(.015)
        adapter.advance(.025)
        return first, second, adapter.sample_waveforms(.025)
    expected = run()
    adapter.load_state(snapshot)
    assert run() == expected
    adapter.advance(.02)
    adapter._capture.hr.clear()
    with pytest.raises(ValueError, match='captured'):
        adapter.sample_waveforms(.02)


def test_waveform_start_times_are_saved_and_restored(adapter):
    adapter.advance(.01)
    adapter.sample_waveforms(.01)
    assert adapter.last_waveform_start_s == {}
    adapter.advance(.01)
    adapter.sample_waveforms(.01)
    assert adapter.last_waveform_start_s == pytest.approx(
        {'CO2': .02, 'ART': .008, 'PLETH': .008, 'ECG_II': .004})
    snapshot = adapter.save_state()
    adapter.advance(.01)
    adapter.sample_waveforms(.01)
    assert adapter.last_waveform_start_s == {}
    adapter.load_state(snapshot)
    assert adapter.last_waveform_start_s == pytest.approx(
        {'CO2': .02, 'ART': .008, 'PLETH': .008, 'ECG_II': .004})
    adapter.advance(.01)
    adapter.sample_waveforms(.01)
    assert adapter.last_waveform_start_s == {}
