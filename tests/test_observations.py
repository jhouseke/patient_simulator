from datetime import datetime, timedelta

import pytest

from engine.observation import ConfigurableObservationModel, MetricObservationConfig


T = datetime(2025, 1, 1)


def test_ideal_and_missing_outputs():
    model = ConfigurableObservationModel()
    truth = {"hr": 70.0, "map": 85.0}
    assert model.observe(T, truth) == truth
    assert model.observe(T + timedelta(seconds=1), {"hr": 71.0}) == {"hr": 71.0}
    assert truth == {"hr": 70.0, "map": 85.0}


def test_lag_and_average_never_use_future_samples():
    model = ConfigurableObservationModel(metrics={"hr": MetricObservationConfig(lag_s=2, averaging_s=2)})
    assert model.observe(T, {"hr": 10}) == {}
    assert model.observe(T + timedelta(seconds=1), {"hr": 20}) == {}
    assert model.observe(T + timedelta(seconds=2), {"hr": 1000}) == {"hr": 10}
    assert model.observe(T + timedelta(seconds=3), {"hr": 2000}) == {"hr": 15}


def test_seed_snapshot_and_restore():
    cfg = {"hr": MetricObservationConfig(noise_std=2, lag_s=1, dropout_probability=.25)}
    a, b = ConfigurableObservationModel(42, cfg), ConfigurableObservationModel(42, cfg)
    for i in range(5):
        assert a.observe(T + timedelta(seconds=i), {"hr": 60 + i}) == b.observe(T + timedelta(seconds=i), {"hr": 60 + i})
    state = a.snapshot_state()
    expected = [a.observe(T + timedelta(seconds=i), {"hr": 60 + i}) for i in range(5, 30)]
    a.restore_state(state)
    assert [a.observe(T + timedelta(seconds=i), {"hr": 60 + i}) for i in range(5, 30)] == expected


def test_interval_dropout_and_missing():
    model = ConfigurableObservationModel(metrics={"hr": MetricObservationConfig(interval_s=2),
                                                   "map": MetricObservationConfig(dropout_probability=1)})
    assert model.observe(T, {"hr": 70, "map": 80}) == {"hr": 70}
    assert model.observe(T + timedelta(seconds=1), {"hr": 71}) == {}
    assert model.observe(T + timedelta(seconds=2), {"hr": 72}) == {"hr": 72}
    assert model.observe(T + timedelta(seconds=3), {}) == {}
    with pytest.raises(ValueError):
        model.observe(T, {})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_truth_rejected_without_changing_state(value):
    model = ConfigurableObservationModel(seed=1, metrics={"hr": MetricObservationConfig(noise_std=1)})
    model.observe(T, {"hr": 70.0})
    state = model.snapshot_state()
    with pytest.raises(ValueError, match="map.*finite"):
        model.observe(T + timedelta(seconds=1), {"hr": 71.0, "map": value})
    assert model.snapshot_state() == state


@pytest.mark.parametrize("field,value", [("noise_std", -1), ("lag_s", float('inf')),
                                        ("averaging_s", float('nan')), ("interval_s", -1),
                                        ("dropout_probability", 1.1)])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        MetricObservationConfig(**{field: value})
