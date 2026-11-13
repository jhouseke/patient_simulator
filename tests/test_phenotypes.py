from statistics import correlation

import pytest

from engine.physiology.phenotypes import sample_patient


def test_reproducible_and_provenance():
    result = sample_patient("adult", 123)
    assert result == sample_patient("adult", 123)
    assert result != sample_patient("adult", 124)
    assert result.review_status == "clinician-review-pending"
    assert result.phenotype == "adult" and result.seed == 123
    assert set(result.parameters) == {"size_factor", "cardiovascular_factor", "bmi"}


def test_bounds_and_correlations():
    for name in ("adult", "older_adult", "larger_adult"):
        patients = [sample_patient(name, seed).patient for seed in range(500)]
        assert all(18 <= p.age_yr <= 85 and 150 <= p.height_cm <= 195 and
                   45 <= p.weight_kg <= 140 and 50 <= p.hr_baseline <= 100 and
                   65 <= p.map_baseline <= 110 for p in patients)
        assert correlation([p.height_cm for p in patients], [p.weight_kg for p in patients]) > .5
        assert correlation([p.hr_baseline for p in patients], [p.map_baseline for p in patients]) > .7


def test_unknown_phenotype():
    with pytest.raises(ValueError, match="unknown phenotype"):
        sample_patient("unknown", 1)
