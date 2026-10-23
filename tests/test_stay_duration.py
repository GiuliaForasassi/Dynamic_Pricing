import numpy as np

from pricing_advertising.utils.constants import (
    MAX_DAYS_STAY,
    STAY_DURATION_PMF,
    STAY_LOGNORMAL_MU,
    STAY_LOGNORMAL_SIGMA,
)


def test_stay_pmf_is_a_valid_distribution() -> None:
    assert len(STAY_DURATION_PMF) == MAX_DAYS_STAY
    assert (STAY_DURATION_PMF > 0).all()
    assert np.isclose(STAY_DURATION_PMF.sum(), 1.0)


def test_stay_pmf_is_unimodal() -> None:
    mode = int(np.argmax(STAY_DURATION_PMF))
    assert (np.diff(STAY_DURATION_PMF[: mode + 1]) > 0).all()
    assert (np.diff(STAY_DURATION_PMF[mode:]) < 0).all()


def test_stay_pmf_matches_ceil_of_lognormal_samples() -> None:
    rng = np.random.default_rng(0)
    samples = np.ceil(rng.lognormal(STAY_LOGNORMAL_MU, STAY_LOGNORMAL_SIGMA, size=400_000))
    samples = samples[samples <= MAX_DAYS_STAY].astype(int)
    empirical = np.bincount(samples, minlength=MAX_DAYS_STAY + 1)[1:] / len(samples)
    assert np.allclose(empirical, STAY_DURATION_PMF, atol=0.005)
