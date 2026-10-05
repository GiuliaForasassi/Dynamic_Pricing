import numpy as np
import pytest

from pricing_advertising.pricing_model.bernstein_model import BernsteinProbModel


def test_coefficients_follow_pdf_order():
    occupancy = np.linspace(0, 1, 101)
    constant = BernsteinProbModel(np.array([1.0, 0.0, 0.0, 0.0]))
    cubic = BernsteinProbModel(np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(constant.beta, [1, 1, 1, 1])
    np.testing.assert_array_equal(cubic.beta, [0, 0, 0, 1])
    np.testing.assert_allclose(constant.predict_increment_probs(occupancy), 0.25)
    np.testing.assert_allclose(cubic.predict_increment_probs(occupancy), occupancy**3 / 4)


def test_probability_is_bounded_monotone_and_matches_scalar_basis():
    occupancy = np.linspace(0, 1, 101)
    rng = np.random.default_rng(42)
    for coefficients in rng.uniform(size=(30, 4)):
        model = BernsteinProbModel(coefficients)
        probabilities = model.predict_increment_probs(occupancy)
        assert model.m == 3
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.all(np.diff(probabilities) >= -1e-14)
        assert probabilities[0] == pytest.approx(coefficients[0] / 4)
        assert probabilities[-1] == pytest.approx(coefficients.sum() / 4)
        np.testing.assert_allclose(probabilities, model._bernstein_polynomial(occupancy) / 4)


def test_zero_and_one_coefficients():
    occupancy = np.linspace(0, 1, 101)
    np.testing.assert_array_equal(
        BernsteinProbModel(np.zeros(4)).predict_increment_probs(occupancy), 0
    )
    np.testing.assert_allclose(
        BernsteinProbModel(np.ones(4)).predict_increment_probs(occupancy),
        (1 + 3 * occupancy) / 4,
    )


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([]),
        np.ones((2, 4)),
        np.array(0.5),
        np.array([-0.1]),
        np.array([1.1]),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_invalid_coefficients_are_rejected(coefficients):
    with pytest.raises(ValueError):
        BernsteinProbModel(coefficients)


@pytest.mark.parametrize("occupancy", [[-0.1], [1.1], [np.nan], [np.inf]])
def test_invalid_occupancy_is_rejected(occupancy):
    with pytest.raises(ValueError, match="Occupancy"):
        BernsteinProbModel(np.ones(4)).predict_increment_probs(np.asarray(occupancy))
