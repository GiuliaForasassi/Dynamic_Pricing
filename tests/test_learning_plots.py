import numpy as np
import pytest

from pricing_advertising.plots import learning_plots as plots


def test_action_statistics_count_clipped_samples():
    stats = plots.ActionStatistics()
    actions = np.zeros((4, 3, 2, 4))
    actions[:2, 0, 0, 0] = -0.5
    actions[2:, 0, 0, 0] = 1.5
    scalars = stats.add(10, actions, 0.0, 1.0)
    assert stats.timesteps == [10]
    assert stats.mean[0][0, 0, 0] == pytest.approx(0.5)
    assert stats.clipped_low[0][0, 0, 0] == 0.5
    assert stats.clipped_high[0][0, 0, 0] == 0.5
    # Every other coefficient is exactly 0, i.e. on the lower bound
    assert stats.clipped_low[0][1, 1, 3] == 1.0
    assert scalars["action_clipped_high_fraction"] == pytest.approx(0.5 / 24)
    with pytest.raises(ValueError, match="shape"):
        stats.add(11, np.zeros((3, 2, 4)), 0.0, 1.0)


def test_policy_curves_match_bernstein_probability():
    coefficients = np.zeros((3, 2, 4))
    coefficients[1, 1] = [0.4, 0.2, 0.2, 0.2]
    curves = plots.policy_probability_curves(coefficients)
    assert curves.shape == (3, 2, plots.OCCUPANCY_GRID.size)
    assert not curves[0].any()
    # beta = cumsum(theta); p(0) = 0.4 / 4, p(1) = sum(theta) / 4
    assert curves[1, 1, 0] == pytest.approx(0.1)
    assert curves[1, 1, -1] == pytest.approx(0.25)
    assert np.all(np.diff(curves[1, 1]) >= 0)

    accumulator = plots.PolicyCurveAccumulator()
    accumulator.add(coefficients)
    accumulator.add(np.zeros((3, 2, 4)))
    np.testing.assert_allclose(accumulator.mean(), curves / 2)
    with pytest.raises(ValueError, match="shape"):
        plots.policy_probability_curves(np.zeros((2, 4)))


def test_discounted_returns_and_value_check_stats():
    np.testing.assert_allclose(plots.discounted_returns([1.0, 2.0, 3.0], 0.5), [2.75, 3.5, 3.0])
    returns = np.array([1.0, 2.0, 3.0])
    perfect = plots.value_check_stats(returns, returns)
    assert perfect == {
        "explained_variance": 1.0,
        "correlation": pytest.approx(1.0),
        "mean_error": 0.0,
    }
    shifted = plots.value_check_stats(returns + 10, returns)
    assert shifted["explained_variance"] == pytest.approx(1.0)
    assert shifted["mean_error"] == pytest.approx(10.0)
    constant = plots.value_check_stats(np.ones(3), returns)
    assert np.isnan(constant["correlation"])
    assert constant["explained_variance"] == pytest.approx(0.0)
    assert np.isnan(plots.value_check_stats(returns, np.ones(3))["explained_variance"])
    with pytest.raises(ValueError, match="same length"):
        plots.value_check_stats(returns, returns[:2])


def test_eval_return_figure_draws_ppo_band_and_baselines():
    fig = plots.eval_return_figure(
        {1000: [90.0, 110.0], 2000: [150.0, 170.0]}, {"random": [80.0, 100.0], "constant": [120.0]}
    )
    panel = fig.axes[0]
    np.testing.assert_allclose(panel.lines[0].get_xdata(), [1000, 2000])
    np.testing.assert_allclose(panel.lines[0].get_ydata(), [100.0, 160.0])
    np.testing.assert_allclose([line.get_ydata()[0] for line in panel.lines[1:]], [90.0, 120.0])
    assert [text.get_text() for text in panel.get_legend().get_texts()][1:] == [
        "random: 90",
        "constant: 120",
    ]
