import random
from datetime import date

import numpy as np
from loguru import logger
from scipy.stats import norm

from pricing_advertising.acceptance.acceptance import ABCAcceptanceModel
from pricing_advertising.enums.enums import RoomCategory
from pricing_advertising.requests.request import Request


class LogisticAcceptanceModel(ABCAcceptanceModel):
    def __init__(self, beta: float, avg_prices: dict, variances: dict) -> None:
        # Beta parameter: check how sensitive the probability is to the difference between the proposed price and the reference price
        self.beta = beta
        # dictionaries with avg price and variance per category
        self.avg_prices = avg_prices
        self.variances = variances
        # can add season_factor to handle the seasonality of the prices

        # --- DEBUG counters ---
        # self._dbg_seen = 0
        # elf._dbg_acc = 0

    def _accept(self, proposed_price: float, request: Request) -> bool:
        category = request.category
        los = max(1, int(request.days_stay))
        proposed_daily = proposed_price / los

        variance = self.variances[category.value]
        avg_price = self.avg_prices[category.value]

        # generate the reference price for the client for the requested category
        reference_price = self.generate_reference_price(category=category)

        acceptance_probability = self.logistic_function(
            reference_price=reference_price,
            proposed_price=proposed_daily,
            beta=self.beta,
            variance=variance,
            avg_price=avg_price,
        )
        # penalty if you propose an alternative: the acceptance is reduced by 20%
        if request.proposed_t_reservation is not None or request.proposed_days_stay is not None:
            acceptance_probability *= 0.8

        # transform the probability in a boolean decision
        random_number = random.random()
        accepted = random_number <= acceptance_probability

        # --- DEBUG SECTION ---
        """
        self._dbg_seen += 1
        self._dbg_acc += int(accepted)

        # Ogni 500 richieste stampa il tasso di accettazione medio
        if self._dbg_seen % 500 == 0:
            rate = self._dbg_acc / self._dbg_seen
            print(f"[ACCEPT-STAT] seen={self._dbg_seen} acc_rate={rate:.2%}")

        # Stampa il dettaglio di alcune richieste (1% circa)
        if random.random() < 0.01:
            std = np.sqrt(variance)
            print(
                f"[ACCEPT-DBG] cat={category.value} avg={avg_price:.1f} ref={reference_price:.1f} "
                f"prop={proposed_price:.1f} std={std:.2f} beta={self.beta:.2f} "
                f"p={acceptance_probability:.3f} accepted={accepted}"
            )"""
        # --- END DEBUG SECTION ---

        return accepted

    def _accept_daily_price(self, daily_price: float, request: Request) -> bool:
        return self._accept(daily_price, request)

    def generate_reference_price(self, category: RoomCategory) -> float:
        avg_price = self.avg_prices[category.value]
        variance = self.variances[category.value]
        std = np.sqrt(variance)

        reference_price = norm.rvs(loc=avg_price, scale=std)
        return max(0, reference_price)

    def logistic_function(
        self,
        proposed_price: float,
        reference_price: float,
        beta: float,
        variance: float,
        avg_price: float,
    ) -> float:
        # Calculate the z-score respect to the average price: how many standard deviations away from the mean
        z_score = (proposed_price - avg_price) / np.sqrt(variance)

        # Calculate the correction factor based on the reference price
        correction_factor = (reference_price - avg_price) / np.sqrt(variance)

        # x measures how the proposed price is below or above the reference price
        x = beta * (z_score - correction_factor)
        # clippo in order to prevent overflow/underflow
        x = np.clip(x, -60, 60)
        # Calculate the acceptance probability
        acceptance_probability = 1 / (1 + np.exp(x))
        return acceptance_probability


if __name__ == "__main__":
    avg_prices = {
        RoomCategory.CATEGORY1: 100,
        RoomCategory.CATEGORY2: 150,
        RoomCategory.CATEGORY3: 200,
    }
    variances = {
        RoomCategory.CATEGORY1: 50,
        RoomCategory.CATEGORY2: 75,
        RoomCategory.CATEGORY3: 100,
    }
    beta = 0.1

    logistic_model = LogisticAcceptanceModel(beta=beta, avg_prices=avg_prices, variances=variances)
    request = Request(
        t_request=date.today(), t_reservation=date(2025, 3, 30), days_stay=3, num_guests=2
    )
    total_proposed_price = 110
    acceptance = logistic_model.accept(total_proposed_price, request)
    logger.info(f"Request: {request}")
    logger.info(f"Acceptance: {acceptance}")

    # Negative beta test
    try:
        LogisticAcceptanceModel(beta=-0.1, avg_prices=avg_prices, variances=variances)
        logger.warning("Negative beta test: Model initialized with negative beta!")
    except ValueError:
        logger.info("Correctly captured negative beta error.")

    # Zero beta test
    zero_beta_model = LogisticAcceptanceModel(beta=0, avg_prices=avg_prices, variances=variances)
    assert (
        zero_beta_model.logistic_function(
            proposed_price=110, reference_price=100, beta=0, variance=50, avg_price=100
        )
        == 0.5
    ), "Zero beta should return 0.5 probability"

    # Acceptance probability test
    acceptance_prob = logistic_model.logistic_function(110, 100, 0.1, 50, 100)
    assert 0 <= acceptance_prob <= 1, "Acceptance probability should be in [0,1]"

    # Reference price test
    reference_price = logistic_model.generate_reference_price(RoomCategory.CATEGORY1)
    assert reference_price >= 0, "Reference price should be non-negative"

    # Invalid room category test
    try:
        logistic_model.generate_reference_price(None)  # type: ignore
    except KeyError:
        logger.info("Correctly caught invalid room category error.")

    logger.info("All tests passed successfully!")
