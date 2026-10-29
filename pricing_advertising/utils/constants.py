# CONSTANTS

import numpy as np
from scipy.stats import lognorm

############### SEEDS #######################

SEEDS = [
    400287,
    26508205,
    45598533,
    48953810,
    39797986,
    50884562,
    54311534,
    21503116,
    33284243,
    23261443,
]

# Evaluation environments shared by every run, so that all evaluations are comparable
EVAL_SEEDS = [1_000_001, 1_000_002, 1_000_003, 1_000_004]

############### REQUEST PARAMETERS ################

# Maximum number of days a client can stay
MAX_DAYS_STAY = 7
# Stay duration = ceil(X) with X ~ LogNormal(mu, sigma), truncated at MAX_DAYS_STAY and renormalized
# (mu and sigma are the mean and std of log(X); the median of X is exp(mu))
STAY_LOGNORMAL_MU = 0.9
STAY_LOGNORMAL_SIGMA = 0.6
# P(ceil(X) = d) = F(d) - F(d - 1): probability of a stay of 1, 2, ..., MAX_DAYS_STAY nights
# (renormalized again when truncated further)
_stay_cdf = lognorm.cdf(
    np.arange(MAX_DAYS_STAY + 1), s=STAY_LOGNORMAL_SIGMA, scale=np.exp(STAY_LOGNORMAL_MU)
)
_stay_pmf = np.diff(_stay_cdf)
STAY_DURATION_PMF = _stay_pmf / _stay_pmf.sum()
# Number of min guests in a room = category 1
MIN_GUESTS = 1
# Number of max guests in a room = category 3
MAX_GUESTS = 3

# Factors to generate the number of requests
MIN_NUM_REQUESTS_FACTOR = 0.2
STD_FACTOR = 0.1

# Interval in the random function to generate the average number of requests
MIN_AVG_NUM_REQUESTS = 150
MAX_AVG_NUM_REQUESTS = 350

# Minimum and maximum advance days for booking
MIN_ADVANCE_DAYS = 1
MAX_ADVANCE_DAYS = 25 * 7  # 25 weeks

############### HOTEL PARAMETERS ################

# Number of room categories
NUM_CATEGORIES = 3

############### SEASONALITY #####################

# High season months
HIGH_SEASON = set([1, 2, 7, 8, 9, 12])
REQUESTS_HIGH_SEASON_MULTIPLIER = 1.2

################# BERSTEIN COEFFICIENTS ################

# Number of Berstein Coefficients
NUM_COEFFICIENTS = 4

################# ACCEPTANCE MODELS PARAMETERS ################

# Simple acceptance parameters
SIMPLE_ACCEPTANCE_THRESHOLD_CATEGORY1 = 100
SIMPLE_ACCEPTANCE_THRESHOLD_CATEGORY2 = 150
SIMPLE_ACCEPTANCE_THRESHOLD_CATEGORY3 = 200

# Logistic acceptance parameters
LOGISTIC_ACCEPTANCE_BETA = 1

# Base reference price p0: low-season, same-day booking
LOGISTIC_ACCEPTANCE_AVG_PRICES = {
    1: 360,
    2: 423,
    3: 450,
}

LOGISTIC_ACCEPTANCE_HIGH_SEASON_MULTIPLIERS = {
    1: 1.1,
    2: 1.1,
    3: 1.1,
}

# Advance factor: the reference price decreases linearly from A_MAX (same-day booking)
# to A_MIN (booking made BOOKING_WINDOW days ahead)
LOGISTIC_ACCEPTANCE_ADVANCE_MAX = 1.0
LOGISTIC_ACCEPTANCE_ADVANCE_MIN = 0.8

# Price scale of the logistic curve (the slope is beta / scale)
LOGISTIC_ACCEPTANCE_STDS = {
    1: 30,
    2: 45,
    3: 60,
}

################# ENVIRONMENT PARAMETERS ####################

# Observation space parameters
WINDOW_SIZE = 30
NUM_WINDOWS = 12
# W: maximum number of days of advance with which a room can be booked
BOOKING_WINDOW = NUM_WINDOWS * WINDOW_SIZE

################# TIMESTEPS for simulation ####################
TOTAL_TIMESTEPS_1K = 1_000
TOTAL_TIMESTEPS_30K = 30_000
TOTAL_TIMESTEPS_500K = 500_000
TOTAL_TIMESTEPS_1M = 1_000_000
TOTAL_TIMESTEPS_2M = 2_000_000
TOTAL_TIMESTEPS_5M = 5_000_000
TOTAL_TIMESTEPS_10M = 10_000_000
TOTAL_TIMESTEPS_20M = 20_000_000
TOTAL_TIMESTEPS_50M = 50_000_000
TOTAL_TIMESTEPS_100M = 100_000_000
