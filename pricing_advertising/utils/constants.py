# CONSTANTS

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

############### REQUEST PARAMETERS ################

# Maximum number of days a client can stay
MAX_DAYS_STAY = 10
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
TOTAL_TIMESTEPS_500K = 500_000
TOTAL_TIMESTEPS_1M = 1_000_000
TOTAL_TIMESTEPS_2M = 2_000_000
TOTAL_TIMESTEPS_5M = 5_000_000
TOTAL_TIMESTEPS_10M = 10_000_000
TOTAL_TIMESTEPS_20M = 20_000_000
TOTAL_TIMESTEPS_50M = 50_000_000
TOTAL_TIMESTEPS_100M = 100_000_000
