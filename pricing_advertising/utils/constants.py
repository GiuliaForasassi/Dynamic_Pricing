# CONSTANTS

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

# Maximum number of days a client can stay
MAX_DAYS_STAY = 10
# Number of min guests in a room = category 1
MIN_GUESTS = 1
# Number of max guests in a room = category 3
MAX_GUESTS = 3
# Number of Berstein Coefficients
NUM_COEFFICIENTS = 4
# Number of room categories
NUM_CATEGORIES = 3

# Factors
VARIABILITY_MIN_MAX_PRICE = 0.2
MIN_NUM_REQUESTS_FACTOR = 0.2
STD_FACTOR = 0.1


# Interval in the random function to generate the average number of requests
MIN_AVG_NUM_REQUESTS = 4
MAX_AVG_NUM_REQUESTS = 15

# Interval in the random function to generate the price
MIN_PRICE = 50
MAX_PRICE = 200

# SIMPLE ACCEPTANCE PARAMETERS
SIMPLE_ACCEPTANCE_THRESHOLD_CATEGORY1 = 100
SIMPLE_ACCEPTANCE_THRESHOLD_CATEGORY2 = 150
SIMPLE_ACCEPTANCE_THRESHOLD_CATEGORY3 = 200

# LOGISTIC ACCEPTANCE PARAMETERS
LOGISTIC_ACCEPTANCE_BETA = 1

LOGISTIC_ACCEPTANCE_AVG_PRICES = {
    1: 100,
    2: 150,
    3: 200,
}

# 1/4 of the avg price
LOGISTIC_ACCEPTANCE_VARIANCES = {
    1: 25,
    2: 37.5,
    3: 50,
}

N_ENVS = 2
