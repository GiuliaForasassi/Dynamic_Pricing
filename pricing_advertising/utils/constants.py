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
MIN_AVG_NUM_REQUESTS = 4
MAX_AVG_NUM_REQUESTS = 15


############### HOTEL PARAMETERS ################

# Number of room categories
NUM_CATEGORIES = 3

############### SEASONALITY #####################

# High season months
HIGH_SEASON = set([1, 2, 6, 7, 8, 12])

################# PRICE PARAMETERS ################

# Interval in the random function to generate the price
MIN_PRICE = 50
MAX_PRICE = 200

VARIABILITY_MIN_MAX_PRICE = 0.2

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

LOGISTIC_ACCEPTANCE_AVG_PRICES = {
    1: 100,
    2: 150,
    3: 200,
}

# 1/4 of the avg price
LOGISTIC_ACCEPTANCE_VARIANCES = {
    1: 900,
    2: 2025,
    3: 3600,
}

################# ENVIRONMENT PARAMETERS ####################

# Number of environments for parallel processing
N_ENVS = 2

# Observation space parameters
WINDOW_SIZE = 30
NUM_WINDOWS = 12
