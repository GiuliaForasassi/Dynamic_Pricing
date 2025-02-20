from datetime import date, timedelta

import numpy as np


def encode_cyclic_time(day: int) -> tuple:
    sin_day = np.sin(2 * np.pi * day / 365)
    cos_day = np.cos(2 * np.pi * day / 365)
    return sin_day, cos_day
