from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np

from pricing_advertising.requests.request import Request
from pricing_advertising.utils.constants import HIGH_SEASON


# ============ ADVANCE ============
def extract_advance(request: Request) -> int:
    return request.advance
