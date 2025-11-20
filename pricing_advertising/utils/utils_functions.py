from pricing_advertising.requests.request import Request


# ============ ADVANCE ============
def extract_advance(request: Request) -> int:
    return request.advance
