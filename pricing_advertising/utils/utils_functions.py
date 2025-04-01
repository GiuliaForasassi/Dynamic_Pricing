from pricing_advertising.requests.request import Request


def extract_advance(request: Request) -> int:
    return request.advance
