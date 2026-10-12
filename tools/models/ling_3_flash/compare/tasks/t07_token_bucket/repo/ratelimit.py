import time


class TokenBucket:
    def __init__(self, rate, capacity, clock=time.monotonic):
        raise NotImplementedError

    def allow(self, n=1):
        raise NotImplementedError

    def retry_after(self, n=1):
        raise NotImplementedError
