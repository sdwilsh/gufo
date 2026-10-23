import time


class TokenBucket:
    def __init__(self, rate, capacity, clock=time.monotonic):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self):
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def allow(self, n=1):
        if n <= 0:
            raise ValueError("n must be positive")
        self._refill()
        if n > self.capacity or self._tokens < n:
            return False
        self._tokens -= n
        return True

    def retry_after(self, n=1):
        if n <= 0:
            raise ValueError("n must be positive")
        if n > self.capacity:
            return float("inf")
        self._refill()
        if self._tokens >= n:
            return 0.0
        return (n - self._tokens) / self.rate
