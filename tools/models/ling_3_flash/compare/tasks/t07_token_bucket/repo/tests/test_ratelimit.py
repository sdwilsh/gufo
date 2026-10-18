import unittest

from ratelimit import TokenBucket


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class BucketTest(unittest.TestCase):
    def test_starts_full_then_empties(self):
        clock = FakeClock()
        b = TokenBucket(rate=1, capacity=2, clock=clock)
        self.assertTrue(b.allow())
        self.assertTrue(b.allow())
        self.assertFalse(b.allow())
