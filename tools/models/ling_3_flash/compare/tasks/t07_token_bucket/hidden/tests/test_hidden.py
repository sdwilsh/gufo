import unittest

from ratelimit import TokenBucket


class Clock:
    def __init__(self):
        self.now = 50.0

    def __call__(self):
        return self.now


class Hidden(unittest.TestCase):
    def test_refill(self):
        c = Clock()
        b = TokenBucket(2, 4, clock=c)
        self.assertTrue(b.allow(4))
        self.assertFalse(b.allow())
        c.now += 0.5
        self.assertTrue(b.allow())
        self.assertFalse(b.allow())

    def test_cap(self):
        c = Clock()
        b = TokenBucket(10, 3, clock=c)
        c.now += 100
        self.assertTrue(b.allow(3))
        self.assertFalse(b.allow())

    def test_failed_allow_consumes_nothing(self):
        c = Clock()
        b = TokenBucket(1, 5, clock=c)
        self.assertTrue(b.allow(3))
        self.assertFalse(b.allow(3))
        self.assertTrue(b.allow(2))

    def test_over_capacity(self):
        b = TokenBucket(1, 2, clock=Clock())
        self.assertFalse(b.allow(3))
        self.assertEqual(b.retry_after(3), float("inf"))

    def test_invalid_n(self):
        b = TokenBucket(1, 2, clock=Clock())
        with self.assertRaises(ValueError):
            b.allow(0)
        with self.assertRaises(ValueError):
            b.retry_after(-1)

    def test_retry_after(self):
        c = Clock()
        b = TokenBucket(2, 4, clock=c)
        self.assertEqual(b.retry_after(), 0.0)
        b.allow(4)
        self.assertAlmostEqual(b.retry_after(), 0.5)
        self.assertAlmostEqual(b.retry_after(3), 1.5)
        c.now += 0.25
        self.assertAlmostEqual(b.retry_after(), 0.25)
