import unittest

from stats.window import moving_average


class Hidden(unittest.TestCase):
    def test_window_equals_length(self):
        self.assertEqual(moving_average([2, 4, 6], 3), [4.0])

    def test_window_larger_than_length(self):
        self.assertEqual(moving_average([1, 2], 5), [])

    def test_window_one(self):
        self.assertEqual(moving_average([5, 6, 7], 1), [5.0, 6.0, 7.0])

    def test_empty(self):
        self.assertEqual(moving_average([], 2), [])

    def test_negative(self):
        with self.assertRaises(ValueError):
            moving_average([1], -1)
