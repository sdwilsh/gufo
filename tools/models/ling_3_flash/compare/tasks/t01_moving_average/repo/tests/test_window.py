import unittest

from stats.window import moving_average


class MovingAverageTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            moving_average([1, 2], 0)
