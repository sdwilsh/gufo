import unittest

from timeparse import parse_duration


class Hidden(unittest.TestCase):
    def test_whitespace_and_case(self):
        self.assertEqual(parse_duration(" 1H 30M "), 5400)

    def test_mixed_trailing_number(self):
        self.assertEqual(parse_duration("1m30"), 90)

    def test_errors(self):
        for bad in ("", "   ", "h", "5x", "1h-3m", "m5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_duration(bad)

    def test_seconds(self):
        self.assertEqual(parse_duration("45s"), 45)
