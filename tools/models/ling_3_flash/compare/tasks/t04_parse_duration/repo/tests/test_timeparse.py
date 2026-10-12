import unittest

from timeparse import parse_duration


class ParseTest(unittest.TestCase):
    def test_units(self):
        self.assertEqual(parse_duration("1h30m"), 5400)
        self.assertEqual(parse_duration("2d"), 172800)

    def test_bare_number_is_seconds(self):
        self.assertEqual(parse_duration("90"), 90)
