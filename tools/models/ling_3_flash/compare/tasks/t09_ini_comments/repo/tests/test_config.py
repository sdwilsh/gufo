import unittest

from config import parse_ini


class ParseTest(unittest.TestCase):
    def test_sections(self):
        self.assertEqual(parse_ini("[a]\nx = 1\n\n[b]\ny=2"), {"a": {"x": "1"}, "b": {"y": "2"}})
