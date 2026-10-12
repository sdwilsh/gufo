import unittest

from textwrap2 import wrap


class WrapTest(unittest.TestCase):
    def test_exact_fit(self):
        self.assertEqual(wrap("ab cd ef", 5), ["ab cd", "ef"])

    def test_empty(self):
        self.assertEqual(wrap("", 10), [])
