import unittest

from intervals import merge


class MergeTest(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(merge([(1, 4), (2, 6), (8, 9)]), [(1, 6), (8, 9)])

    def test_touching(self):
        self.assertEqual(merge([(1, 3), (3, 5)]), [(1, 5)])
