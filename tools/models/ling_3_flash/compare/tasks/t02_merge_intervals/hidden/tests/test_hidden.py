import unittest

from intervals import merge


class Hidden(unittest.TestCase):
    def test_input_not_modified(self):
        data = [(5, 6), (1, 2)]
        merge(data)
        self.assertEqual(data, [(5, 6), (1, 2)])

    def test_unsorted_and_nested(self):
        self.assertEqual(merge([(6, 7), (1, 10), (2, 3)]), [(1, 10)])

    def test_lists_become_tuples(self):
        self.assertEqual(merge([[3, 4], [1, 2]]), [(1, 2), (3, 4)])

    def test_empty(self):
        self.assertEqual(merge([]), [])

    def test_gap_of_one_is_kept(self):
        self.assertEqual(merge([(1, 2), (3, 4)]), [(1, 2), (3, 4)])
