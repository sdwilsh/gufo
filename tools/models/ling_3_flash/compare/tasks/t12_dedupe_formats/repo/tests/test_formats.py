import unittest

from formats import to_csv, to_pipe, to_tsv


class FormatsTest(unittest.TestCase):
    def test_csv(self):
        self.assertEqual(to_csv([["a", "b,c", 3], ['x"y', "z", None]]), 'a,"b,c",3\n"x""y",z,None')

    def test_tsv(self):
        self.assertEqual(to_tsv([["a", "b\tc"]]), 'a\t"b\tc"')

    def test_pipe(self):
        self.assertEqual(to_pipe([[1, 2], [3, "a|b"]]), '1|2\n3|"a|b"')
