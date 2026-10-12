import inspect
import unittest

import formats
from formats import format_rows, to_csv, to_pipe, to_tsv


class Hidden(unittest.TestCase):
    def test_arbitrary_separator(self):
        self.assertEqual(format_rows([["a", "b;c"], [1, 2]], ";"), 'a;"b;c"\n1;2')
        self.assertEqual(format_rows([["a", "b,c"]], ";"), "a;b,c")

    def test_multichar_separator(self):
        self.assertEqual(format_rows([["a", "x::y"]], "::"), 'a::"x::y"')

    def test_newline_and_empty(self):
        self.assertEqual(format_rows([["a\nb"]], ","), '"a\nb"')
        self.assertEqual(format_rows([], ","), "")
        self.assertEqual(to_csv([[]]), "")

    def test_wrappers_are_thin(self):
        for fn in (to_csv, to_tsv, to_pipe):
            body = [l for l in inspect.getsource(fn).splitlines() if l.strip() and not l.strip().startswith(("#", '"""'))]
            self.assertLessEqual(len(body), 2, fn.__name__)

    def test_wrappers_delegate(self):
        original = formats.format_rows
        calls = []
        formats.format_rows = lambda rows, sep: calls.append(sep) or "x"
        try:
            self.assertEqual(formats.to_pipe([[1]]), "x")
            self.assertEqual(calls, ["|"])
        finally:
            formats.format_rows = original
