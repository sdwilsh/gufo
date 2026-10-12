import unittest

from config import parse_ini


class Hidden(unittest.TestCase):
    def test_comment_lines(self):
        self.assertEqual(parse_ini("; c\n  # d\n[a]\nx=1"), {"a": {"x": "1"}})

    def test_inline_comments(self):
        self.assertEqual(parse_ini("[a]\nx = 1 ; one\ny = 2   # two\nz = a#b"),
                         {"a": {"x": "1", "y": "2", "z": "a#b"}})

    def test_quotes(self):
        self.assertEqual(parse_ini('[a]\nx = "  hi ; there # you "  ; c\ny = ""'),
                         {"a": {"x": "  hi ; there # you ", "y": ""}})

    def test_global_section(self):
        self.assertEqual(parse_ini("k = v\n[s]\nq=1"), {"": {"k": "v"}, "s": {"q": "1"}})

    def test_malformed(self):
        with self.assertRaises(ValueError) as ctx:
            parse_ini("[a]\nx=1\nnonsense\n")
        self.assertIn("3", str(ctx.exception))

    def test_empty_value_and_equals_in_value(self):
        self.assertEqual(parse_ini("[a]\nx =\ny = a=b"), {"a": {"x": "", "y": "a=b"}})
