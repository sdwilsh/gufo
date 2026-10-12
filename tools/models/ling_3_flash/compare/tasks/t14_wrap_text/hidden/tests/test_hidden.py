import unittest

from textwrap2 import wrap


class Hidden(unittest.TestCase):
    def test_long_word_split(self):
        self.assertEqual(wrap("abcdefghij", 4), ["abcd", "efgh", "ij"])

    def test_long_word_after_text(self):
        self.assertEqual(wrap("hi abcdefg", 4), ["hi", "abcd", "efg"])

    def test_whitespace_collapse(self):
        self.assertEqual(wrap("a   b\t\tc", 10), ["a b c"])

    def test_paragraphs(self):
        self.assertEqual(wrap("one two\n\n\nthree", 7), ["one two", "", "three"])

    def test_single_newline_is_space(self):
        self.assertEqual(wrap("one\ntwo", 10), ["one two"])

    def test_whitespace_only(self):
        self.assertEqual(wrap("  \n \n\n ", 5), [])

    def test_bad_width(self):
        with self.assertRaises(ValueError):
            wrap("x", 0)

    def test_line_limit(self):
        lines = wrap("the quick brown fox jumps over the lazy dog", 11)
        self.assertEqual(lines, ["the quick", "brown fox", "jumps over", "the lazy", "dog"])
