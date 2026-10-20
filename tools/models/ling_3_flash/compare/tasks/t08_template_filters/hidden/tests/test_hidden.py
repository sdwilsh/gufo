import unittest

from tmpl import render


class Hidden(unittest.TestCase):
    def test_filters(self):
        ctx = {"n": "hello world"}
        self.assertEqual(render("{{ n|upper }}", ctx), "HELLO WORLD")
        self.assertEqual(render("{{n|title}}", ctx), "Hello World")
        self.assertEqual(render("{{ n | lower }}", {"n": "ABC"}), "abc")

    def test_default(self):
        self.assertEqual(render("{{ who|default:guest }}", {}), "guest")
        self.assertEqual(render("{{ who|default:guest }}", {"who": None}), "guest")
        self.assertEqual(render("{{ who|default:guest }}", {"who": "ann"}), "ann")

    def test_chain(self):
        self.assertEqual(render("{{ who|default:guest user|upper }}", {}), "GUEST USER")
        self.assertEqual(render("{{who|upper|default:x}}", {"who": "a"}), "A")

    def test_none_and_non_string(self):
        self.assertEqual(render("[{{ a }}]", {"a": None}), "[]")
        self.assertEqual(render("{{ n|upper }}", {"n": 12}), "12")

    def test_unknown_filter(self):
        with self.assertRaises(ValueError):
            render("{{ a|reverse }}", {"a": "x"})

    def test_missing_without_default(self):
        with self.assertRaises(KeyError):
            render("{{ a|upper }}", {})

    def test_text_untouched(self):
        self.assertEqual(render("a { b } {{ c }} }}", {"c": 1}), "a { b } 1 }}")
