import unittest

from tmpl import render


class RenderTest(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(render("Hi {{name}} and {{ other }}!", {"name": "A", "other": 3}), "Hi A and 3!")

    def test_missing_key(self):
        with self.assertRaises(KeyError):
            render("{{ x }}", {})
