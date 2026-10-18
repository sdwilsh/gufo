import unittest

from deps import resolve


class ResolveTest(unittest.TestCase):
    def test_chain(self):
        self.assertEqual(resolve({"app": ["lib"], "lib": ["core"], "core": []}), ["core", "lib", "app"])

    def test_diamond(self):
        graph = {"app": ["a", "b"], "a": ["core"], "b": ["core"], "core": []}
        self.assertEqual(resolve(graph), ["core", "a", "b", "app"])
