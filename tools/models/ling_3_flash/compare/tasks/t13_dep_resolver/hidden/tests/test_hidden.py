import unittest

from deps import CycleError, resolve


class Hidden(unittest.TestCase):
    def test_alphabetical_ties(self):
        self.assertEqual(resolve({"c": [], "b": [], "a": []}), ["a", "b", "c"])

    def test_ready_smallest_first(self):
        graph = {"a": ["z"], "b": [], "z": []}
        self.assertEqual(resolve(graph), ["b", "z", "a"])

    def test_unlisted_dependencies(self):
        self.assertEqual(resolve({"app": ["libx"]}), ["libx", "app"])

    def test_cycle(self):
        with self.assertRaises(CycleError) as ctx:
            resolve({"a": ["b"], "b": ["c"], "c": ["a"], "d": []})
        for name in "abc":
            self.assertIn(name, str(ctx.exception))
        self.assertTrue(issubclass(CycleError, ValueError))

    def test_self_cycle(self):
        with self.assertRaises(CycleError):
            resolve({"a": ["a"]})

    def test_empty(self):
        self.assertEqual(resolve({}), [])
