import unittest

import counter
from app import Server
from counter import Counter


class Hidden(unittest.TestCase):
    def setUp(self):
        counter.reset()

    def test_counter_class(self):
        c = Counter()
        self.assertEqual(c.hit("a"), 1)
        self.assertEqual(c.hit("a"), 2)
        self.assertEqual(c.count("a"), 2)
        self.assertEqual(c.total(), 2)
        c.reset()
        self.assertEqual(c.total(), 0)

    def test_instances_independent(self):
        a, b = Counter(), Counter()
        a.hit("x")
        self.assertEqual(b.count("x"), 0)

    def test_servers_independent_by_default(self):
        s1, s2 = Server(), Server()
        s1.handle("/a")
        self.assertEqual(s1.stats(), 1)
        self.assertEqual(s2.stats(), 0)
        self.assertEqual(counter.total(), 0)

    def test_injected_counter(self):
        c = Counter()
        Server(counter=c).handle("/z")
        self.assertEqual(c.count("/z"), 1)

    def test_default_counter_delegation(self):
        counter.hit("q")
        self.assertEqual(counter.default_counter.count("q"), 1)
        counter.default_counter.hit("q")
        self.assertEqual(counter.count("q"), 2)

    def test_no_module_global_dict(self):
        self.assertFalse(hasattr(counter, "_counts"))
