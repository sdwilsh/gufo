import unittest

import counter
from app import Server


class AppTest(unittest.TestCase):
    def setUp(self):
        counter.reset()

    def test_handle_counts(self):
        s = Server()
        s.handle("/a")
        s.handle("/a")
        self.assertEqual(s.stats(), 2)

    def test_module_functions(self):
        counter.hit("x")
        self.assertEqual(counter.count("x"), 1)
        self.assertEqual(counter.total(), 1)
