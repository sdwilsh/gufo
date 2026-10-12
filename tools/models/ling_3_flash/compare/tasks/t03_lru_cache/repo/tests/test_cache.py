import unittest

from cache import LRUCache


class CacheTest(unittest.TestCase):
    def test_evicts_oldest(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)
        self.assertNotIn("a", c)
        self.assertEqual(len(c), 2)

    def test_capacity_validation(self):
        with self.assertRaises(ValueError):
            LRUCache(0)
