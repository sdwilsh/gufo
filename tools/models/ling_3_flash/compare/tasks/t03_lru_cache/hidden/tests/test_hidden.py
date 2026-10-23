import unittest

from cache import LRUCache


class Hidden(unittest.TestCase):
    def test_get_refreshes(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(c.get("a"), 1)
        c.put("c", 3)
        self.assertIn("a", c)
        self.assertNotIn("b", c)

    def test_put_existing_refreshes_and_updates(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 10)
        c.put("c", 3)
        self.assertEqual(c.get("a"), 10)
        self.assertNotIn("b", c)

    def test_missing_default(self):
        c = LRUCache(1)
        self.assertIsNone(c.get("x"))
        self.assertEqual(c.get("x", 7), 7)

    def test_capacity_one(self):
        c = LRUCache(1)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(len(c), 1)
        self.assertEqual(c.get("b"), 2)
