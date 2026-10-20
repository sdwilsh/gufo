import unittest

from inventory import Inventory, InsufficientStock


class Hidden(unittest.TestCase):
    def setUp(self):
        self.inv = Inventory()
        self.inv.add("bolt", 5)
        self.inv.add("nut", 2)
        self.inv.add("cog", 2)

    def test_remove(self):
        self.inv.remove("bolt", 3)
        self.assertEqual(self.inv.count("bolt"), 2)

    def test_remove_to_zero_keeps_sku(self):
        self.inv.remove("nut", 2)
        self.assertEqual(self.inv.count("nut"), 0)
        self.assertIn(("nut", 0), self.inv.low_stock(0))

    def test_remove_errors(self):
        with self.assertRaises(ValueError):
            self.inv.remove("bolt", 0)
        with self.assertRaises(KeyError):
            self.inv.remove("washer", 1)
        with self.assertRaises(InsufficientStock):
            self.inv.remove("bolt", 6)
        self.assertEqual(self.inv.count("bolt"), 5)

    def test_insufficient_is_not_valueerror(self):
        self.assertFalse(issubclass(InsufficientStock, (ValueError, KeyError)))

    def test_low_stock_sorted(self):
        self.assertEqual(self.inv.low_stock(2), [("cog", 2), ("nut", 2)])
        self.assertEqual(self.inv.low_stock(10), [("cog", 2), ("nut", 2), ("bolt", 5)])
        self.assertEqual(self.inv.low_stock(1), [])
