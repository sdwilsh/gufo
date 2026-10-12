import unittest

from inventory import Inventory


class InventoryTest(unittest.TestCase):
    def test_add_and_count(self):
        inv = Inventory()
        inv.add("bolt", 5)
        inv.add("bolt", 2)
        self.assertEqual(inv.count("bolt"), 7)
        self.assertEqual(inv.count("nut"), 0)
