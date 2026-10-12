import unittest

from shop.cart import Cart
from shop.checkout import receipt
from shop.cli import main


class ShopTest(unittest.TestCase):
    def test_cart(self):
        c = Cart()
        c.add("pen", 1.25, 4)
        c.add("pad", 3.5)
        self.assertEqual(c.total(), 8.5)

    def test_receipt(self):
        self.assertEqual(receipt([("a", 10.0, 1)], 0.1), {"subtotal": 10.0, "tax": 1.0, "total": 11.0})

    def test_cli(self):
        self.assertEqual(main([("a", 10.0, 2)]), "1 items, total 20.00 (with tax 22.00)")
