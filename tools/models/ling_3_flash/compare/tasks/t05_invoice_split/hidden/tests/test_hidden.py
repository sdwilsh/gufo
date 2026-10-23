import unittest

from billing.invoice import Invoice
from billing.split import split_cents


class Hidden(unittest.TestCase):
    def test_larger_first(self):
        self.assertEqual(split_cents(10, 3), [4, 3, 3])

    def test_exact(self):
        self.assertEqual(split_cents(9, 3), [3, 3, 3])

    def test_fewer_cents_than_parts(self):
        self.assertEqual(split_cents(2, 5), [1, 1, 0, 0, 0])

    def test_zero_total(self):
        self.assertEqual(split_cents(0, 2), [0, 0])

    def test_errors(self):
        with self.assertRaises(ValueError):
            split_cents(10, 0)
        with self.assertRaises(ValueError):
            split_cents(-1, 2)

    def test_invoice_shares_order(self):
        shares = Invoice([("x", 101)]).split_between(["a", "b"])
        self.assertEqual(shares, {"a": 51, "b": 50})
