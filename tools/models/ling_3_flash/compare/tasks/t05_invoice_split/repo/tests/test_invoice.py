import unittest

from billing.invoice import Invoice


class InvoiceTest(unittest.TestCase):
    def test_total(self):
        self.assertEqual(Invoice([("a", 250), ("b", 100)]).total, 350)

    def test_split_adds_up(self):
        shares = Invoice([("dinner", 1000)]).split_between(["ann", "bob", "cy"])
        self.assertEqual(sum(shares.values()), 1000)
