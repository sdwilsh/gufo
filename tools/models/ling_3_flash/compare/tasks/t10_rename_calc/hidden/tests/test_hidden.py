import ast
import pathlib
import unittest
import warnings

from shop import calc as calc_module

ROOT = pathlib.Path(calc_module.__file__).parent


class Hidden(unittest.TestCase):
    def test_new_name(self):
        self.assertEqual(calc_module.compute_total([("a", 1.25, 3), ("b", 2, 1)]), 5.75)
        self.assertEqual(calc_module.compute_total([]), 0)

    def test_deprecated_alias(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(calc_module.calc([("a", 2, 2)]), 4)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_no_other_module_uses_old_name(self):
        for path in ROOT.glob("*.py"):
            if path.name == "calc.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "shop.calc":
                    self.assertNotIn("calc", [a.name for a in node.names], path.name)
                if isinstance(node, ast.Attribute) and node.attr == "calc":
                    self.fail(f"{path.name} still references .calc")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "calc":
                    self.fail(f"{path.name} still calls calc()")

    def test_callers_do_not_warn(self):
        from shop.cart import Cart
        from shop.checkout import receipt
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            c = Cart()
            c.add("a", 2)
            c.total()
            receipt([("a", 1, 1)])
