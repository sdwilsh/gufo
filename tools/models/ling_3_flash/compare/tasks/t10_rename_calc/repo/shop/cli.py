from shop.calc import calc
from shop.checkout import receipt


def summary_line(items):
    return f"{len(items)} items, total {calc(items):.2f}"


def main(items):
    r = receipt(items, 0.1)
    return summary_line(items) + f" (with tax {r['total']:.2f})"
