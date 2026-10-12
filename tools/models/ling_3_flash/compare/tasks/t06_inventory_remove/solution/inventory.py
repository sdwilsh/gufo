class InsufficientStock(Exception):
    pass


class Inventory:
    def __init__(self):
        self._stock = {}

    def add(self, sku, qty):
        if qty <= 0:
            raise ValueError("qty must be positive")
        self._stock[sku] = self._stock.get(sku, 0) + qty

    def count(self, sku):
        return self._stock.get(sku, 0)

    def remove(self, sku, qty):
        if qty <= 0:
            raise ValueError("qty must be positive")
        if sku not in self._stock:
            raise KeyError(sku)
        if self._stock[sku] < qty:
            raise InsufficientStock(f"{sku}: have {self._stock[sku]}, need {qty}")
        self._stock[sku] -= qty

    def low_stock(self, threshold):
        items = [(sku, n) for sku, n in self._stock.items() if n <= threshold]
        return sorted(items, key=lambda item: (item[1], item[0]))
