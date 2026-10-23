from shop.calc import compute_total


class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, quantity=1):
        self.items.append((name, price, quantity))

    def total(self):
        return compute_total(self.items)
