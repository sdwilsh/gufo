class Counter:
    def __init__(self):
        self._counts = {}

    def hit(self, name):
        self._counts[name] = self._counts.get(name, 0) + 1
        return self._counts[name]

    def count(self, name):
        return self._counts.get(name, 0)

    def total(self):
        return sum(self._counts.values())

    def reset(self):
        self._counts.clear()


default_counter = Counter()


def hit(name):
    return default_counter.hit(name)


def count(name):
    return default_counter.count(name)


def total():
    return default_counter.total()


def reset():
    default_counter.reset()
