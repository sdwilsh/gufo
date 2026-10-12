_counts = {}


def hit(name):
    _counts[name] = _counts.get(name, 0) + 1
    return _counts[name]


def count(name):
    return _counts.get(name, 0)


def total():
    return sum(_counts.values())


def reset():
    _counts.clear()
