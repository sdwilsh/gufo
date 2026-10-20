import warnings


def compute_total(items):
    """Total price of (name, price, quantity) tuples, rounded to cents."""
    return round(sum(price * quantity for _, price, quantity in items), 2)


def calc(items):
    warnings.warn("calc() is deprecated; use compute_total()", DeprecationWarning, stacklevel=2)
    return compute_total(items)
