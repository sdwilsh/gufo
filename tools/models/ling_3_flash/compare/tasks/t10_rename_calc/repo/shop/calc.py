def calc(items):
    """Total price of (name, price, quantity) tuples, rounded to cents."""
    return round(sum(price * quantity for _, price, quantity in items), 2)
