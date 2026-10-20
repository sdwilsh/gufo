def split_cents(total_cents, parts):
    """Split an amount of cents into `parts` shares that sum exactly to the total."""
    if parts < 1 or total_cents < 0:
        raise ValueError("invalid split")
    base, extra = divmod(total_cents, parts)
    return [base + 1 if i < extra else base for i in range(parts)]
