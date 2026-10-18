def moving_average(values, window):
    """Average of each full window of `window` consecutive values."""
    if window <= 0:
        raise ValueError("window must be positive")
    out = []
    for i in range(len(values) - window + 1):
        out.append(sum(values[i:i + window]) / window)
    return out
