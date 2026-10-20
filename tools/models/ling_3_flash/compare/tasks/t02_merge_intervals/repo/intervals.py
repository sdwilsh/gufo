def merge(intervals):
    """Merge overlapping or touching [start, end] intervals.

    Returns a new sorted list of (start, end) tuples; the input is not modified.
    """
    intervals.sort()
    result = []
    for start, end in intervals:
        if result and start < result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result
