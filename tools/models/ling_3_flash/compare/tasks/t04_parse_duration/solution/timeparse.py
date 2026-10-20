UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text):
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    total = 0
    number = ""
    for ch in text:
        if ch.isspace():
            continue
        if ch.isdigit():
            number += ch
        elif ch in UNITS:
            if not number:
                raise ValueError(f"unit {ch!r} without a number")
            total += int(number) * UNITS[ch]
            number = ""
        else:
            raise ValueError(f"bad character {ch!r} in duration")
    if number:
        total += int(number)
    return total
