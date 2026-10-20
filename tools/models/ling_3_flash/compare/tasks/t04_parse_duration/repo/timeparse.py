UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text):
    total = 0
    number = ""
    for ch in text.strip():
        if ch.isdigit():
            number += ch
        elif ch in UNITS:
            total += int(number) * UNITS[ch]
            number = ""
        else:
            raise ValueError(f"bad character {ch!r} in duration")
    return total
