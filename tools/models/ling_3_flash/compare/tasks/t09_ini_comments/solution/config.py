def _value(raw):
    raw = raw.strip()
    if raw.startswith('"'):
        end = raw.find('"', 1)
        if end != -1:
            return raw[1:end]
    for i, ch in enumerate(raw):
        if ch in ";#" and i > 0 and raw[i - 1].isspace():
            return raw[:i].strip()
    return raw


def parse_ini(text):
    result = {}
    section = ""
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            result.setdefault(section, {})
            continue
        if "=" not in line:
            raise ValueError(f"line {number}: expected key = value")
        key, _, value = line.partition("=")
        result.setdefault(section, {})[key.strip()] = _value(value)
    return result
