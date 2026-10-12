def parse_ini(text):
    result = {}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            result.setdefault(section, {})
            continue
        key, _, value = line.partition("=")
        result.setdefault(section, {})[key.strip()] = value.strip()
    return result
