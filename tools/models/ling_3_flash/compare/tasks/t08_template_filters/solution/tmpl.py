import re

_PLACEHOLDER = re.compile(r"\{\{\s*(.*?)\s*\}\}")
_FILTERS = {"upper": str.upper, "lower": str.lower, "title": str.title}


def render(template, context):
    def substitute(match):
        parts = [p.strip() for p in match.group(1).split("|")]
        name, filters = parts[0], parts[1:]
        default = None
        for f in filters:
            if f.startswith("default:"):
                default = f[len("default:"):].strip()
        value = context.get(name)
        if value is None:
            if default is not None:
                value = default
            elif name in context:
                value = ""
            else:
                raise KeyError(name)
        text = str(value)
        for f in filters:
            if f.startswith("default:"):
                continue
            if f not in _FILTERS:
                raise ValueError(f"unknown filter {f!r}")
            text = _FILTERS[f](text)
        return text

    return _PLACEHOLDER.sub(substitute, template)
