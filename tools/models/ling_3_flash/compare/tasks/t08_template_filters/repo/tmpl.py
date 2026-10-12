import re

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render(template, context):
    def substitute(match):
        name = match.group(1)
        return str(context[name])

    return _PLACEHOLDER.sub(substitute, template)
