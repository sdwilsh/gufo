import re


def _wrap_paragraph(words, width):
    lines = []
    current = ""
    for word in words:
        while len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def wrap(text, width):
    if width < 1:
        raise ValueError("width must be positive")
    paragraphs = [p.split() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if p]
    lines = []
    for i, words in enumerate(paragraphs):
        if i:
            lines.append("")
        lines.extend(_wrap_paragraph(words, width))
    return lines
