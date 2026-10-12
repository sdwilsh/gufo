def wrap(text, width):
    if width < 1:
        raise ValueError("width must be positive")
    lines = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) < width:
            current += " " + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
