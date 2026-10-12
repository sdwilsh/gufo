def to_csv(rows):
    lines = []
    for row in rows:
        cells = []
        for value in row:
            text = str(value)
            if "," in text or '"' in text or "\n" in text:
                text = '"' + text.replace('"', '""') + '"'
            cells.append(text)
        lines.append(",".join(cells))
    return "\n".join(lines)


def to_tsv(rows):
    lines = []
    for row in rows:
        cells = []
        for value in row:
            text = str(value)
            if "\t" in text or '"' in text or "\n" in text:
                text = '"' + text.replace('"', '""') + '"'
            cells.append(text)
        lines.append("\t".join(cells))
    return "\n".join(lines)


def to_pipe(rows):
    lines = []
    for row in rows:
        cells = []
        for value in row:
            text = str(value)
            if "|" in text or '"' in text or "\n" in text:
                text = '"' + text.replace('"', '""') + '"'
            cells.append(text)
        lines.append("|".join(cells))
    return "\n".join(lines)
