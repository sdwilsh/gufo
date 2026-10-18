def format_rows(rows, sep):
    lines = []
    for row in rows:
        cells = []
        for value in row:
            text = str(value)
            if sep in text or '"' in text or "\n" in text:
                text = '"' + text.replace('"', '""') + '"'
            cells.append(text)
        lines.append(sep.join(cells))
    return "\n".join(lines)


def to_csv(rows):
    return format_rows(rows, ",")


def to_tsv(rows):
    return format_rows(rows, "\t")


def to_pipe(rows):
    return format_rows(rows, "|")
