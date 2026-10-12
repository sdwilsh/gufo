#!/usr/bin/env python3
"""Build the real-text corpora of the MTP speed/acceptance measurement from the
repository itself (no network, nothing memorised from the model's training
text): `prose.txt` = project documentation (Markdown) and `code.txt` = C++ and
Python sources, each well over 70k tokens so depth 64k plus the generation fits.

usage: make_corpus.py REPO_ROOT OUT_DIR [NOVEL_TEXT]
"""
import pathlib
import sys

root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
SKIP = {"build", "artifacts", ".git", "fixtures", "vendor", "licenses"}


def files(patterns, limit):
    paths = sorted(p for pat in patterns for p in root.rglob(pat)
                   if p.is_file() and not SKIP.intersection(p.relative_to(root).parts))
    text, size = [], 0
    for p in paths:
        try:
            data = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        text.append(data)
        size += len(data)
        if size >= limit:
            break
    return "\n\n".join(text)


prose = files(["*.md"], 900_000)
if len(sys.argv) > 3:
    prose = pathlib.Path(sys.argv[3]).read_text(encoding="utf-8") + "\n\n" + prose
(out / "prose.txt").write_text(prose, encoding="utf-8")
(out / "code.txt").write_text(files(["*.cpp", "*.hpp", "*.py"], 900_000), encoding="utf-8")
for name in ("prose.txt", "code.txt"):
    print(name, (out / name).stat().st_size, "bytes")
