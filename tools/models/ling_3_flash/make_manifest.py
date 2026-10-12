#!/usr/bin/env python3
"""Writes the reproducibility files of the oracle directory.

    make_manifest.py ORACLE_DIR

Copies the exact input text (`input_text.txt`), writes `prompts.jsonl` (the 3
greedy prompts as raw text and token ids, ids taken from greedy.jsonl) and
`text_tokens.txt` (the token ids llama-perplexity consumed, chunk per line),
then MANIFEST.json with the sha256 and size of every input and output file,
the exact command lines (parsed from COMMANDS.txt), the llama.cpp version, and
n_ctx / chunks / KV types / threads. Needs numpy (via kl_base.py).
"""

import hashlib
import json
import os
import re
import shutil
import sys

import kl_base

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_info(cmd):
    first = cmd.split(" -m ")[-1].split(" ")[0]
    info = {"first_shard": first, "shard_bytes": {}}
    second = first.replace("00001-of-00002", "00002-of-00002")
    for path in (first, second):
        if os.path.exists(path):
            info["shard_bytes"][os.path.basename(path)] = os.path.getsize(path)
    return info


def main():
    out = sys.argv[1]
    shutil.copyfile(f"{HERE}/oracle_text.txt", f"{out}/input_text.txt")
    shutil.copyfile(f"{HERE}/oracle_prompts.json", f"{out}/prompts_raw.json")

    greedy = [json.loads(l) for l in open(f"{out}/greedy.jsonl") if l.strip()]
    with open(f"{out}/prompts.jsonl", "w") as f:
        for r in greedy:
            f.write(json.dumps({"name": r["name"], "prompt": r["prompt"],
                                "prompt_ids": r["prompt_ids"]}) + "\n")

    base = kl_base.KlBase(f"{out}/kl_base.bin")
    with open(f"{out}/text_tokens.txt", "w") as f:
        for c in range(base.n_chunk):
            f.write(" ".join(str(int(t)) for t in base.tokens[c]) + "\n")

    version = None
    commands = {}
    for line in open(f"{out}/COMMANDS.txt"):
        line = line.rstrip("\n")
        if line.startswith("llama.cpp:"):
            version = line.split(":", 1)[1].strip()
        m = re.match(r"^(kl_base|greedy server|greedy client|kl self-check): (.*)$", line)
        if m:
            commands[m.group(1)] = m.group(2)
    kl_cmd = commands.get("kl_base", "")
    flag = lambda name: (re.search(rf"(?:^| ){name} (\S+)", kl_cmd) or [None, None])[1]

    files = {}
    for name in sorted(os.listdir(out)):
        p = f"{out}/{name}"
        if name == "MANIFEST.json" or not os.path.isfile(p):
            continue
        files[name] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    for name in ("oracle_text.txt", "oracle_prompts.json"):
        files[f"(repo) tools/models/ling_3_flash/{name}"] = {
            "sha256": sha256(f"{HERE}/{name}"), "bytes": os.path.getsize(f"{HERE}/{name}")}

    manifest = {
        "llama_cpp_version": version,
        "backend": "CPU (-ngl 0 -dev none)",
        "model": model_info(commands.get("kl_base", "")),
        "kl_base": {
            "n_ctx": base.n_ctx,
            "n_chunk": base.n_chunk,
            "n_vocab": base.n_vocab,
            "first_scored_position": base.first,
            "scored_rows_per_chunk": base.rows_per_chunk,
            "batch": flag("-b"),
            "kv_type_k": flag("-ctk"),
            "kv_type_v": flag("-ctv"),
            "threads": flag("-t"),
        },
        "greedy": {"n_predict": 64, "sampling": "temperature 0, top_k 1, no penalties",
                   "kv_type_k": flag("-ctk"), "kv_type_v": flag("-ctv"),
                   "prompts": [r["name"] for r in greedy]},
        "commands": commands,
        "files": files,
    }
    with open(f"{out}/MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}/MANIFEST.json ({len(files)} files)")


if __name__ == "__main__":
    main()
