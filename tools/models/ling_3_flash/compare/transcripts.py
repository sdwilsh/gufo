#!/usr/bin/env python3
"""Agent episode transcripts: the side file under --work and the pod-log copy.

Record: one JSON object per episode (task, seed, repeat, mode, passed, stop,
grade, system, prompt, turns[]), each turn holding the assistant content,
reasoning, finish_reason, usage and every tool call with the raw arguments
string exactly as the server sent it, its validation error and the tool result
text sent back.  The record's bytes are canonical JSON (sorted keys, no spaces,
UTF-8); its sha256 is the sha256 of exactly those bytes.

File: <work>/agent-transcripts-<mode>.jsonl.gz (agent-transcripts.jsonl.gz
without --agent-speculative), one gzip member per episode holding
`<record bytes>\\n`, so `gzip -dc FILE` is JSONL and every line hashes to its
episode's transcript sha256.

Log (--agent-transcript-log): gzip(record bytes), base64, split into lines
    [transcript] <key> 0/<n> sha256=<hex> bytes=<record bytes>
    [transcript] <key> <i>/<n> <base64 chunk>        i = 1..n
where <key> is <mode>/<task>/r<repeat>/s<seed>.  Text before `[transcript]`
(kubectl --timestamps, other prefixes) is ignored and lines may interleave.

    transcripts.py decode POD.log [-o OUT.jsonl[.gz]]   reassemble, verify every sha256
    transcripts.py verify FILE.jsonl.gz [...]           per-record and whole-file sha256
Exit status 1 when any record is incomplete or fails its hash.
"""
import argparse
import base64
import gzip
import hashlib
import json
import re
import sys
import threading
from pathlib import Path

SCHEMA = "gufo.ling-compare.agent-transcript.v1"
CHUNK = 3000  # base64 characters per log line, below the container runtime line-split limit
LOCK = threading.Lock()
LINE = re.compile(r"\[transcript\] (\S+) (\d+)/(\d+) (\S.*?)\s*$")


def episode_key(mode, task, repeat, seed):
    return f"{mode or 'agent'}/{task}/r{repeat}/s{seed}"


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def append(path, record):
    """Append one episode as its own gzip member and return its reference."""
    data = canonical(record)
    with LOCK, open(path, "ab") as handle:
        handle.write(gzip.compress(data + b"\n", mtime=0))
    return {"key": record.get("key"), "file": str(path), "sha256": sha256_hex(data), "json_bytes": len(data)}


def file_info(path):
    data = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256_hex(data)}


def log_lines(key, data, width=CHUNK):
    text = base64.b64encode(gzip.compress(data, mtime=0)).decode()
    chunks = [text[i:i + width] for i in range(0, len(text), width)] or [""]
    n = len(chunks)
    return ([f"[transcript] {key} 0/{n} sha256={sha256_hex(data)} bytes={len(data)}"]
            + [f"[transcript] {key} {i}/{n} {c}" for i, c in enumerate(chunks, 1)])


def emit(key, data, stream=None):
    """Write the chunked record to the log (stderr) with no other transcript lines in between."""
    stream = stream or sys.stderr
    with LOCK:
        for line in log_lines(key, data):
            stream.write(line + "\n")
        stream.flush()


def decode_log(lines):
    """Reassemble every transcript in a log.

    Returns [{key, sha256, ok, error, record, data}] in the order each transcript
    started.  A key seen twice (a resumed run) yields two entries."""
    out, open_by_key = [], {}
    for line in lines:
        match = LINE.search(line)
        if not match:
            continue
        key, index, total, payload = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
        if index == 0:
            fields = dict(f.split("=", 1) for f in payload.split() if "=" in f)
            entry = {"key": key, "sha256": fields.get("sha256"), "bytes": int(fields.get("bytes", -1)),
                     "total": total, "chunks": {}, "ok": False, "error": None, "record": None, "data": None}
            out.append(entry)
            open_by_key[key] = entry
            continue
        entry = open_by_key.get(key)
        if entry is None or entry["total"] != total:
            continue
        entry["chunks"][index] = payload
    for entry in out:
        chunks = entry.pop("chunks")
        total = entry.pop("total")
        missing = [i for i in range(1, total + 1) if i not in chunks]
        if missing:
            entry["error"] = f"missing chunk(s) {missing[:8]} of {total}"
            continue
        try:
            data = gzip.decompress(base64.b64decode("".join(chunks[i] for i in range(1, total + 1)), validate=True))
        except (ValueError, OSError, EOFError) as error:
            entry["error"] = f"undecodable: {error}"
            continue
        if sha256_hex(data) != entry["sha256"] or len(data) != entry["bytes"]:
            entry["error"] = f"sha256 mismatch: got {sha256_hex(data)} ({len(data)} bytes)"
            continue
        entry.update(ok=True, data=data, record=json.loads(data))
    return out


def read_file(path):
    """Return [(sha256, record)] for a transcript file (one JSON line per episode)."""
    with gzip.open(path, "rb") as handle:
        return [(sha256_hex(line.rstrip(b"\n")), json.loads(line)) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    decode = sub.add_parser("decode", help="reassemble [transcript] lines from a pod log")
    decode.add_argument("log", type=Path)
    decode.add_argument("-o", "--output", type=Path,
                        help="write verified records as JSONL (gzip-compressed when the name ends in .gz)")
    verify = sub.add_parser("verify", help="check a transcript file copied out of the pod")
    verify.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args(argv)
    bad = 0
    if args.command == "decode":
        entries = decode_log(args.log.read_text(errors="replace").splitlines())
        for e in entries:
            if e["ok"]:
                r = e["record"]
                print(f"ok  {e['key']} sha256={e['sha256']} passed={r.get('passed')} stop={r.get('stop')} "
                      f"turns={len(r.get('turns') or [])}")
            else:
                bad += 1
                print(f"BAD {e['key']} sha256={e['sha256']}: {e['error']}")
        if args.output:
            body = b"".join(e["data"] + b"\n" for e in entries if e["ok"])
            args.output.write_bytes(gzip.compress(body, mtime=0) if args.output.suffix == ".gz" else body)
        print(f"{len(entries) - bad}/{len(entries)} transcripts verified" +
              (f"; wrote {args.output}" if args.output else ""))
    else:
        for path in args.files:
            info = file_info(path)
            try:
                records = read_file(path)
            except (OSError, EOFError, ValueError) as error:
                bad += 1
                print(f"BAD {path}: {error}")
                continue
            print(f"{path} bytes={info['bytes']} sha256={info['sha256']} episodes={len(records)}")
            for sha, r in records:
                print(f"  {r.get('key')} sha256={sha} passed={r.get('passed')} stop={r.get('stop')} "
                      f"turns={len(r.get('turns') or [])}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
