#!/usr/bin/env python3
"""Build a tensor-free, vocab-only GGUF from the head of a Ling-3.0-flash shard.

usage: make_vocab_gguf.py HEAD_SHARD.bin OUT   (committed as ling-vocab.gguf.fixture; *.gguf is gitignored)
HEAD_SHARD.bin only needs to contain the GGUF header and key/value metadata
(the first ~16 MiB of shard 1 is enough). Every key except split.* and
quantize.* is copied, so llama.cpp can load it with --vocab-only.
"""
import struct
import sys

import gguf
from gguf import GGUFValueType as T

FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
       10: "Q", 11: "q", 12: "d"}


def main(src, dst):
    d = open(src, "rb").read()
    o = 4

    def u(f):
        nonlocal o
        v = struct.unpack_from("<" + f, d, o)[0]
        o += struct.calcsize("<" + f)
        return v

    def s():
        nonlocal o
        n = u("Q")
        v = d[o:o + n].decode("utf-8")
        o += n
        return v

    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et = u("I")
            n = u("Q")
            return et, [val(et) for _ in range(n)]
        return u(FMT[t])

    assert d[:4] == b"GGUF"
    u("I"); u("Q")
    nkv = u("Q")
    w = gguf.GGUFWriter(dst, "bailingmoe3")
    for _ in range(nkv):
        k = s(); t = u("I"); v = val(t)
        if k.startswith(("split.", "quantize.")) or k == "general.architecture":
            continue
        if t == 9:
            et, items = v
            w.add_key_value(k, items, T.ARRAY, T(et))
        else:
            w.add_key_value(k, v, T(t))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
