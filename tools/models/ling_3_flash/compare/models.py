"""Per-model settings: artifact, serve arguments and where each comes from.

All three models are served with the same context, sessions, and cache
settings (set in compare.py).  Each uses the sampling and reasoning defaults
that `gufo serve` applies when a request omits them.  Those defaults are the
vendors' recommendations (docs/SERVER.md "Sampling defaults").  The summary
records the server's own startup sampling log.
"""
import os

MODELS = {
    "ling": {
        "label": "Ling-3.0-flash AD-Q4_K_S",
        "model": "/models/Ling-3.0-flash-AD-Q4_K_S-00001-of-00002.gguf",
        "served_name": "Ling-3.0-flash",
        "sampling": "server default for Ling-3.0-flash: temperature 0.6, top_p 0.95, top_k 20 "
                    "(model card defaults; tests/functional ling3 preset)",
        "thinking": "on (template default)",
        "serve_extra": [],
        "bench_jsonl": True,
        "speculative": None,
        "speculative_note": "Ling's default is MTP (its GGUF block 42), already used by the default bench and "
                            "serve; no separate pass",
        "sources": ["artifacts/ling-3-flash (this branch)", "docs/SERVER.md#sampling"],
    },
    "ds4": {
        "label": "DeepSeek V4 Flash 0731 IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8",
        "model": "/models2/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
        "served_name": "DeepSeek-V4-Flash",
        "sampling": "server default for DeepSeek V4 Flash 0731 agentic: temperature 1.0, top_p 0.95, "
                    "top_k 0 (DeepSeek-V4-Flash-0731 model card)",
        "thinking": "on, effort high (DeepSeek API default; docs/models/deepseek-v4-flash/README.md)",
        "serve_extra": [],
        "bench_jsonl": False,
        "speculative": None,
        "speculative_note": "DSpark is optional in docs/models/deepseek-v4-flash/README.md and its "
                            "support GGUF is not on the node; AR only",
        "sources": ["docs/models/deepseek-v4-flash/README.md", "docs/SERVER.md#sampling"],
    },
    "flashnext": {
        "label": "Qwen3.8 Flash-Next UD-Q4_K_XL",
        "model": "/models2/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
        "served_name": "Qwen3.8-Flash-Next",
        "sampling": "server default for Qwen3.8 Flash-Next thinking: temperature 1.0, top_p 0.95, top_k 20, "
                    "presence 0 (Qwen3.8-Flash-Next model card best practices)",
        "thinking": "on, effort xhigh, prior reasoning preserved (official template default)",
        "serve_extra": [],
        "bench_jsonl": False,
        "speculative": None,
        "speculative_note": "AR by default; measure MTP with --spec-mode mtp --spec-model <sidecar> "
                            "(docs README recommends --speculative mtp; AR GTT peak 81.6 GiB + 2.6 GiB sidecar "
                            "fits 96 GiB)",
        "sources": ["docs/models/qwen3.8-flash-next/README.md", "docs/SERVER.md#sampling"],
    },
}


def resolve(name):
    spec = dict(MODELS[name])
    spec["model"] = os.environ.get("GUFO_COMPARE_MODEL", spec["model"])
    return spec
