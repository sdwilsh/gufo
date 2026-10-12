#!/usr/bin/env python3
"""Greedy token streams from a running llama-server (stdlib only).

Used by run_oracle.sh. Prompts are tokenised with /tokenize (add_special as
the server would do), then sent to /completion as token ids so the prompt ids
are exactly what the oracle consumed. Writes one JSON line per prompt:

  {"name", "prompt", "prompt_ids", "n_predict", "tokens", "text", "stop",
   "top": [[{"id", "logprob"} x n_probs] per generated token]}

    oracle_greedy.py --url http://127.0.0.1:8099 --prompts oracle_prompts.json
                     --n-predict 64 --out greedy.jsonl
"""

import argparse
import json
import urllib.request


def post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--n-probs", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)
    with open(args.out, "w") as out:
        for p in prompts:
            ids = post(args.url + "/tokenize", {"content": p["prompt"], "add_special": True})["tokens"]
            res = post(
                args.url + "/completion",
                {
                    "prompt": ids,
                    "n_predict": args.n_predict,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "repeat_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "dry_multiplier": 0.0,
                    "mirostat": 0,
                    "cache_prompt": False,
                    "return_tokens": True,
                    "n_probs": args.n_probs,
                    "stream": False,
                },
            )
            top = []
            for t in res.get("completion_probabilities", []):
                cand = t.get("top_logprobs") or t.get("top_probs") or []
                top.append([{"id": c["id"], "logprob": c.get("logprob")} for c in cand])
            rec = {
                "name": p["name"],
                "prompt": p["prompt"],
                "prompt_ids": ids,
                "n_predict": args.n_predict,
                "tokens": res["tokens"],
                "text": res["content"],
                "stop": res.get("stop_type"),
                "top": top,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{p['name']}: {len(rec['tokens'])} tokens, stop={rec['stop']}")


if __name__ == "__main__":
    main()
