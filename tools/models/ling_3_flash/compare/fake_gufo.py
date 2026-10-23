#!/usr/bin/env python3
"""Development stand-in for the gufo binary.

serve runs mock_server.py, bench prints a canned table, and eval runs the
binary named by $GUFO_REAL."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "serve":
        port = rest[rest.index("--port") + 1]
        name = rest[rest.index("--served-model-name") + 1]
        extra = os.environ.get("FAKE_MOCK_ARGS", "").split()
        os.execv(sys.executable, [sys.executable, str(HERE / "mock_server.py"), "--port", port,
                                  "--name", name, *extra])
    if cmd == "bench":
        if "jsonl" in rest:
            print('{"record": "header", "model_type": "mock"}')
            for d in (0, 16384, 65536):
                print('{"record": "result", "n_prompt": 2048, "n_gen": 0, "n_depth": %d, "avg_ts": 800.0, '
                      '"samples_ts": [799.0, 800.0, 801.0]}' % d)
                print('{"record": "result", "n_prompt": 0, "n_gen": 128, "n_depth": %d, "avg_ts": 30.0}' % d)
        else:
            print("| model | size | backend | test | t/s |")
            print("| ----- | ---: | ------- | ---: | ---: |")
            for d in (0, 16384, 65536):
                print(f"| mock | 1.00 GiB | ROCm (HIP) | pp2048 @ d{d} | 800.00 ± 1.00 |")
                print(f"| mock | 1.00 GiB | ROCm (HIP) | tg128 @ d{d} | 30.00 ± 0.10 |")
        return 0
    if cmd == "eval" and os.environ.get("GUFO_REAL"):
        os.execv(os.environ["GUFO_REAL"], [os.environ["GUFO_REAL"], *sys.argv[1:]])
    print("fake_gufo: unsupported", cmd, file=sys.stderr)
    return 2


sys.exit(main())
