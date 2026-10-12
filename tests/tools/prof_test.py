#!/usr/bin/env python3

import importlib.util
import argparse
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "prof" / "prof.py"
SPEC = importlib.util.spec_from_file_location("gufo_prof", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load profiler module")
prof = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prof
SPEC.loader.exec_module(prof)

for quant in (12, 13, 23):
    symbol = (
        f"SmallBatchKQuantExactFp32GEMMKernel<16u, 8ul, 3ul, "
        f"(gufo::core::GgmlType){quant}, 1ul, 12u, 1ul>"
    )
    assert prof.short(f"void gufo::hip::{symbol}(void const*, float*)") == (
        symbol.replace("gufo::", "")
    )
    anonymous = prof.short(
        f"void gufo::hip::(anonymous namespace)::{symbol}(void const*, float*)"
    )
    assert anonymous == symbol.replace("gufo::", "")

assert prof.stage_of(
    prof.short(
        "void gufo::hip::(anonymous namespace)::HalfPrefillGemmKernel"
        "<256, 256, (gufo::core::GgmlType)13, true>(void const*, float*)"
    ),
    prof.STAGE_MAPS["qwen"],
) == "gemm: quant x fp16 prefill"

for symbol, expected in (
    ("qfn_mmq::mul_mat_vec_q_moe<(qfn_mmq::ggml_type)12, 2>",
     "moe: quantized vectors"),
    ("qfn_mmq::mul_mat_vec_q8<7, true>", "dense: quantized vectors"),
    ("SelectMarkKernel", "attention: exact selection"),
):
    assert prof.stage_of(symbol, prof.STAGE_MAPS["qwen-flash"]) == expected

with tempfile.TemporaryDirectory() as directory:
    database = Path(directory) / "profile.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE rocpd_info_kernel_symbol (
            id INTEGER PRIMARY KEY,
            display_name TEXT
        );
        CREATE TABLE rocpd_kernel_dispatch (
            id INTEGER PRIMARY KEY,
            kernel_id INTEGER,
            start INTEGER,
            end INTEGER,
            grid_size_x INTEGER,
            grid_size_y INTEGER,
            grid_size_z INTEGER,
            workgroup_size_x INTEGER,
            workgroup_size_y INTEGER,
            workgroup_size_z INTEGER
        );
        INSERT INTO rocpd_info_kernel_symbol VALUES (1, 'valid_kernel');
        INSERT INTO rocpd_info_kernel_symbol VALUES (2, 'invalid_kernel');
        INSERT INTO rocpd_kernel_dispatch
        VALUES (1, 1, 100, 300, 64, 1, 1, 32, 1, 1);
        INSERT INTO rocpd_kernel_dispatch
        VALUES (2, 2, 500, 400, 64, 1, 1, 32, 1, 1);
        """
    )
    connection.commit()
    connection.close()

    result = prof.load(str(database))
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO rocpd_kernel_dispatch VALUES "
        "(3, 1, 150, 200, 64, 1, 1, 32, 1, 1)"
    )
    connection.commit()
    connection.close()
    overlapping = prof.load(str(database))

assert result.dispatches == 1
assert result.invalid_dispatches == 1
assert result.total_ns == 200.0
assert result.busy_ns == 200.0
assert result.wall_ns == 200.0
assert "valid_kernel" in result.kernels
assert "invalid_kernel" not in result.kernels

assert overlapping.total_ns == 250.0
assert overlapping.busy_ns == 200.0
assert overlapping.wall_ns == 200.0

with tempfile.TemporaryDirectory() as directory:
    # A failing profiled command: the exit code is preserved and the END of
    # the output (the exception line) reaches stdout even after long output.
    fake = Path(directory) / "rocprofv3"
    fake.write_text(
        "#!/bin/sh\nfor i in $(seq 1 3000); do echo noise $i; done\n"
        "echo 'ValueError: the real failure' >&2\nexit 23\n"
    )
    fake.chmod(0o755)
    out_dir = Path(directory) / "out"
    args = argparse.Namespace(command=["false"], out=str(out_dir), tag="failed")
    import contextlib
    import io
    import os

    captured = io.StringIO()
    old_path = os.environ["PATH"]
    os.environ["PATH"] = f"{directory}:{old_path}"
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(io.StringIO()):
            try:
                prof.cmd_run(args)
            except SystemExit as error:
                assert error.code == 23
            else:
                raise AssertionError("profiler hid the child command failure")
    finally:
        os.environ["PATH"] = old_path
    assert "ValueError: the real failure" in captured.getvalue()
    assert "exited 23" in captured.getvalue()
    assert (out_dir / "failed.log").exists()

print("Profiler tests passed.")
