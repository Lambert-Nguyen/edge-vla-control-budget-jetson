#!/usr/bin/env python3
"""Check one TensorRT engine against reference cases from the export (TR-3).

For the Hy-VLA action expert before the full handoff package exists: the export
saves expert_reference.npz (inputs and ONNX Runtime outputs, see
export/export_onnx.py --reference). This runs the engine on each case, reports
the largest difference as a share of the largest output (limit 2%), and times
one call from Python.

    python3 engine/check_engine.py --engine build/engines/hyvla-expert-fp16.engine \\
        --reference build/onnx/hyvla-fp32/expert_reference.npz --json results/<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmarks"))
from trt_runtime import TrtEngine, mem_available_mib, sha256  # noqa: E402

LIMIT = 0.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import tensorrt as trt
    from hello_latency import board_state, git_state

    ref = dict(np.load(args.reference))
    cases = sorted({k.split("/")[0] for k in ref})
    ram0 = mem_available_mib()
    eng = TrtEngine(args.engine)
    ram1 = mem_available_mib()
    out_name = eng.outputs[0]
    errs = []
    for c in cases:
        out = eng(**{n: ref[f"{c}/{n}"] for n in eng.inputs})[out_name].float().cpu().numpy()
        want = ref[f"{c}/{out_name}"]
        errs.append(float(np.abs(out - want).max() / np.abs(want).max()))
    ok = max(errs) <= LIMIT
    feeds = {n: ref[f"{cases[0]}/{n}"] for n in eng.inputs}
    for _ in range(10):
        eng(**feeds)
    ts = []
    for _ in range(args.iters):
        t = time.perf_counter()
        eng(**feeds)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    p50, p99 = ts[len(ts) // 2], ts[int(len(ts) * 0.99)]
    print(f"{args.engine.name}: {args.engine.stat().st_size / 2**20:.0f} MiB, inputs {eng.inputs}")
    print("  vs reference: " + ", ".join(f"{c} {100 * e:.3f}%" for c, e in zip(cases, errs))
          + f" -> worst {100 * max(errs):.3f}% (limit {100 * LIMIT:.0f}%) {'PASS' if ok else 'FAIL'}")
    print(f"  one call from Python with input copies: p50 {p50:.2f} ms, p99 {p99:.2f} ms")
    print(f"  board RAM available: {ram0:.0f} MiB before load, {ram1:.0f} MiB after")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "kind": "trt_engine_check", "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine": {"file": args.engine.name, "mib": args.engine.stat().st_size / 2**20,
                       "sha256": sha256(args.engine), "inputs": eng.inputs, "outputs": eng.outputs,
                       "device_mib": eng.device_mib},
            "reference": {"file": args.reference.name, "cases": cases, "sha256": sha256(args.reference)},
            "check": {"max_rel_diff": errs, "limit": LIMIT, "pass": ok},
            "latency_ms": {"p50": p50, "p99": p99, "iters": args.iters},
            "ram_available_mib": {"before": ram0, "after": ram1},
            "tensorrt": trt.__version__, "board": board_state(), "git": git_state(),
        }, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
