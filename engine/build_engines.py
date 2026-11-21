#!/usr/bin/env python3
"""Build, check and time the Hy-VLA engines from a handoff package, on the Jetson.

The flow of the engine builder design spec (Figure 1):

1. Read the package and check every hash in manifest.json.
2. Look in the engine cache (build/engines/). Build only what changed.
3. Load each engine after a memory check.
4. Run the reference cases. Stop with [FAIL] if any output differs from
   the package's reference by more than 2% of its largest value (TR-3).
5. Time the backbone, one expert step and a full chunk, and write a record.

    python3 engine/build_engines.py --handoff build/handoff/test-fp16
    python3 engine/build_engines.py --handoff DIR --steps 5 --json results/<name>.json

The real backbone comes in several parts (the manifest lists them). Each part
becomes its own engine, and they run as one chain. On the Jetson the big parts
only build when trtexec-style host copies can go to swap, so build them first
under a RAM cap, then run the check without it (the engines are then cache hits):

    systemd-run --user --scope -p MemoryHigh=1500M -p MemorySwapMax=infinity \\
        python3 engine/build_engines.py --handoff DIR --build-only
    python3 engine/build_engines.py --handoff DIR --json results/<name>.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "benchmarks"))
from trt_runtime import (EngineBuilder, EngineCache, EngineChain, HandoffPackage,  # noqa: E402
                         HyVlaEnginePolicy, TrtEngine, mem_available_mib)

LIMIT = 0.02  # TR-3: 2% of the largest reference value


def rel_err(got, want) -> float:
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-9))


def cosine(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def pct(xs, p):
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", required=True, type=Path)
    ap.add_argument("--steps", type=int, help="Euler steps per chunk (default: the manifest's)")
    ap.add_argument("--rebuild", action="store_true", help="ignore the engine cache")
    ap.add_argument("--workspace-mib", type=int, default=1024)
    ap.add_argument("--trt-verbose", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--build-only", action="store_true", help="build what is missing, then stop")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import tensorrt as trt
    import torch

    from hello_latency import board_state, git_state

    pkg = HandoffPackage.open(args.handoff)  # stops on a wrong hash
    m = pkg.manifest
    steps = args.steps or m["flow"]["steps_default"]
    print(f"package {m['name']}: precision {m['precision']}, {m['reference']['cases']} reference cases, "
          f"hashes OK")

    cache, builder = EngineCache(), EngineBuilder(args.workspace_mib, args.trt_verbose)
    graphs = pkg.parts("backbone") + pkg.parts("expert")
    # Build everything first, with no engine loaded, so each build has the most memory.
    keys, hits = {}, {}
    for name in graphs:
        key = EngineCache.key(pkg.file_hashes(name), pkg.precision(name), builder.settings())
        engine_path, side_path = cache.paths(key)
        keys[name] = key
        hits[name] = cache.find(key) is not None and not args.rebuild
        if not hits[name]:
            print(f"  building {name} ({pkg.precision(name)}) ...", flush=True)
            side = builder.build(pkg.onnx_path(name), pkg.precision(name), engine_path, side_path,
                                 extra={"package": m["name"], "graph": name})
            print(f"  {name} built in {side['build_s']:.0f} s, engine {side['engine_mib']:.0f} MiB", flush=True)
    if args.build_only:
        print("all engines are in the cache")
        return 0

    engines, facts = {}, {}
    ram_start = mem_available_mib()
    ram_low = ram_start
    for name in graphs:
        engine_path, side_path = cache.paths(keys[name])
        side = json.loads(side_path.read_text())
        engines[name] = TrtEngine(engine_path, min_free_mib=200)
        ram_low = min(ram_low, mem_available_mib())
        facts[name] = {**side, "cache_key": keys[name], "cache_hit": hits[name],
                       "device_mib": engines[name].device_mib}
        how = "cache hit" if hits[name] else f"built in {side['build_s']:.1f} s"
        print(f"  {name:12s} {how}, "
              f"engine {side['engine_mib']:.1f} MiB, working memory {engines[name].device_mib:.1f} MiB")

    def chain(name):
        parts = pkg.parts(name)
        if parts == [name]:
            return engines[name]
        g = pkg.graph(name)
        return EngineChain([engines[p] for p in parts], g["inputs"], g["outputs"])

    engines["backbone"], engines["expert"] = chain("backbone"), chain("expert")
    policy = HyVlaEnginePolicy(engines["backbone"], engines["expert"], steps, m["flow"]["context"])
    ref = pkg.reference()
    in_names = list(m["graphs"]["backbone"]["inputs"])
    worst = {"k": 0.0, "v": 0.0, "velocity_t1": 0.0, "actions": 0.0}
    cos_ref = {k: 1.0 for k in worst}
    vs_fp32 = {}
    for c in range(m["reference"]["cases"]):
        r = {k.split("/", 1)[1]: v for k, v in ref.items() if k.startswith(f"case{c:02d}/")}
        out = engines["backbone"](**{n: r[n] for n in in_names})
        for key in ("k", "v"):
            worst[key] = max(worst[key], rel_err(out[key].float().cpu(), r[key]))
            cos_ref[key] = min(cos_ref[key], cosine(out[key].float().cpu(), r[key]))
        v0 = engines["expert"](state=r["state"], prefix_pad_mask=out["prefix_pad_mask"], k=out["k"], v=out["v"],
                               x_t=r["noise"], t=np.array([1.0], np.float32))["velocity"]
        worst["velocity_t1"] = max(worst["velocity_t1"], rel_err(v0.float().cpu(), r["velocity_t1"]))
        cos_ref["velocity_t1"] = min(cos_ref["velocity_t1"], cosine(v0.float().cpu(), r["velocity_t1"]))
        if steps == m["flow"]["steps_default"]:
            acts = policy.sample_chunk({n: r[n] for n in in_names}, r["state"], r["noise"])
            worst["actions"] = max(worst["actions"], rel_err(acts.cpu(), r["actions"]))
            cos_ref["actions"] = min(cos_ref["actions"], cosine(acts.cpu(), r["actions"]))
            if "actions_fp32" in r:  # quantized packages: how far from the FP32 model (not a pass/fail check)
                d = {"k": cosine(out["k"].float().cpu(), r["k_fp32"]),
                     "v": cosine(out["v"].float().cpu(), r["v_fp32"]),
                     "actions": cosine(acts.cpu(), r["actions_fp32"]),
                     "actions_max_diff": rel_err(acts.cpu(), r["actions_fp32"]),
                     "reference_k": cosine(r["k"], r["k_fp32"]), "reference_v": cosine(r["v"], r["v_fp32"]),
                     "reference_actions": cosine(r["actions"], r["actions_fp32"])}
                vs_fp32[f"case{c:02d}"] = d
    ok = all(v <= LIMIT for v in worst.values())
    print("  check    " + ", ".join(f"{k} {100 * v:.3f}%" for k, v in worst.items()) +
          f"  (limit {100 * LIMIT:.0f}% of the largest value)")
    print("  cosine   " + ", ".join(f"{k} {v:.4f}" for k, v in cos_ref.items()) + "  (lowest over the cases)")
    for c, d in vs_fp32.items():
        print(f"  vs FP32  {c}: k cosine {d['k']:.4f}, v {d['v']:.4f}, actions {d['actions']:.4f} "
              f"(max diff {100 * d['actions_max_diff']:.2f}%). The reference itself: k {d['reference_k']:.4f}, "
              f"v {d['reference_v']:.4f}, actions {d['reference_actions']:.4f}")
    if not ok:
        print("[FAIL] the engines do not match the package's reference outputs. Not timing them.")
        return 1

    r = {k.split("/", 1)[1]: v for k, v in ref.items() if k.startswith("case00/")}
    feeds = {n: r[n] for n in in_names}
    timings = {"backbone": [], "expert_step": [], "chunk": []}
    for i in range(args.iters + 5):
        t = time.perf_counter()
        out = engines["backbone"](**feeds)
        tb = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        engines["expert"](state=r["state"], prefix_pad_mask=out["prefix_pad_mask"], k=out["k"], v=out["v"],
                          x_t=r["noise"], t=np.array([1.0], np.float32))
        te = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        policy.sample_chunk(feeds, r["state"], r["noise"])
        torch.cuda.synchronize()
        tc = (time.perf_counter() - t) * 1e3
        if i >= 5:
            timings["backbone"].append(tb)
            timings["expert_step"].append(te)
            timings["chunk"].append(tc)
        ram_low = min(ram_low, mem_available_mib())
    summary = {k: {"p50_ms": pct(v, 50), "p99_ms": pct(v, 99), "mean_ms": statistics.fmean(v)}
               for k, v in timings.items()}
    for k, s in summary.items():
        print(f"  {k:12s} p50 {s['p50_ms']:8.3f} ms   p99 {s['p99_ms']:8.3f} ms")
    print(f"  chunk = backbone + {steps} expert steps. Board RAM available: {ram_start:.0f} MiB at start, "
          f"{ram_low:.0f} MiB at the lowest")

    if args.json:
        board = board_state()
        report = {"kind": "hyvla_engines", "timestamp": datetime.now(timezone.utc).isoformat(),
                  "package": {"name": m["name"], "path": str(args.handoff), "precision": m["precision"],
                              "sha256": m["sha256"], "export_commit": m.get("export_commit")},
                  "tensorrt": trt.__version__, "steps": steps, "engines": facts,
                  "check": {"worst_rel_err": worst, "cosine": cos_ref, "limit": LIMIT, "pass": ok}, "vs_fp32": vs_fp32,
                  "latency": summary, "ram_available_mib": {"start": ram_start, "lowest": ram_low},
                  "board": board, "git": git_state()}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
