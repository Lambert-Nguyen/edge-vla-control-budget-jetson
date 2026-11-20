#!/usr/bin/env python3
"""FP16 and BF16 Hy-VLA on the workstation GPU: load, size and chunk latency.

This is the first part of the FP16 baseline (Q-1 in the export design spec).
It loads the released RoboTwin checkpoint, counts the weights of each part
(backbone and action expert), and times one action chunk with dummy inputs.
It does not measure task success. That needs RoboTwin.

Run it in the Hy-VLA uv env on the workstation:

    ~/hyvla/Hy-Embodied-0.5-VLA/.venv/bin/python export/hyvla_baseline_timing.py \\
        --ckpt ~/hyvla/ckpt/Hy-Embodied-0.5-VLA-RoboTwin --json results/<name>.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_state() -> dict:
    def g(*a):
        try:
            return subprocess.check_output(["git", "-C", str(REPO_ROOT), *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def summarize(xs: list[float]) -> dict:
    s = sorted(xs)

    def pct(p):
        return s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]
    return {"n": len(xs), "mean_ms": statistics.fmean(xs), "p50_ms": pct(50), "p90_ms": pct(90),
            "p99_ms": pct(99), "min_ms": s[0], "max_ms": s[-1]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dtypes", default="bfloat16,float16")
    ap.add_argument("--steps", type=int, help="denoising steps (default: the checkpoint's)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import torch
    from hy_vla import HyVLA, HyVLAConfig

    cfg = HyVLAConfig.from_pretrained(args.ckpt)
    if args.steps:
        cfg.num_steps = args.steps
    t = time.perf_counter()
    policy = HyVLA.from_pretrained(args.ckpt, config=cfg)
    policy.enable_video_encoder_if_needed()
    load_s = time.perf_counter() - t

    tower = policy.model.dual_tower
    parts = {"backbone_vlm": tower.vlm, "action_expert": tower.expert}
    counted = {k: sum(p.numel() for p in m.parameters()) for k, m in parts.items()}
    total = sum(p.numel() for p in policy.parameters())
    counted["other"] = total - sum(counted.values())
    params_m = {k: v / 1e6 for k, v in counted.items()}

    k_hist = 6  # history slots of the memory encoder, as in the repo quick start
    res = cfg.resize_imgs_with_padding[0]
    runs = {}
    for name in args.dtypes.split(","):
        dtype = getattr(torch, name)
        policy = policy.to(device="cuda", dtype=dtype).eval()
        img = torch.rand(1, k_hist, 3, res, res, device="cuda", dtype=dtype)
        state = torch.zeros((1, cfg.max_state_dim), device="cuda", dtype=dtype)
        batch = {"observation.images.top_head": img, "observation.images.hand_left": img,
                 "observation.images.hand_right": img, "observation.state": state,
                 "task": ["pick up the bottle"]}
        with torch.no_grad():
            for _ in range(args.warmup):
                policy.forward_evaluate(batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ms, finite = [], True
            for _ in range(args.iters):
                t = time.perf_counter()
                pred = policy.forward_evaluate(batch)["pred"]
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t) * 1e3)
                finite &= bool(torch.isfinite(pred).all())
        runs[name] = {"chunk": summarize(ms), "finite": finite,
                      "pred_shape": list(pred.shape),
                      "weights_gib": sum(p.numel() * p.element_size() for p in policy.parameters()) / 2**30,
                      "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}

    gpu = torch.cuda.get_device_name(0)
    print("=" * 72)
    print(f"Hy-VLA {Path(args.ckpt).name} on {gpu}")
    print(f"3 cameras x {k_hist} history frames at {res} px, chunk {cfg.chunk_size}, "
          f"{cfg.num_steps} denoising steps, loaded in {load_s:.0f} s")
    print("=" * 72)
    print("  params  " + ", ".join(f"{k} {v:.0f} M" for k, v in params_m.items()) + f", total {total / 1e6:.0f} M")
    for name, r in runs.items():
        c = r["chunk"]
        print(f"  {name:9s} chunk p50 {c['p50_ms']:6.1f} ms  p99 {c['p99_ms']:6.1f} ms   "
              f"weights {r['weights_gib']:.2f} GiB, gpu peak {r['gpu_peak_allocated_gib']:.2f} GiB, "
              f"{'finite' if r['finite'] else 'NaN/inf!'}")

    if args.json:
        report = {"kind": "hyvla_workstation_timing",
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "config": {k: str(v) for k, v in vars(args).items()},
                  "checkpoint": {"path": args.ckpt, "chunk_size": cfg.chunk_size, "num_steps": cfg.num_steps,
                                 "image_px": res, "history_frames": k_hist, "cameras": 3},
                  "params_m": params_m, "load_s": load_s, "runs": runs,
                  "host": {"gpu": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
                           "python": platform.python_version()},
                  "git": git_state()}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0 if all(r["finite"] for r in runs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
