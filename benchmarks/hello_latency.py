#!/usr/bin/env python3
"""First latency measurement, and the template every later harness copies.

By default it times a small stand-in vision model so you can take a real
measurement before any VLA checkpoint exists. The point is the harness, not the
model: warmup discarded, percentiles over the steady state, peak memory, board
power mode, git commit, and the derived control rate all captured in one
machine-readable record.

    python3 benchmarks/hello_latency.py
    python3 benchmarks/hello_latency.py --iters 200 --resolution 224 --chunk 8
    python3 benchmarks/hello_latency.py --out results/baseline_dummy.json

Read the output as: if the policy emits `chunk` actions every `p50` ms, the arm
can be commanded at `chunk / p50` Hz without ever waiting on inference.

CONTRIBUTING.md requires every result to carry its config and commit. This
script does that automatically. Copy it rather than starting a harness from
scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sh(cmd: list[str], timeout: int = 15) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout.strip() or None


def board_state() -> dict:
    """Everything needed to make this number reproducible.

    A latency figure without the power mode and clock state is not a result.
    """
    model = Path("/proc/device-tree/model")
    l4t = Path("/etc/nv_tegra_release")

    clocks = sh(["sudo", "-n", "jetson_clocks", "--show"])
    gpu_clock = None
    if clocks:
        for line in clocks.splitlines():
            if line.strip().startswith("GPU"):
                gpu_clock = line.strip()
                break

    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "model": (model.read_bytes().decode("utf-8", "ignore").rstrip("\x00").strip()
                  if model.exists() else None),
        "l4t": l4t.read_text().splitlines()[0].strip() if l4t.exists() else None,
        "nvpmodel": sh(["sudo", "-n", "nvpmodel", "-q"]),
        "jetson_clocks_gpu": gpu_clock,
        "python": platform.python_version(),
    }


def git_state() -> dict:
    def g(*args: str) -> str | None:
        return sh(["git", "-C", str(REPO_ROOT), *args])

    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        # A measurement taken with uncommitted changes cannot be reproduced.
        "dirty": bool(g("status", "--porcelain")),
    }


def build_dummy_model(torch, resolution: int, chunk: int):
    """A stand-in with roughly VLA-shaped inputs and outputs.

    Not a scientific baseline. It exists so the harness can be exercised,
    debugged, and reviewed before the real checkpoint is on the board.
    """
    import torch.nn as nn

    class DummyPolicy(nn.Module):
        def __init__(self, chunk: int, dof: int = 6):
            super().__init__()
            self.vision = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Linear(128 + dof, 256), nn.ReLU(),
                nn.Linear(256, chunk * dof),
            )
            self.chunk, self.dof = chunk, dof

        def forward(self, image, state):
            features = self.vision(image)
            out = self.head(torch.cat([features, state], dim=1))
            return out.view(-1, self.chunk, self.dof)

    model = DummyPolicy(chunk).cuda().eval().half()
    image = torch.randn(1, 3, resolution, resolution, device="cuda", dtype=torch.float16)
    state = torch.randn(1, 6, device="cuda", dtype=torch.float16)
    return model, (image, state)


def measure(torch, model, inputs, iters: int, warmup: int) -> list[float]:
    """Return per-iteration wall-clock latency in milliseconds.

    Warmup is discarded because the first calls include CUDA context creation,
    kernel autotuning, and memory-pool growth. Including them inflates p99 by
    hundreds of milliseconds and tells you nothing about steady-state control.
    """
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            model(*inputs)
            # Synchronize inside the loop. CUDA launches are asynchronous, so
            # timing without this measures queueing, not execution.
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)

    def pct(p: float) -> float:
        # Nearest-rank: the smallest value at or below which p% of samples fall.
        idx = math.ceil(p / 100.0 * len(ordered)) - 1
        return ordered[max(0, min(len(ordered) - 1, idx))]

    return {
        "n": len(samples),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": ordered[0],
        "p50_ms": pct(50),
        "p90_ms": pct(90),
        "p99_ms": pct(99),
        "max_ms": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--chunk", type=int, default=8,
                        help="action chunk length, used to derive control rate")
    parser.add_argument("--target-hz", type=float, default=30.0,
                        help="control rate the arm needs")
    parser.add_argument("--label", default="dummy-fp16",
                        help="short name for this configuration")
    parser.add_argument("--config", default=None,
                        help="path to the YAML config this run came from")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the JSON record")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("[FAIL] torch not importable. See docs/guides/03-python-environments.md",
              file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available. You are almost certainly in the CPU "
              "environment or on a PyPI torch wheel. "
              "See docs/guides/03-python-environments.md", file=sys.stderr)
        return 1

    torch.cuda.reset_peak_memory_stats()

    model, inputs = build_dummy_model(torch, args.resolution, args.chunk)
    samples = measure(torch, model, inputs, args.iters, args.warmup)
    stats = summarize(samples)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

    # With action chunking the policy emits `chunk` actions per inference, so
    # the loop closes at chunk/latency Hz. The p99 number is the one that
    # decides whether the arm ever stalls.
    sustained_hz = args.chunk / (stats["p50_ms"] / 1000.0)
    worst_hz = args.chunk / (stats["p99_ms"] / 1000.0)
    meets_budget = worst_hz >= args.target_hz

    record = {
        "schema": "latency/v1",
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "config_file": args.config,
            "resolution": args.resolution,
            "action_chunk": args.chunk,
            "precision": "fp16",
            "iters": args.iters,
            "warmup": args.warmup,
            "target_hz": args.target_hz,
        },
        "latency": stats,
        "memory": {"peak_allocated_gib": peak_gb, "peak_reserved_gib": reserved_gb},
        "control_rate": {
            "sustained_hz_at_p50": sustained_hz,
            "worst_case_hz_at_p99": worst_hz,
            "meets_budget": meets_budget,
        },
        "board": board_state(),
        "git": git_state(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }

    print("=" * 68)
    print(f"{args.label}   {args.resolution}px   chunk={args.chunk}")
    print("=" * 68)
    print(f"  p50 {stats['p50_ms']:7.2f} ms    p90 {stats['p90_ms']:7.2f} ms"
          f"    p99 {stats['p99_ms']:7.2f} ms")
    print(f"  mean {stats['mean_ms']:6.2f} ms    sd  {stats['stdev_ms']:7.2f} ms"
          f"    max {stats['max_ms']:7.2f} ms")
    print(f"  peak gpu memory {peak_gb:.2f} GiB allocated, {reserved_gb:.2f} GiB reserved")
    print(f"  control rate    {sustained_hz:.1f} Hz typical, {worst_hz:.1f} Hz worst case")
    print(f"  budget {args.target_hz:.0f} Hz: {'MET' if meets_budget else 'MISSED'}")

    if record["git"]["dirty"]:
        print("\n[warn] working tree is dirty. This run cannot be reproduced from a "
              "commit. Commit before recording anything that goes in the report.")

    out = args.out or (REPO_ROOT / "results" /
                       f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
