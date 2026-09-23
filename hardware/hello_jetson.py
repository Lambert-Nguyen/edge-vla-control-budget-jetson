#!/usr/bin/env python3
"""Board bring-up check for the Jetson Orin Nano.

Run this after flashing and after building a Python environment. It verifies
the pieces this project depends on and fails loudly on the ones that silently
degrade, above all a CPU-only torch that reports success at install time and
then never touches the GPU.

    python3 hardware/hello_jetson.py
    python3 hardware/hello_jetson.py --json results/board_state.json

Exit code is 0 only if every required check passes.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REQUIRED = "required"
OPTIONAL = "optional"

results: list[dict] = []


def record(name: str, status: str, detail: str, level: str = REQUIRED) -> None:
    results.append({"check": name, "status": status, "detail": detail, "level": level})
    tag = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]"}[status]
    print(f"{tag} {name}: {detail}")


def sh(cmd: list[str], timeout: int = 15) -> str | None:
    """Run a command, return stripped stdout, or None if it is unavailable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout.strip() or None


def check_board() -> None:
    model = Path("/proc/device-tree/model")
    if model.exists():
        # device-tree strings are NUL terminated
        name = model.read_bytes().decode("utf-8", "ignore").rstrip("\x00").strip()
        status = "ok" if "Orin" in name else "warn"
        record("board", status, name)
    else:
        record("board", "warn", f"not a Tegra device ({platform.machine()})", OPTIONAL)

    release = Path("/etc/nv_tegra_release")
    if release.exists():
        record("L4T", "ok", release.read_text().splitlines()[0].strip())
    else:
        record("L4T", "warn", "/etc/nv_tegra_release missing", OPTIONAL)


def check_memory() -> None:
    proc_meminfo = Path("/proc/meminfo")
    if not proc_meminfo.exists():
        record("memory", "warn", "/proc/meminfo missing, not a Linux host", OPTIONAL)
        return

    meminfo = {}
    for line in proc_meminfo.read_text().splitlines():
        key, _, value = line.partition(":")
        meminfo[key] = value.strip()
    total_gb = int(meminfo["MemTotal"].split()[0]) / 1024 / 1024
    avail_gb = int(meminfo["MemAvailable"].split()[0]) / 1024 / 1024
    # 8 GB modules report ~7.4 GiB after carve-outs.
    status = "ok" if total_gb > 6.5 else "warn"
    record("memory", status, f"{total_gb:.1f} GiB total, {avail_gb:.1f} GiB available")

    swap = sh(["swapon", "--show=NAME,TYPE,SIZE", "--noheadings"])
    record("swap", "ok", swap.replace("\n", " | ") if swap else "none configured", OPTIONAL)


def check_power_mode() -> None:
    """Capture the board state that every benchmark record must cite."""
    mode = sh(["sudo", "-n", "nvpmodel", "-q"])
    if mode:
        record("nvpmodel", "ok", " / ".join(mode.splitlines()[-2:]), OPTIONAL)
    else:
        record("nvpmodel", "warn", "run `sudo nvpmodel -q` manually", OPTIONAL)

    clocks = sh(["sudo", "-n", "jetson_clocks", "--show"])
    if clocks:
        gpu = [ln for ln in clocks.splitlines() if "GPU" in ln]
        record("jetson_clocks", "ok", gpu[0].strip() if gpu else "reported", OPTIONAL)
    else:
        record("jetson_clocks", "warn", "run `sudo jetson_clocks --show` manually", OPTIONAL)


def check_torch() -> "object | None":
    try:
        import torch
    except ImportError as exc:
        record("torch", "fail", f"import failed: {exc}")
        return None

    cuda_build = torch.version.cuda
    available = torch.cuda.is_available()
    detail = f"{torch.__version__}  cuda={cuda_build}  available={available}"

    if not available:
        record("torch", "fail", detail + "  <- CPU-only wheel, see guide 03")
        return None

    record("torch", "ok", detail)

    props = torch.cuda.get_device_properties(0)
    record(
        "cuda device",
        "ok" if props.major == 8 and props.minor == 7 else "warn",
        f"{props.name} sm_{props.major}{props.minor}, "
        f"{props.total_memory / 1024**3:.1f} GiB reported",
    )
    return torch


def check_tensorrt() -> None:
    try:
        import tensorrt
    except ImportError as exc:
        record(
            "tensorrt",
            "warn",
            f"not importable ({exc}). Create the venv with --system-site-packages "
            "rather than pip installing tensorrt",
            OPTIONAL,
        )
        return
    record("tensorrt", "ok", tensorrt.__version__)


def check_lerobot() -> None:
    try:
        import lerobot
    except ImportError:
        record("lerobot", "warn", "not installed in this environment", OPTIONAL)
        return
    record("lerobot", "ok", getattr(lerobot, "__version__", "version unknown"), OPTIONAL)


def check_numpy() -> None:
    try:
        import numpy
    except ImportError as exc:
        record("numpy", "fail", f"import failed: {exc}")
        return
    # Several Jetson-built wheels are compiled against the NumPy 1.x ABI.
    status = "ok" if numpy.__version__.startswith("1.") else "warn"
    note = "" if status == "ok" else "  <- pin numpy<2 if imports start failing"
    record("numpy", status, numpy.__version__ + note, OPTIONAL)


def bench_gpu(torch) -> None:
    """A real GPU workload, so 'CUDA available' is backed by arithmetic."""
    n, iters, warmup = 4096, 30, 10
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)

    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000

    tflops = (2 * n**3) / (ms / 1000) / 1e12
    # An Orin Nano in a 15-25 W mode lands in the single-digit TFLOP/s range.
    status = "ok" if tflops > 1.0 else "warn"
    record("matmul fp16 4096^3", status, f"{ms:.2f} ms  ({tflops:.1f} TFLOP/s)")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    record("peak gpu alloc", "ok", f"{peak_gb:.2f} GiB", OPTIONAL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the full check record here")
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print("Jetson bring-up check")
    print("=" * 68)

    check_board()
    check_memory()
    check_power_mode()
    torch = check_torch()
    check_tensorrt()
    check_numpy()
    check_lerobot()

    if torch is not None and not args.skip_bench:
        bench_gpu(torch)

    failures = [r for r in results if r["status"] == "fail" and r["level"] == REQUIRED]
    warnings = [r for r in results if r["status"] == "warn"]

    print("-" * 68)
    print(f"{len(results)} checks, {len(failures)} failed, {len(warnings)} warnings")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"checks": results}, indent=2) + "\n")
        print(f"wrote {args.json}")

    if failures:
        print("\nFailed checks block the project. See docs/guides/07-troubleshooting.md")
        return 1

    print("\nBoard is ready. Next: docs/guides/05-first-benchmark.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
