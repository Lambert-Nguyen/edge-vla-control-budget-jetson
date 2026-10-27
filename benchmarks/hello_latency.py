#!/usr/bin/env python3
"""First latency measurement, and the template every later harness copies.

By default it times a small stand-in vision model so you can take a real
measurement before any VLA checkpoint exists. The point is the harness, not the
model: warmup discarded, percentiles over the steady state, peak memory, board
power mode, clock state, board power over the timed loop, git commit, and the
derived control rate all captured in one machine-readable record.

    python3 benchmarks/hello_latency.py
    python3 benchmarks/hello_latency.py --iters 200 --resolution 224 --chunk 8
    python3 benchmarks/hello_latency.py --out results/baseline_dummy.json
    python3 benchmarks/hello_latency.py --runtime tensorrt

`--runtime tensorrt` exports the same model to ONNX, builds an FP16 TensorRT
engine on this board, checks that the engine gives the same actions as
PyTorch, and then times the engine in the same loop. The ONNX file and the
engine go to build/engines/, which git ignores.

Board power comes from the INA3221 rails in sysfs and needs no root. It is
sampled only while the timed loop runs, so a run shorter than a couple of
seconds yields too few samples to trust. Raise --iters for power numbers.

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
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fewer power samples than this and the mean and peak are mostly noise.
MIN_POWER_SAMPLES = 20

# The board setting every sweep record is taken in, decided 2026-10-26. See
# "Sweep setting" in README.md. Other modes are only for the power comparison.
SWEEP_MODE = "MAXN_SUPER"

# Engines and ONNX files are build output, never committed.
ENGINE_DIR = REPO_ROOT / "build" / "engines"

# TensorRT FP16 and PyTorch FP16 round differently. Past this share of the
# largest action, the engine computes something else.
TRT_MAX_REL_DIFF = 0.02


def sh(cmd: list[str], timeout: int = 15) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout.strip() or None


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def gpu_devfreq() -> Path | None:
    # The Orin GPU is a devfreq device, 17000000.gpu on the Orin Nano.
    hits = sorted(Path("/sys/class/devfreq").glob("*.gpu"))
    return hits[0] if hits else None


def thermal_zone(kind: str) -> Path | None:
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == kind:
                return zone / "temp"
        except OSError:
            continue
    return None


def clock_state() -> dict:
    """GPU and CPU clock limits, read from sysfs without root.

    `jetson_clocks --show` needs sudo. jetson_clocks works by raising each
    clock's minimum to its maximum, so min == max in sysfs is how to tell
    whether it is active without it.
    """
    state: dict = {"source": "sysfs"}
    gpu = gpu_devfreq()
    if gpu:
        lo, hi, cur = (read_int(gpu / f) for f in ("min_freq", "max_freq", "cur_freq"))
        state["gpu"] = {"min_hz": lo, "max_hz": hi, "cur_hz": cur,
                        "pinned": lo is not None and lo == hi}
    cpu = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    if cpu.exists():
        lo, hi, cur = (read_int(cpu / f) for f in
                       ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq"))
        governor = cpu / "scaling_governor"
        state["cpu0"] = {"min_khz": lo, "max_khz": hi, "cur_khz": cur,
                         "pinned": lo is not None and lo == hi,
                         "governor": governor.read_text().strip() if governor.exists() else None}
    pinned = [v["pinned"] for k, v in state.items() if k != "source"]
    state["jetson_clocks_active"] = all(pinned) if pinned else None
    return state


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

    # `nvpmodel -q` needs no root on JetPack 6. Older images want sudo.
    nvpmodel = sh(["nvpmodel", "-q"]) or sh(["sudo", "-n", "nvpmodel", "-q"])
    mode_name = mode_id = None
    if nvpmodel:
        for line in nvpmodel.splitlines():
            line = line.strip()
            if line.startswith("NV Power Mode:"):
                mode_name = line.split(":", 1)[1].strip()
            elif line.isdigit():
                mode_id = int(line)

    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "model": (model.read_bytes().decode("utf-8", "ignore").rstrip("\x00").strip()
                  if model.exists() else None),
        "l4t": l4t.read_text().splitlines()[0].strip() if l4t.exists() else None,
        "nvpmodel": nvpmodel,
        "nvpmodel_mode": mode_name,
        "nvpmodel_id": mode_id,
        "jetson_clocks_gpu": gpu_clock,
        "clocks": clock_state(),
        "python": platform.python_version(),
    }


class PowerSampler:
    """Samples board power, GPU clock and junction temperature in a thread.

    Reads the INA3221 rails straight from sysfs, so it needs neither root nor
    jtop. On a machine without the sensor, such as the host workstation, it
    finds no rails and reports nothing rather than failing.

    Each current read is an I2C transaction of roughly 0.7 ms, and the sensor
    updates about every 8 ms, so 50 ms between samples costs a few percent of
    one core. Rail voltages are read once, since all three sit on the
    regulated 5 V input.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.rails = self._find_rails()
        self.gpu = gpu_devfreq()
        self.tj = thermal_zone("tj-thermal")
        self.samples: list[list] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _find_rails() -> dict[str, tuple[int, Path]]:
        """Map rail label to (millivolts, path of its current in mA)."""
        for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            try:
                if (hwmon / "name").read_text().strip() != "ina3221":
                    continue
            except OSError:
                continue
            rails = {}
            for label_file in sorted(hwmon.glob("in*_label")):
                label = label_file.read_text().strip()
                n = label_file.name[2:-6]
                millivolts = read_int(hwmon / f"in{n}_input")
                current = hwmon / f"curr{n}_input"
                if label and millivolts and current.exists():
                    rails[label] = (millivolts, current)
            return rails
        return {}

    @property
    def available(self) -> bool:
        return bool(self.rails)

    def start(self) -> None:
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join()

    def _run(self) -> None:
        deadline = time.perf_counter()
        while not self._stop.is_set():
            row = [time.perf_counter()]
            for millivolts, current in self.rails.values():
                milliamps = read_int(current)
                row.append(None if milliamps is None else millivolts * milliamps / 1000.0)
            row.append(read_int(self.gpu / "cur_freq") if self.gpu else None)
            row.append(read_int(self.tj) if self.tj else None)
            self.samples.append(row)
            deadline += self.interval_s
            self._stop.wait(max(0.0, deadline - time.perf_counter()))

    def summary(self) -> dict | None:
        if not self.samples:
            return None
        rails = {}
        for i, label in enumerate(self.rails, start=1):
            mw = [row[i] for row in self.samples if row[i] is not None]
            if mw:
                rails[label] = {"mean_w": statistics.fmean(mw) / 1000.0,
                                "peak_w": max(mw) / 1000.0}
        gpu_hz = [row[-2] for row in self.samples if row[-2]]
        tj = [row[-1] for row in self.samples if row[-1] is not None]
        return {
            "source": "ina3221 sysfs",
            "interval_ms": self.interval_s * 1000.0,
            "n_samples": len(self.samples),
            "duration_s": self.samples[-1][0] - self.samples[0][0],
            "rail_voltage_mv": {label: mv for label, (mv, _) in self.rails.items()},
            "rails": rails,
            "gpu_freq_mhz": ({"min": min(gpu_hz) / 1e6, "max": max(gpu_hz) / 1e6}
                             if gpu_hz else None),
            "tj_temp_c": {"start": tj[0] / 1000.0, "max": max(tj) / 1000.0} if tj else None,
        }


def git_state() -> dict:
    def g(*args: str) -> str | None:
        return sh(["git", "-C", str(REPO_ROOT), *args])

    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        # A measurement taken with uncommitted changes cannot be reproduced.
        # Records already written to results/ are output, not code. Counting
        # them would mark every run after the first in a sweep as dirty.
        "dirty": bool(g("status", "--porcelain", "--", ".", ":(exclude)results")),
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


class TensorRTPolicy:
    """A PyTorch model exported to ONNX and built into a TensorRT engine here.

    Engines only run on the GPU and TensorRT version they were built with, so
    the build happens on the board, never on the workstation. The stand-in
    model has random weights, so the engine is rebuilt from the same model
    object on every run, and the actions can be compared one to one.

    Calling it runs the engine on the bound input tensors, the same way the
    harness calls a PyTorch model, so `measure` times both the same way.
    """

    INPUTS = ("image", "state")
    OUTPUT = "actions"

    def __init__(self, torch, model, inputs, name: str, workdir: Path = ENGINE_DIR):
        import tensorrt as trt

        workdir.mkdir(parents=True, exist_ok=True)
        self.onnx_path = workdir / f"{name}.onnx"
        self.engine_path = workdir / f"{name}.engine"

        start = time.perf_counter()
        # The TorchScript exporter. Torch 2.8 still defaults to it, and the
        # newer dynamo exporter needs onnxscript, which this env does not have.
        # Its "legacy exporter" notice is about torch 2.9 and says nothing here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "You are using the legacy TorchScript",
                                    DeprecationWarning)
            torch.onnx.export(model, inputs, str(self.onnx_path), dynamo=False, opset_version=17,
                              input_names=list(self.INPUTS), output_names=[self.OUTPUT])
        self.export_s = time.perf_counter() - start

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)  # explicit batch, the only mode in TensorRT 10
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(self.onnx_path)):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse {self.onnx_path}: {errors}")
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        start = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        self.build_s = time.perf_counter() - start
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed. The builder log above says why.")
        self.engine_path.write_bytes(bytes(serialized))
        self.engine_bytes = serialized.nbytes

        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        self.device_memory_bytes = (self.engine.device_memory_size_v2
                                    if hasattr(self.engine, "device_memory_size_v2")
                                    else self.engine.device_memory_size)
        # Not the default stream: TensorRT adds a sync on every call there.
        self.stream = torch.cuda.Stream()

        to_torch = {trt.float16: torch.float16, trt.float32: torch.float32}
        self.bound = {}
        for name, tensor in zip(self.INPUTS, inputs):
            want = to_torch[self.engine.get_tensor_dtype(name)]
            self.bound[name] = tensor.to(want).contiguous()
        out_dtype = to_torch[self.engine.get_tensor_dtype(self.OUTPUT)]
        out_shape = tuple(self.context.get_tensor_shape(self.OUTPUT))
        self.bound[self.OUTPUT] = torch.empty(out_shape, dtype=out_dtype, device="cuda")
        for name, tensor in self.bound.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

    def __call__(self, *_inputs):
        # The inputs were bound by address in __init__. The harness feeds the
        # same tensors every call, so there is nothing to copy here.
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.bound[self.OUTPUT]


def check_engine(torch, model, engine, inputs) -> dict:
    """Run PyTorch and the engine once on the same inputs and compare actions."""
    with torch.no_grad():
        want = model(*inputs).float()
        got = engine(*inputs)
        torch.cuda.synchronize()
        got = got.float()
    max_abs = (want - got).abs().max().item()
    scale = want.abs().max().item()
    return {"max_abs_diff": max_abs, "max_rel_diff": max_abs / scale if scale else None,
            "ok": bool(scale) and max_abs / scale <= TRT_MAX_REL_DIFF}


def measure(torch, model, inputs, iters: int, warmup: int,
            sampler: PowerSampler | None = None) -> list[float]:
    """Return per-iteration wall-clock latency in milliseconds.

    Warmup is discarded because the first calls include CUDA context creation,
    kernel autotuning, and memory-pool growth. Including them inflates p99 by
    hundreds of milliseconds and tells you nothing about steady-state control.
    The power sampler covers the timed loop only, for the same reason.
    """
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()

        if sampler:
            sampler.start()
        samples = []
        try:
            for _ in range(iters):
                start = time.perf_counter()
                model(*inputs)
                # Synchronize inside the loop. CUDA launches are asynchronous, so
                # timing without this measures queueing, not execution.
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
        finally:
            if sampler:
                sampler.stop()
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
    parser.add_argument("--runtime", choices=("pytorch", "tensorrt"), default="pytorch",
                        help="run the model in PyTorch eager, or as a TensorRT engine built here")
    parser.add_argument("--label", default=None,
                        help="short name for this configuration "
                             "(default: dummy-fp16, or dummy-trt-fp16 with --runtime tensorrt)")
    parser.add_argument("--config", default=None,
                        help="path to the YAML config this run came from")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the JSON record")
    parser.add_argument("--power-interval-ms", type=float, default=50.0,
                        help="board power sampling period during the timed loop")
    parser.add_argument("--no-power", action="store_true",
                        help="skip power sampling, e.g. to check it does not perturb latency")
    args = parser.parse_args()
    if args.label is None:
        args.label = "dummy-trt-fp16" if args.runtime == "tensorrt" else "dummy-fp16"

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

    # Before the run, so the record holds the state the run was taken in.
    board = board_state()
    sampler = None if args.no_power else PowerSampler(args.power_interval_ms / 1000.0)

    torch.cuda.reset_peak_memory_stats()

    model, inputs = build_dummy_model(torch, args.resolution, args.chunk)
    trt_info = None
    if args.runtime == "tensorrt":
        try:
            import tensorrt as trt
        except ImportError:
            print("[FAIL] tensorrt not importable. It is the system python3-libnvinfer "
                  "package, so the venv needs --system-site-packages. Never pip install it.",
                  file=sys.stderr)
            return 1
        engine = TensorRTPolicy(torch, model, inputs,
                                f"{args.label}-{args.resolution}px-chunk{args.chunk}")
        check = check_engine(torch, model, engine, inputs)
        trt_info = {
            "version": trt.__version__,
            "onnx": str(engine.onnx_path.relative_to(REPO_ROOT)),
            "engine": str(engine.engine_path.relative_to(REPO_ROOT)),
            "engine_bytes": engine.engine_bytes,
            "device_memory_bytes": engine.device_memory_bytes,
            "export_s": engine.export_s,
            "build_s": engine.build_s,
            "check_against_pytorch": check,
        }
        if not check["ok"]:
            print(f"[FAIL] the engine's actions differ from PyTorch's by "
                  f"{check['max_abs_diff']:.4g} ({check['max_rel_diff']:.2%} of the largest), "
                  f"more than {TRT_MAX_REL_DIFF:.0%}. Not timing a wrong engine.", file=sys.stderr)
            return 1
        model = engine
    samples = measure(torch, model, inputs, args.iters, args.warmup, sampler)
    stats = summarize(samples)
    power = sampler.summary() if sampler else None
    if power and "VDD_IN" in power["rails"]:
        # Whole-board energy, idle draw included, which is what a battery sees.
        power["energy_per_inference_j"] = (power["rails"]["VDD_IN"]["mean_w"]
                                           * stats["mean_ms"] / 1000.0)

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
            "runtime": args.runtime,
            "precision": "fp16",
            "iters": args.iters,
            "warmup": args.warmup,
            "target_hz": args.target_hz,
        },
        "latency": stats,
        # PyTorch's allocator only. A TensorRT engine allocates its own memory,
        # which is in tensorrt.engine_bytes and tensorrt.device_memory_bytes.
        "memory": {"peak_allocated_gib": peak_gb, "peak_reserved_gib": reserved_gb},
        "control_rate": {
            "sustained_hz_at_p50": sustained_hz,
            "worst_case_hz_at_p99": worst_hz,
            "meets_budget": meets_budget,
        },
        "power": power,
        "board": board,
        "git": git_state(),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
    }
    if trt_info:
        record["tensorrt"] = trt_info

    print("=" * 68)
    print(f"{args.label}   {args.resolution}px   chunk={args.chunk}   {args.runtime}")
    print("=" * 68)
    if trt_info:
        check = trt_info["check_against_pytorch"]
        print(f"  engine          TensorRT {trt_info['version']}, built in {trt_info['build_s']:.1f} s,"
              f" {trt_info['engine_bytes'] / 2**20:.1f} MiB, {trt_info['device_memory_bytes'] / 2**20:.1f}"
              f" MiB working memory")
        print(f"  vs PyTorch      max diff {check['max_abs_diff']:.4g}"
              f" ({check['max_rel_diff']:.2%} of the largest action)")
    print(f"  p50 {stats['p50_ms']:7.2f} ms    p90 {stats['p90_ms']:7.2f} ms"
          f"    p99 {stats['p99_ms']:7.2f} ms")
    print(f"  mean {stats['mean_ms']:6.2f} ms    sd  {stats['stdev_ms']:7.2f} ms"
          f"    max {stats['max_ms']:7.2f} ms")
    print(f"  peak gpu memory {peak_gb:.2f} GiB allocated, {reserved_gb:.2f} GiB reserved")
    print(f"  control rate    {sustained_hz:.1f} Hz typical, {worst_hz:.1f} Hz worst case")
    print(f"  budget {args.target_hz:.0f} Hz: {'MET' if meets_budget else 'MISSED'}")

    clocks = board["clocks"]
    pinned = {True: "on", False: "off"}.get(clocks["jetson_clocks_active"], "unknown")
    print(f"  power mode      {board['nvpmodel_mode'] or 'unknown'} (id {board['nvpmodel_id']}),"
          f" jetson_clocks {pinned}")
    if power and "VDD_IN" in power["rails"]:
        vdd_in = power["rails"]["VDD_IN"]
        print(f"  board power     {vdd_in['mean_w']:.2f} W mean, {vdd_in['peak_w']:.2f} W peak"
              f" (VDD_IN, {power['n_samples']} samples)")
        print(f"  energy          {power['energy_per_inference_j'] * 1000:.2f} mJ per inference")
    if power and power["gpu_freq_mhz"] and power["tj_temp_c"]:
        print(f"  gpu clock       {power['gpu_freq_mhz']['min']:.0f}-"
              f"{power['gpu_freq_mhz']['max']:.0f} MHz during run,"
              f" tj max {power['tj_temp_c']['max']:.1f} C")

    if sampler is not None and not sampler.available:
        print("\n[warn] no INA3221 rails found, so no power figures. Expected on the "
              "workstation, not on the Jetson.")
    elif power and power["n_samples"] < MIN_POWER_SAMPLES:
        print(f"\n[warn] power rests on only {power['n_samples']} samples. Raise --iters "
              f"so the timed loop runs for a few seconds.")
    gpu_clock = clocks.get("gpu", {})
    if (power and power["gpu_freq_mhz"] and gpu_clock.get("pinned")
            and power["gpu_freq_mhz"]["min"] < gpu_clock["max_hz"] / 1e6):
        print("\n[warn] the GPU clock fell below its pinned value during the run. "
              "Thermal or power throttling, so this run is not comparable to a cold one.")

    if board["nvpmodel_mode"] and (board["nvpmodel_mode"] != SWEEP_MODE
                                   or not clocks["jetson_clocks_active"]):
        print(f"\n[warn] not the sweep setting ({SWEEP_MODE}, jetson_clocks on). Fine for a "
              f"check or the power comparison, not for a sweep point.")

    if record["git"]["dirty"]:
        print("\n[warn] working tree is dirty. This run cannot be reproduced from a "
              "commit. Commit before recording anything that goes in the report.")

    out = args.out or (REPO_ROOT / "results" /
                       f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    try:
        shown = out.resolve().relative_to(REPO_ROOT)
    except ValueError:
        # --out may point outside the repo, e.g. a scratch directory.
        shown = out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
