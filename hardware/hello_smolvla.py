#!/usr/bin/env python3
"""SmolVLA bring-up check: does the policy load and run on the Jetson GPU?

Loads a SmolVLA checkpoint, lerobot/smolvla_base by default, with the same
LeRobot calls lerobot-record makes on the arm, and times what the control loop
will pay for:

- chunk inference. SmolVLA plans 50 actions per call. When the queue runs dry,
  lerobot-record stops and waits this long, once per chunk. At 30 Hz a chunk
  lasts about 1.7 s.
- the other steps. They still run the preprocessor and pop one queued action,
  and that has to fit in the 33 ms step next to the camera reads and the
  servo bus.
- memory: GPU peak, and how much of the board's shared RAM is left
- board power, GPU clock and temperature during the timed calls

No arm is needed. By default the images are random noise and the joint state
is zero, which is enough to time the model and nothing more: the actions it
prints are not meaningful. --camera feeds real frames into an image slot.

    python3 hardware/hello_smolvla.py
    python3 hardware/hello_smolvla.py --cameras 3 --steps 5
    python3 hardware/hello_smolvla.py --camera /dev/v4l/by-id/usb-...-video-index0
    HF_HUB_OFFLINE=1 python3 hardware/hello_smolvla.py   # proves the cache is complete

Run it in the CUDA env. The first run downloads about 2.9 GB into
~/.cache/huggingface: the SmolVLA weights and the SmolVLM2 backbone they are
built on. Run it once with HF_HUB_OFFLINE=1 before taking the board somewhere
without a network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
from hello_latency import (MIN_POWER_SAMPLES, PowerSampler, board_state,  # noqa: E402
                           git_state, summarize)

CONTROL_HZ = 30


def mem_available_gib() -> float | None:
    """MemAvailable from /proc/meminfo. CPU and GPU share it on a Jetson."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def open_camera(device: str, width: int, height: int):
    import cv2

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {device}. Check it with hardware/hello_camera.py --list.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def read_rgb(cap):
    import cv2

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("The camera returned no frame.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def params_by_dtype(policy) -> dict[str, float]:
    counts: dict[str, int] = {}
    for p in policy.parameters():
        key = str(p.dtype).replace("torch.", "")
        counts[key] = counts.get(key, 0) + p.numel()
    return {k: v / 1e6 for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lerobot/smolvla_base",
                    help="hub id or local checkpoint folder (default: %(default)s)")
    ap.add_argument("--cameras", type=int, default=2, choices=(1, 2, 3),
                    help="image inputs to feed. The base model takes up to 3 (default: %(default)s)")
    ap.add_argument("--camera", action="append", default=[], metavar="DEVICE",
                    help="real camera for the next image slot, in order. Repeat for more")
    ap.add_argument("--width", type=int, default=640, help="frame width (default: %(default)s)")
    ap.add_argument("--height", type=int, default=480, help="frame height (default: %(default)s)")
    ap.add_argument("--steps", type=int,
                    help="flow-matching denoising steps (default: the checkpoint's, 10 for the base model)")
    ap.add_argument("--iters", type=int, default=20, help="timed chunk inferences (default: %(default)s)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed calls first (default: %(default)s)")
    ap.add_argument("--task", default="Pick up the cube and put it in the box.",
                    help="language instruction passed to the policy")
    ap.add_argument("--no-power", action="store_true", help="skip board power sampling")
    ap.add_argument("--json", type=Path, metavar="PATH", help="also write the report to this file")
    args = ap.parse_args()

    if len(args.camera) > args.cameras:
        ap.error(f"{len(args.camera)} --camera devices but only --cameras {args.cameras} image slots")

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available. Run this in the CUDA env (vla), with cudss-lib on LD_LIBRARY_PATH.")
        return 1

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action

    device = torch.device("cuda")
    ram_before = mem_available_gib()
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(args.policy)
    policy.to(device).eval()
    if args.steps:
        policy.config.num_steps = args.steps
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": device.type}})
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    ram_loaded = mem_available_gib()
    gpu_loaded = torch.cuda.memory_allocated() / 2**30

    image_keys = list(policy.config.image_features)[: args.cameras]
    state_dim = policy.config.robot_state_feature.shape[0]
    caps = [open_camera(d, args.width, args.height) for d in args.camera]
    rng = np.random.default_rng(0)

    def observation() -> dict:
        obs = {"observation.state": np.zeros(state_dim, dtype=np.float32)}
        for i, key in enumerate(image_keys):
            obs[key] = (read_rgb(caps[i]) if i < len(caps) else
                        rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8))
        return obs

    def infer(obs: dict):
        # Returns on the CPU, which waits for the GPU, so wall time is honest.
        return predict_action(obs, policy, device, preprocessor, postprocessor,
                              use_amp=policy.config.use_amp, task=args.task,
                              robot_type="so101_follower")

    # Chunk inference: an empty queue forces the full forward pass each call.
    for _ in range(args.warmup):
        policy.reset()
        infer(observation())
    torch.cuda.reset_peak_memory_stats()
    sampler = None if args.no_power else PowerSampler()
    if sampler:
        sampler.start()
    chunk_ms = []
    action = None
    for _ in range(args.iters):
        obs = observation()
        policy.reset()
        t = time.perf_counter()
        action = infer(obs)
        chunk_ms.append((time.perf_counter() - t) * 1e3)
    if sampler:
        sampler.stop()
    power = sampler.summary() if sampler else None

    # Steps served from the queue, the other 49 of every 50.
    policy.reset()
    infer(observation())
    obs = observation()
    step_ms = []
    for _ in range(policy.config.n_action_steps - 1):
        t = time.perf_counter()
        infer(obs)
        step_ms.append((time.perf_counter() - t) * 1e3)

    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    ram_after = mem_available_gib()
    first_action = [round(float(x), 3) for x in action.reshape(-1).tolist()]
    finite = bool(torch.isfinite(action).all())
    chunk, step = summarize(chunk_ms), summarize(step_ms)
    chunk_s = policy.config.n_action_steps / CONTROL_HZ
    board = board_state()

    source = f"{len(caps)} real + {len(image_keys) - len(caps)} synthetic" if caps else "synthetic"
    print("=" * 68)
    print(f"SmolVLA {args.policy}")
    print(f"{len(image_keys)} image(s) {args.width}x{args.height} ({source}), state {state_dim}, "
          f"chunk {policy.config.chunk_size}, {policy.config.num_steps} denoising steps")
    print("=" * 68)
    params = params_by_dtype(policy)
    print(f"  loaded in {load_s:.1f} s, {sum(params.values()):.0f} M params ("
          + ", ".join(f"{k} {v:.0f} M" for k, v in params.items()) + ")")
    print(f"  chunk inference  p50 {chunk['p50_ms']:7.1f} ms   p90 {chunk['p90_ms']:7.1f} ms   "
          f"p99 {chunk['p99_ms']:7.1f} ms   max {chunk['max_ms']:7.1f} ms   (n={chunk['n']})")
    print(f"  queued step      p50 {step['p50_ms']:7.1f} ms   p99 {step['p99_ms']:7.1f} ms   "
          f"of a {1000 / CONTROL_HZ:.0f} ms step at {CONTROL_HZ} Hz")
    print(f"  gpu memory       {gpu_loaded:.2f} GiB after load, peak {peak_alloc:.2f} GiB allocated, "
          f"{peak_reserved:.2f} GiB reserved")
    if ram_before is not None and ram_after is not None:
        print(f"  board ram free   {ram_before:.2f} GiB before load, {ram_loaded:.2f} after, "
              f"{ram_after:.2f} after the runs")
    if power and power.get("rails", {}).get("VDD_IN"):
        vin = power["rails"]["VDD_IN"]
        gpu = power.get("gpu_freq_mhz") or {}
        tj = (power.get("tj_temp_c") or {}).get("max")
        print(f"  board power      VDD_IN mean {vin['mean_w']:.2f} W, peak {vin['peak_w']:.2f} W, "
              f"{vin['mean_w'] * chunk['mean_ms'] / 1000:.2f} J per chunk")
        if gpu:
            print(f"  gpu clock        {gpu['min']:.0f}-{gpu['max']:.0f} MHz, tj max {tj:.1f} C")
        if power["n_samples"] < MIN_POWER_SAMPLES:
            print(f"[warn] only {power['n_samples']} power samples. Raise --iters for a power number.")
    clocks = board["clocks"].get("jetson_clocks_active")
    print(f"  board            {board['nvpmodel_mode']} (mode {board['nvpmodel_id']}), "
          f"clocks {'pinned' if clocks else 'scaling'}")
    print(f"  at {CONTROL_HZ} Hz a {policy.config.n_action_steps}-action chunk lasts {chunk_s * 1000:.0f} ms, "
          f"and the loop waits about {chunk['p99_ms']:.0f} ms (p99) for the next one")
    print(f"  first action     {first_action}")
    if not finite:
        print("[FAIL] the policy returned NaN or inf")
    elif chunk["p99_ms"] > chunk_s * 1000:
        print(f"[warn] p99 chunk inference is longer than a chunk lasts at {CONTROL_HZ} Hz")
    else:
        print("  OK")

    if args.json:
        report = {
            "kind": "smolvla_bringup",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "policy": {"path": args.policy, "num_steps": policy.config.num_steps,
                       "chunk_size": policy.config.chunk_size,
                       "n_action_steps": policy.config.n_action_steps,
                       "image_keys": image_keys, "params_m": params, "load_s": load_s},
            "latency": {"chunk": chunk, "queued_step": step},
            "memory": {"gpu_after_load_gib": gpu_loaded, "gpu_peak_allocated_gib": peak_alloc,
                       "gpu_peak_reserved_gib": peak_reserved, "ram_available_gib":
                       {"before_load": ram_before, "after_load": ram_loaded, "after_runs": ram_after}},
            "power": power,
            "first_action": first_action,
            "finite": finite,
            "board": board,
            "git": git_state(),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    for cap in caps:
        cap.release()
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
