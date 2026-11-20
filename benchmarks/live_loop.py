#!/usr/bin/env python3
"""Time the SmolVLA control loop with live cameras, without moving the arm.

This runs the same 30 Hz loop that lerobot-record runs when a policy drives
the follower. It reads both cameras with read_latest(), reads the joint state,
runs the policy and pops one action per step. It never sends an action to the
arm, so the arm does not move.

Two modes:

- sync: like lerobot-record. When the 50-action queue is empty, the step waits
  for a full chunk inference (about 0.9 s), so the arm would stop.
- async: a worker thread computes the next chunk while the loop keeps
  stepping. When the queue gets down to --trigger actions, the worker starts
  on the newest observation. When the chunk is ready, the actions for the
  steps that already passed are dropped, and the rest replace the queue.

It reports the step period, steps over the 33 ms budget, the longest stall,
chunk latency, and in async mode the queue gaps: steps where the queue was
empty, so the arm would hold its last action.

The joint state comes from the follower when --follower-port is set. It is
read only: the script never writes to the servos and never turns torque on
or off. Stop the teleop service first, because it holds the port. Without
--follower-port the state is zero.

    python3 benchmarks/live_loop.py --mode sync
    python3 benchmarks/live_loop.py --mode async --json results/live.json
    source ~/team_arm.env && python3 benchmarks/live_loop.py --mode async \\
        --follower-port "$FOLLOWER_PORT" --follower-id "$FOLLOWER_ID"

Run it in the CUDA env, with clocks pinned and no GPU apps open.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from hello_latency import PowerSampler, board_state, git_state, summarize  # noqa: E402

BRIO = "/dev/v4l/by-id/usb-046d_Logitech_BRIO_72580AD9-video-index0"
WRIST = "/dev/v4l/by-id/usb-Generic_USB_HD_Camera_200910120001-video-index0"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class StateReader:
    """Reads the follower's joint positions. Never writes to the bus."""

    def __init__(self, port: str | None, robot_id: str):
        self.robot = None
        if port is None:
            return
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
        if not self.robot.calibration:
            raise SystemExit(f"No calibration file for {robot_id}. Calibrate with hardware/so101_lab.sh.")
        # Only open the bus. robot.connect() would also write settings and
        # turn torque on, and disconnecting would turn torque off.
        self.robot.bus.connect()

    def read(self):
        import numpy as np

        if self.robot is None:
            return np.zeros(len(JOINTS), dtype=np.float32)
        pos = self.robot.bus.sync_read("Present_Position")
        return np.array([pos[j] for j in JOINTS], dtype=np.float32)

    def close(self) -> None:
        if self.robot is not None:
            self.robot.bus.disconnect(disable_torque=False)


class ChunkWorker:
    """Runs one chunk inference at a time in a background thread."""

    def __init__(self, infer_chunk):
        self.infer_chunk = infer_chunk
        self.lock = threading.Lock()
        self.job = None  # (obs, tick)
        self.result = None  # (actions, tick, latency_ms)
        self.busy = False
        self.wake = threading.Event()
        self.stop_flag = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, obs, tick: int) -> None:
        with self.lock:
            self.job = (obs, tick)
            self.busy = True
        self.wake.set()

    def take(self):
        with self.lock:
            result, self.result = self.result, None
            return result

    def _run(self) -> None:
        while not self.stop_flag:
            self.wake.wait(0.1)
            self.wake.clear()
            with self.lock:
                job, self.job = self.job, None
            if job is None:
                continue
            obs, tick = job
            t = time.perf_counter()
            try:
                actions = self.infer_chunk(obs)
            except Exception as e:  # report it from the main loop
                self.error = e
                self.stop_flag = True
                return
            latency_ms = (time.perf_counter() - t) * 1e3
            with self.lock:
                self.result = (actions, tick, latency_ms)
                self.busy = False

    def close(self) -> None:
        self.stop_flag = True
        self.wake.set()
        self.thread.join(timeout=5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("sync", "async"), default="sync")
    ap.add_argument("--policy", default="lerobot/smolvla_base")
    ap.add_argument("--cam1", default=BRIO, help="front camera (default: our Brio)")
    ap.add_argument("--cam2", default=WRIST, help="wrist camera (default: the kit camera)")
    ap.add_argument("--cam1-fps", type=int, default=60, help="Brio rate, in MJPG (default: %(default)s)")
    ap.add_argument("--cam2-fps", type=int, default=30, help="wrist camera rate (default: %(default)s)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hz", type=float, default=30.0, help="control rate (default: %(default)s)")
    ap.add_argument("--seconds", type=float, default=30.0, help="timed loop length (default: %(default)s)")
    ap.add_argument("--steps", type=int, help="denoising steps (default: the checkpoint's, 10)")
    ap.add_argument("--trigger", type=int,
                    help="async: start the next chunk when this many actions are left "
                         "(default: enough for the measured p99 latency plus 25%%)")
    ap.add_argument("--follower-port", help="read the real joint state from this port (read only)")
    ap.add_argument("--follower-id", default="team_follower")
    ap.add_argument("--task", default="Pick up the cube and put it in the box.")
    ap.add_argument("--no-power", action="store_true", help="skip board power sampling")
    ap.add_argument("--json", type=Path, metavar="PATH", help="also write the report to this file")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available. Run this in the CUDA env.")
        return 1

    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import predict_action, prepare_observation_for_inference

    device = torch.device("cuda")
    policy = SmolVLAPolicy.from_pretrained(args.policy)
    policy.to(device).eval()
    if args.steps:
        policy.config.num_steps = args.steps
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": device.type}})
    image_keys = list(policy.config.image_features)[:2]
    n_actions = policy.config.n_action_steps

    cams = [
        OpenCVCamera(OpenCVCameraConfig(index_or_path=args.cam1, width=args.width, height=args.height,
                                        fps=args.cam1_fps, fourcc="MJPG", warmup_s=3)),
        OpenCVCamera(OpenCVCameraConfig(index_or_path=args.cam2, width=args.width, height=args.height,
                                        fps=args.cam2_fps, warmup_s=3)),
    ]
    for cam in cams:
        cam.connect()
    state = StateReader(args.follower_port, args.follower_id)

    obs_ms: list[float] = []

    def observation() -> dict:
        t = time.perf_counter()
        obs = {"observation.state": state.read()}
        for key, cam in zip(image_keys, cams):
            obs[key] = cam.read_latest()
        obs_ms.append((time.perf_counter() - t) * 1e3)
        return obs

    def infer_step(obs: dict):
        # What lerobot-record calls: select_action() pops the queue, and runs
        # a full chunk inference when the queue is empty.
        return predict_action(obs, policy, device, pre, post, use_amp=policy.config.use_amp,
                              task=args.task, robot_type="so101_follower")

    def infer_chunk(obs: dict):
        # The same pipeline as predict_action(), but returns all actions.
        with torch.inference_mode():
            batch = prepare_observation_for_inference(dict(obs), device, args.task, "so101_follower")
            batch = pre(batch)
            actions = policy.predict_action_chunk(batch)[0, :n_actions]
            return post(actions).cpu()

    # Warm up both paths and measure the chunk latency for the async trigger.
    for _ in range(3):
        policy.reset()
        infer_step(observation())
    warm = []
    for _ in range(5):
        t = time.perf_counter()
        infer_chunk(observation())
        warm.append((time.perf_counter() - t) * 1e3)
    warm_p99 = max(warm)
    period = 1.0 / args.hz
    trigger = args.trigger or min(n_actions - 1, math.ceil(1.25 * warm_p99 / 1000 / period))
    obs_ms.clear()

    sampler = None if args.no_power else PowerSampler()
    if sampler:
        sampler.start()

    step_ms: list[float] = []  # work time inside each step
    period_ms: list[float] = []  # start-to-start time between steps
    chunk_ms: list[float] = []
    gaps = 0
    longest_gap = 0
    gap_run = 0
    dropped = 0
    finite = True
    n_steps = int(args.seconds * args.hz)
    queue: deque = deque()
    worker = ChunkWorker(infer_chunk) if args.mode == "async" else None
    policy.reset()
    if worker:
        # Start with a full queue, like the arm after its first chunk. Then
        # the gaps below come from the loop itself, not from the start.
        queue = deque(infer_chunk(observation()))
        obs_ms.clear()
    last_start = None
    next_tick = time.perf_counter()

    try:
        for tick in range(n_steps):
            now = time.perf_counter()
            if last_start is not None:
                period_ms.append((now - last_start) * 1e3)
            last_start = now
            obs = observation()

            if args.mode == "sync":
                was_empty = len(policy._queues["action"]) == 0
                t = time.perf_counter()
                action = infer_step(obs)
                if was_empty:
                    chunk_ms.append((time.perf_counter() - t) * 1e3)
                finite &= bool(torch.isfinite(action).all())
            else:
                if worker.error:
                    raise worker.error
                result = worker.take()
                if result is not None:
                    actions, start_tick, latency = result
                    chunk_ms.append(latency)
                    finite &= bool(torch.isfinite(actions).all())
                    # These steps passed while the chunk was computed.
                    late = tick - start_tick
                    dropped += min(late, len(actions))
                    queue = deque(actions[late:])
                if len(queue) <= trigger and not worker.busy:
                    worker.submit(obs, tick)
                if queue:
                    queue.popleft()
                    longest_gap = max(longest_gap, gap_run)
                    gap_run = 0
                else:
                    gaps += 1  # the arm would hold its last action
                    gap_run += 1

            step_ms.append((time.perf_counter() - now) * 1e3)
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()  # late: do not try to catch up
    finally:
        if worker:
            worker.close()
        if sampler:
            sampler.stop()
        for cam in cams:
            cam.disconnect()
        state.close()

    longest_gap = max(longest_gap, gap_run)
    power = sampler.summary() if sampler else None
    budget_ms = period * 1e3
    over = sum(1 for x in period_ms if x > budget_ms * 1.5)
    step, per, chunk, obs_t = (summarize(step_ms), summarize(period_ms),
                               summarize(chunk_ms) if chunk_ms else None, summarize(obs_ms))
    board = board_state()
    real_rate = len(period_ms) / (sum(period_ms) / 1e3) if period_ms else 0.0
    stall_ms = max(period_ms) if period_ms else 0.0

    print("=" * 72)
    print(f"SmolVLA live loop, {args.mode}, {args.hz:.0f} Hz for {args.seconds:.0f} s, "
          f"{policy.config.num_steps} denoising steps")
    print(f"cameras: Brio {args.cam1_fps} fps MJPG, wrist {args.cam2_fps} fps, "
          f"state from {'the follower (read only)' if args.follower_port else 'zeros'}")
    print("=" * 72)
    print(f"  loop rate        {real_rate:.1f} Hz over {len(step_ms)} steps")
    print(f"  step period      p50 {per['p50_ms']:6.1f} ms   p99 {per['p99_ms']:6.1f} ms   "
          f"max {per['max_ms']:6.1f} ms   (budget {budget_ms:.1f} ms)")
    print(f"  late steps       {over} of {len(period_ms)} took over 1.5x the budget")
    print(f"  longest stall    {stall_ms:.0f} ms, about {stall_ms / budget_ms:.0f} steps without a new action")
    print(f"  observation      p50 {obs_t['p50_ms']:5.1f} ms   p99 {obs_t['p99_ms']:5.1f} ms   "
          f"(cameras + joint state)")
    if chunk:
        print(f"  chunk latency    p50 {chunk['p50_ms']:6.1f} ms   p99 {chunk['p99_ms']:6.1f} ms   "
              f"(n={chunk['n']})")
    if args.mode == "async":
        print(f"  trigger          next chunk at {trigger} actions left "
              f"(warm-up chunk max {warm_p99:.0f} ms)")
        print(f"  queue gaps       {gaps} empty steps, longest run {longest_gap} "
              f"(the queue starts full)")
        print(f"  dropped actions  {dropped}, for steps that passed during inference")
    if power and power.get("rails", {}).get("VDD_IN"):
        vin = power["rails"]["VDD_IN"]
        print(f"  board power      VDD_IN mean {vin['mean_w']:.2f} W, peak {vin['peak_w']:.2f} W")
    clocks = board["clocks"].get("jetson_clocks_active")
    print(f"  board            {board['nvpmodel_mode']}, clocks {'pinned' if clocks else 'scaling'}")
    print("  OK" if finite else "[FAIL] the policy returned NaN or inf")

    if args.json:
        report = {
            "kind": "smolvla_live_loop",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "policy": {"path": args.policy, "num_steps": policy.config.num_steps,
                       "n_action_steps": n_actions, "image_keys": image_keys},
            "trigger": trigger if args.mode == "async" else None,
            "loop_rate_hz": real_rate,
            "latency": {"step_work": step, "step_period": per, "chunk": chunk, "observation": obs_t},
            "late_steps": over,
            "longest_stall_ms": stall_ms,
            "queue_gaps": gaps if args.mode == "async" else None,
            "longest_gap_steps": longest_gap if args.mode == "async" else None,
            "dropped_actions": dropped if args.mode == "async" else None,
            "power": power,
            "finite": finite,
            "board": board,
            "git": git_state(),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
