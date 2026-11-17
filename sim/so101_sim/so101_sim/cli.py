"""Command line: python -m so101_sim <command> [options]

  app        the interactive simulator in your browser (teleop, record, watch a policy)
  generate   record N scripted-expert demos into a LeRobot dataset, headless
  eval       closed-loop success rate of a policy, with the Jetson's latency injected
  check      verify this computer can run the simulator (and write a test image)

Run any command with -h for its options.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from .config import HOME_POSE, RANDOMIZE_LEVELS, SimConfig


def _floats(text: str, n: int) -> tuple[float, ...]:
    vals = tuple(float(v) for v in text.replace(" ", "").split(","))
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got {text!r}")
    return vals


def add_sim_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("simulation")
    g.add_argument("--randomize", choices=RANDOMIZE_LEVELS, default="light",
                   help="none: one fixed layout; light: cube and box positions (default); "
                        "full: also colours, lights, camera1 pose, cube mass and friction")
    g.add_argument("--fps", type=int, default=30, help="control rate, Hz (default 30, as in so101_lab.sh)")
    g.add_argument("--episode-s", type=float, default=20.0, help="episode time limit, s (default 20)")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--cameras", default="camera1,camera2",
                   help="cameras to render and record: camera1 (front), camera2 (wrist)")
    g.add_argument("--camera1-pos", type=lambda s: _floats(s, 3), metavar="X,Y,Z",
                   help="front camera position, m (default 0.62,0,0.42)")
    g.add_argument("--camera1-target", type=lambda s: _floats(s, 3), metavar="X,Y,Z",
                   help="point the front camera looks at, m (default 0.20,0,0)")
    g.add_argument("--max-step-deg", type=float, help="cap joint moves per step like MAX_STEP (default off)")
    g.add_argument("--joint-map", help="JSON with per-joint sign/offset if your real arm disagrees with the sim")
    g.add_argument("--home", type=lambda s: _floats(s, 6), metavar="6 VALUES",
                   help=f"start pose in LeRobot units (default {','.join(f'{v:g}' for v in HOME_POSE)})")


def sim_config(a: argparse.Namespace) -> SimConfig:
    kw = dict(fps=a.fps, episode_s=a.episode_s, width=a.width, height=a.height, randomize=a.randomize,
              cameras=tuple(c.strip() for c in a.cameras.split(",") if c.strip()),
              max_step_deg=a.max_step_deg)
    if a.camera1_pos:
        kw["camera1_pos"] = a.camera1_pos
    if a.camera1_target:
        kw["camera1_target"] = a.camera1_target
    if a.home:
        kw["home_pose"] = a.home
    return SimConfig(**kw)


def default_results_dir() -> Path:
    """The repo's results/ folder when the simulator sits in its sim/ folder, else ./sim_results."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "results").is_dir() and (parent / "sim").is_dir():
            return parent / "results"
    return Path("sim_results")


def policy_label(spec: str) -> str:
    """Short name for a policy in result file names.

    outputs/train/smolvla_sim/checkpoints/020000/pretrained_model -> smolvla_sim-020000
    lerobot/smolvla_base (a hub id) -> smolvla_base;  expert -> expert
    """
    parts = [x for x in spec.replace("\\", "/").rstrip("/").split("/") if x not in ("", ".")]
    if not parts:
        return "policy"
    if parts[-1] == "pretrained_model":
        parts = parts[:-1]
    if len(parts) >= 3 and parts[-2] == "checkpoints":
        label = f"{parts[-3]}-{parts[-1]}"
    else:
        label = parts[-1] if parts else "policy"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label) or "policy"


def joint_map(a):
    from .units import JointMap
    return JointMap.load(a.joint_map)


# ----------------------------------------------------------------------------- commands


def cmd_app(a) -> int:
    from .app.server import SimApp, serve
    app = SimApp(sim_config(a), joint_map=joint_map(a), dataset=a.dataset, root=a.root, task=a.task,
                 seed=a.seed, leader_port=a.leader_port, leader_id=a.leader_id, policy=a.policy,
                 latency_ms=a.latency_ms, policy_mode=a.mode, threshold=a.threshold,
                 num_steps=a.num_steps, ui_hz=a.ui_hz, view_size=a.view_size)
    serve(app, host=a.host, port=a.port, open_browser=not a.no_browser)
    return 0


def cmd_generate(a) -> int:
    from .generate import generate
    cfg = sim_config(a)
    if a.task:
        from dataclasses import replace
        cfg = cfg.with_(task=replace(cfg.task, instruction=a.task))
    summary = generate(a.name, a.episodes, cfg, seed=a.seed, root=a.root, resume=a.resume,
                       overwrite=a.overwrite, keep_failures=a.keep_failures)
    return 0 if summary["saved"] == a.episodes else 1


def cmd_eval(a) -> int:
    from .policy_eval import LatencyModel, evaluate
    cfg = sim_config(a)
    if a.latency_from:
        latencies = [LatencyModel.from_result(a.latency_from, jitter=not a.no_jitter)]
    else:
        latencies = [LatencyModel.constant(float(v)) for v in str(a.latency_ms).split(",")]
    out_dir = Path(a.out_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = policy_label(a.policy)
    rows = []
    for lat in latencies:
        out = out_dir / f"{stamp}_sim-eval_{name}_{a.mode}_{lat.mean_ms:.0f}ms.json" if not a.no_save else None
        rec = evaluate(a.policy, cfg, a.episodes, lat, mode=a.mode, threshold=a.threshold, seed=a.seed,
                       device=a.device, num_steps=a.num_steps, n_action_steps=a.n_action_steps, out=out)
        rows.append((lat.mean_ms, rec))
    if len(rows) > 1:
        print("\nlatency_ms  success  wilson95        staleness_mean_ms  stall_fraction")
        for ms, rec in rows:
            s = rec["success"]
            print(f"{ms:10.0f}  {s['k']:3d}/{s['n']:<3d}  {s['wilson95'][0]:.2f}-{s['wilson95'][1]:.2f}  "
                  f"{rec['staleness_ms']['mean']:17.0f}  {rec['stall_fraction']:14.3f}")
    return 0


class _Skip(Exception):
    pass


def cmd_check(a) -> int:
    import numpy as np
    ok = True
    print(f"python   {platform.python_version()} on {platform.system()} {platform.machine()}")
    try:
        import mujoco
        print(f"mujoco   {mujoco.__version__}   (MUJOCO_GL={os.environ.get('MUJOCO_GL', 'default')})")
    except ImportError:
        print("mujoco   MISSING: pip install mujoco")
        return 1
    try:
        import lerobot
        print(f"lerobot  {lerobot.__version__}   recording and policy eval available")
    except ImportError:
        print("lerobot  not installed: the app runs, but recording and policy eval are off")
    from .env import SO101Env
    from .expert import run_expert_episode
    cfg = sim_config(a)
    env = SO101Env(cfg, joint_map=joint_map(a), render=not a.no_render)
    t = time.perf_counter()
    try:
        if a.no_render:
            raise _Skip()
        obs = env.reset(seed=0)
        render_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        for _ in range(10):
            env.render_camera("camera1")
        render_ms = (time.perf_counter() - t) * 100
        print(f"render   OK, {render_ms:.1f} ms per {cfg.width}x{cfg.height} frame")
        from PIL import Image
        img = np.concatenate([obs.images[k] for k in cfg.cameras], axis=1)
        out = Path(a.image)
        Image.fromarray(img).save(out)
        print(f"         wrote {out.resolve()}")
    except _Skip:
        print("render   skipped (--no-render)")
    except Exception as e:
        ok = False
        print(f"render   FAILED: {e}")
        print("         On Linux without a display, try MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa.")
    t = time.perf_counter()
    n_ok = 0
    for seed in range(a.episodes):
        for _ in run_expert_episode(env, seed):
            pass
        n_ok += env.is_success()
    dt = (time.perf_counter() - t) / max(a.episodes, 1)
    print(f"physics  expert succeeded {n_ok}/{a.episodes} episodes, {dt:.2f} s each without rendering")
    ok &= n_ok >= a.episodes - 1
    env.close()
    print("all good" if ok else "something is wrong, see above")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="so101_sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("app", help="interactive simulator in the browser")
    add_sim_args(s)
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open the page from another computer")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    s.add_argument("--ui-hz", type=float, default=30.0, help="how often the page's images update")
    s.add_argument("--view-size", type=lambda v: tuple(int(x) for x in v.lower().split("x")), default=(960, 720),
                   metavar="WxH", help="resolution of the 3D view in the page (default 960x720)")
    s.add_argument("--seed", type=int, default=0, help="layout of the first scene")
    s.add_argument("--dataset", help="dataset name for recording (default sim_cube_box)")
    s.add_argument("--root", help="dataset folder (default $HF_LEROBOT_HOME/local/<name>)")
    s.add_argument("--task", help="instruction stored with every frame")
    s.add_argument("--leader-port", help="serial port of a real SO-101 leader arm")
    s.add_argument("--leader-id", default="my_leader", help="its LeRobot calibration id")
    s.add_argument("--policy", help="start driving with this checkpoint (or 'expert')")
    s.add_argument("--latency-ms", type=float, default=0.0, help="injected chunk latency for --policy")
    s.add_argument("--mode", choices=("async", "sync"), default="async")
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--num-steps", type=int, help="flow-matching denoising steps (SmolVLA, pi0)")
    s.set_defaults(func=cmd_app)

    s = sub.add_parser("generate", help="record scripted-expert demos, headless")
    add_sim_args(s)
    s.add_argument("--name", required=True, help="dataset name; saved as local/<name>")
    s.add_argument("--episodes", type=int, required=True)
    s.add_argument("--seed", type=int, default=0, help="first layout seed (default 0)")
    s.add_argument("--root", help="dataset folder (default $HF_LEROBOT_HOME/local/<name>)")
    s.add_argument("--task", help="instruction stored with every frame")
    s.add_argument("--resume", action="store_true", help="add episodes to an existing dataset")
    s.add_argument("--overwrite", action="store_true", help="delete an existing dataset of that name first")
    s.add_argument("--keep-failures", action="store_true", help="also save episodes the expert failed")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("eval", help="success rate of a policy under injected latency")
    add_sim_args(s)
    s.add_argument("--policy", required=True, help="checkpoint folder (.../pretrained_model), hub id, or 'expert'")
    s.add_argument("--episodes", type=int, default=50)
    lat = s.add_mutually_exclusive_group()
    lat.add_argument("--latency-ms", default="0",
                     help="chunk latency on the Jetson, ms; a comma list runs a sweep (e.g. 0,250,500,1000)")
    lat.add_argument("--latency-from", help="harness record in results/ to take latency.chunk p50/stdev from")
    s.add_argument("--no-jitter", action="store_true", help="with --latency-from, use p50 without stdev")
    s.add_argument("--mode", choices=("async", "sync"), default="async")
    s.add_argument("--threshold", type=float, default=0.5,
                   help="async: request the next chunk when this fraction of a chunk is left")
    s.add_argument("--seed", type=int, default=100_000, help="first layout seed (default 100000)")
    s.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    s.add_argument("--num-steps", type=int, help="flow-matching denoising steps (SmolVLA, pi0)")
    s.add_argument("--n-action-steps", type=int, help="actions used per chunk (default: the checkpoint's)")
    s.add_argument("--out-dir", default=str(default_results_dir()),
                   help="where JSON records go (default: the repo's results/ folder)")
    s.add_argument("--no-save", action="store_true")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("check", help="test this computer")
    add_sim_args(s)
    s.add_argument("--episodes", type=int, default=3)
    s.add_argument("--image", default="so101_sim_check.png")
    s.add_argument("--no-render", action="store_true", help="skip the rendering test (machines without OpenGL)")
    s.set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
