"""Headless demo generation with the scripted expert.

    python -m so101_sim generate --name sim_cube_box --episodes 200

Only successful episodes are kept unless --keep-failures is given, so the dataset
is clean imitation data. Seeds are consecutive from --seed, and every episode's
seed and outcome are written to <dataset>/meta/so101_sim_episodes.jsonl so that a
layout can be reproduced later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import SimConfig
from .env import SO101Env
from .expert import ScriptedExpert
from .recorder import Recorder


def generate(name: str, episodes: int, cfg: SimConfig, seed: int = 0, root: str | None = None,
             resume: bool = False, overwrite: bool = False, keep_failures: bool = False,
             tail_s: float = 0.5, max_attempts_factor: float = 2.0, log=print) -> dict:
    env = SO101Env(cfg, render=True)
    rec = Recorder(name, cfg, env.task, root=root, resume=resume, overwrite=overwrite)
    log_path = Path(rec.root) / "meta" / "so101_sim_episodes.jsonl"
    saved = attempts = 0
    t0 = time.perf_counter()
    s = seed
    try:
        while saved < episodes and attempts < max(episodes * max_attempts_factor, episodes + 5):
            rng = np.random.default_rng(s)
            obs = env.reset(seed=s)
            expert = ScriptedExpert(env, rng)
            n = expert.total_steps + int(tail_s * cfg.fps)
            for _ in range(n):
                action = expert.act()
                rec.add(obs, action, env.task)
                obs = env.step(action)
            ok = env.is_success()
            attempts += 1
            if ok or keep_failures:
                idx = rec.save_episode() - 1
                saved += 1
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as f:
                    entry = {"episode_index": idx, "seed": s, "success": bool(ok), "frames": n,
                             "randomize": cfg.randomize, "expert_speed": expert.speed,
                             "start": [round(float(v), 2) for v in env.jmap.to_lerobot(env.start)]}
                    if not env.original_cube or env.noise is not None:
                        entry.update(env.episode_info())
                    f.write(json.dumps(entry) + "\n")
                what = "" if env.original_cube else f"  {env.task}"
                log(f"episode {idx:4d}  seed {s:6d}  {'success' if ok else 'FAILED '}  {n} frames  "
                    f"[{saved}/{episodes}, {time.perf_counter() - t0:.0f}s]{what}")
            else:
                rec.discard()
                log(f"seed {s:6d}: expert failed, discarded")
            s += 1
    finally:
        rec.finalize()
        env.close()
    summary = {"dataset": str(rec.root), "repo_id": rec.repo_id, "saved": saved, "attempts": attempts,
               "expert_success_rate": saved / max(attempts, 1) if not keep_failures else None,
               "seconds": round(time.perf_counter() - t0, 1)}
    log(json.dumps(summary, indent=2))
    return summary
