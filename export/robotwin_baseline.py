#!/usr/bin/env python3
"""Run Hy-VLA in RoboTwin on our task list and record every episode (Q-1).

For each task in configs/robotwin_tasks.yaml this runs RoboTwin's
script/eval_policy.py with the Hy-VLA adapter, keeps the full log, and reads
the result of every episode with its seed from the log. RoboTwin itself only
saves the success rate, but we compare setups episode by episode on the same
seeds, so each episode is needed.

Run it on the workstation, in the RoboTwinHy conda env:

    conda activate RoboTwinHy
    python export/robotwin_baseline.py --robotwin ~/robotwin/RoboTwin-0726 \\
        --ckpt ~/hyvla/ckpt/Hy-Embodied-0.5-VLA-RoboTwin --hyvla ~/hyvla/Hy-Embodied-0.5-VLA \\
        --out results/<name>

It writes <out>.episodes.jsonl (one line per episode) and <out>.json (the
summary per task, with 95% Wilson intervals). Logs go to <out>.logs/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RATE = re.compile(r"Success rate: (\d+)/(\d+) => [\d.]+%, current seed: (\d+)")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def git_state() -> dict:
    def g(*a):
        try:
            return subprocess.check_output(["git", "-C", str(REPO_ROOT), *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def parse_episodes(log_text: str) -> list[dict]:
    """One dict per finished episode, from RoboTwin's progress lines.

    After each episode RoboTwin prints "Success rate: k/n ..., current seed: s",
    where s is the seed of that episode. Success is k going up by one.
    """
    episodes, last_k = [], 0
    for line in ANSI.sub("", log_text.replace("\r", "\n")).splitlines():
        m = RATE.search(line)
        if not m:
            continue
        k, n, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if n != len(episodes) + 1:
            continue  # a repeated line
        episodes.append({"episode": n - 1, "seed": seed, "success": k > last_k})
        last_k = k
    return episodes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robotwin", required=True, type=Path, help="RoboTwin checkout at the pinned commit")
    ap.add_argument("--ckpt", required=True, help="Hy-VLA checkpoint folder")
    ap.add_argument("--hyvla", required=True, type=Path, help="Hy-VLA code (for PYTHONPATH)")
    ap.add_argument("--tasks", type=Path, default=REPO_ROOT / "configs" / "robotwin_tasks.yaml")
    ap.add_argument("--only", nargs="*", help="run only these tasks")
    ap.add_argument("--out", required=True, type=Path, help="output path without extension")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(args.tasks.read_text())
    tasks = args.only or cfg["tasks"]
    logs = args.out.with_suffix(".logs")
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f"{args.hyvla}:{os.environ.get('PYTHONPATH', '')}",
               PYTHONWARNINGS="ignore::UserWarning")
    epi_path = args.out.with_suffix(".episodes.jsonl")
    summary = {}
    with open(epi_path, "a") as epi_file:
        for task in tasks:
            cmd = [sys.executable, "-u", "script/eval_policy.py", "--config", "policy/hy_vla/deploy_policy.yml",
                   "--overrides", "--task_name", task, "--task_config", cfg["task_config"],
                   "--ckpt_setting", "Hy-VLA-RoboTwin", "--instruction_type", cfg["instruction_type"],
                   "--seed", str(cfg["eval_seed"]), "--policy_name", "policy.hy_vla", "--ckpt_path", args.ckpt]
            print(f"[{task}] running ...", flush=True)
            t = time.time()
            log_path = logs / f"{task}.log"
            with open(log_path, "w") as log:
                rc = subprocess.run(cmd, cwd=args.robotwin, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            minutes = (time.time() - t) / 60
            episodes = parse_episodes(log_path.read_text(errors="replace"))
            for e in episodes:
                epi_file.write(json.dumps({"task": task, **e}) + "\n")
            epi_file.flush()
            k, n = sum(e["success"] for e in episodes), len(episodes)
            lo, hi = wilson(k, n)
            summary[task] = {"success": k, "episodes": n, "rate": k / n if n else None,
                             "wilson_low": lo, "wilson_high": hi, "returncode": rc, "minutes": minutes}
            print(f"[{task}] {k}/{n} = {100 * k / max(n, 1):.1f}%  (95% CI {100 * lo:.1f} to {100 * hi:.1f}), "
                  f"{minutes:.0f} min, rc={rc}", flush=True)

    k = sum(s["success"] for s in summary.values())
    n = sum(s["episodes"] for s in summary.values())
    report = {"kind": "hyvla_robotwin_baseline", "timestamp": datetime.now(timezone.utc).isoformat(),
              "config": {"tasks_file": str(args.tasks), **cfg, "ckpt": args.ckpt, "robotwin": str(args.robotwin),
                         "precision": "bfloat16 (released checkpoint, PyTorch)"},
              "tasks": summary, "total": {"success": k, "episodes": n, "rate": k / n if n else None,
                                          "wilson": wilson(k, n)},
              "episodes_file": epi_path.name, "git": git_state()}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"total {k}/{n} = {100 * k / max(n, 1):.1f}%. Wrote {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
