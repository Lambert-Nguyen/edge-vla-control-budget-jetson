"""Closed-loop policy evaluation with the Jetson's inference latency injected.

The simulator does not care how long the policy takes on this computer: physics
waits for it. What it models is how long the policy would take on the Jetson, in
simulated time, taken from a measured number (or a harness JSON in results/).
So the success rate this reports is the success rate of that policy at the
Jetson's latency, measured on any laptop or workstation.

Two ways to run a chunked policy:

sync   What lerobot-record does today: when the action queue runs dry, the arm
       holds its last target while the next chunk is computed, then plays it.
       Latency costs episode time and gives the arm a pause at every chunk.

async  What LeRobot's async inference does: a new chunk is requested while the old
       one is still playing (when the queue drops to --threshold of a chunk). It
       arrives `latency` later, its first actions are already in the past and are
       dropped, and the rest replaces the queue. The arm never pauses unless the
       queue runs dry, but every action is computed from an observation that is
       `latency` old: the observation staleness the project measures.

The policy can be any LeRobot checkpoint that implements predict_action_chunk
(SmolVLA, pi0/pi0.5, ACT, Diffusion), or "expert", which replays the scripted
expert's plan in chunks and checks the timing logic without a trained model.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import ROBOT_TYPE, SimConfig
from .env import Obs, SO101Env
from .expert import ScriptedExpert, expert_rng
from .units import JointMap

SUCCESS_HOLD_S = 0.5          # the cube must stay in the box this long


# --------------------------------------------------------------------------- latency


@dataclass
class LatencyModel:
    """Chunk inference latency on the target board, in milliseconds."""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float | None = None
    max_ms: float | None = None
    source: str = "constant"

    @classmethod
    def constant(cls, ms: float) -> "LatencyModel":
        return cls(mean_ms=ms, source=f"constant {ms:g} ms")

    @classmethod
    def from_result(cls, path: str | Path, jitter: bool = True) -> "LatencyModel":
        """Read latency.chunk from a harness record, e.g. results/*smolvla*.json."""
        rec = json.loads(Path(path).read_text())
        try:
            ch = rec["latency"]["chunk"]
        except KeyError as e:
            raise ValueError(f"{path} has no latency.chunk block; is it a SmolVLA harness record?") from e
        return cls(mean_ms=float(ch["p50_ms"]), std_ms=float(ch.get("stdev_ms", 0.0)) if jitter else 0.0,
                   min_ms=ch.get("min_ms"), max_ms=ch.get("max_ms"), source=str(path))

    def sample_s(self, rng: np.random.Generator) -> float:
        ms = self.mean_ms + (rng.normal(0.0, self.std_ms) if self.std_ms > 0 else 0.0)
        lo = self.min_ms if self.min_ms is not None else 0.0
        hi = self.max_ms if self.max_ms is not None else math.inf
        return max(min(ms, hi), lo, 0.0) / 1000.0


# --------------------------------------------------------------------------- policies


class ChunkPolicy:
    """Anything that maps an observation to a chunk of actions (LeRobot units)."""
    name = "policy"
    n_action_steps = 50

    def reset(self, env: SO101Env) -> None: ...

    def chunk(self, obs: Obs, step: int, task: str) -> np.ndarray:
        raise NotImplementedError


class ExpertChunkPolicy(ChunkPolicy):
    """The scripted expert's plan, served in chunks indexed by simulator step.

    The plan is open loop, so this policy ignores the observation. It exists to
    check the latency plumbing: if chunks were misaligned in time, the gripper
    would close in the wrong place and the episode would fail.
    """
    name = "expert"

    def __init__(self, n_action_steps: int = 50, seed: int = 0):
        self.n_action_steps = n_action_steps
        self.seed = seed
        self.mode = "async"
        self.plan: np.ndarray | None = None
        self.cursor = 0

    def reset(self, env: SO101Env) -> None:
        expert = ScriptedExpert(env, expert_rng(self.seed))
        n = expert.total_steps
        actions = [expert.act() for _ in range(n)]
        self.plan = np.asarray(actions, dtype=np.float32)
        self.cursor = 0

    def chunk(self, obs: Obs, step: int, task: str) -> np.ndarray:
        # In sync mode every chunk is played in full after the pause, so a reactive
        # policy would carry on where its last chunk ended: follow a cursor. In async
        # mode elapsed actions are dropped, so the plan is indexed by simulator time.
        assert self.plan is not None
        start = self.cursor if self.mode == "sync" else step
        self.cursor = start + self.n_action_steps
        idx = np.minimum(np.arange(start, start + self.n_action_steps), len(self.plan) - 1)
        return self.plan[idx]


class LeRobotChunkPolicy(ChunkPolicy):
    """A LeRobot checkpoint (local folder or hub id), returning unnormalized chunks."""

    def __init__(self, path: str, device: str = "auto", num_steps: int | None = None,
                 n_action_steps: int | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        if device == "auto":
            device = ("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
                      "cpu")
        self.device = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        policy.to(self.device).eval()
        if num_steps is not None and hasattr(policy.config, "num_steps"):
            policy.config.num_steps = num_steps
        self.policy = policy
        self.pre, self.post = make_pre_post_processors(
            policy.config, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        self.n_action_steps = int(n_action_steps or getattr(policy.config, "n_action_steps", 1))
        self.name = f"{cfg.type}:{path}"
        self.image_keys = list(getattr(policy.config, "image_features", {}) or {})
        self._torch = torch

    def reset(self, env: SO101Env) -> None:
        self.policy.reset()
        missing = [k for k in self.image_keys if k.split(".")[-1] not in env.cfg.cameras]
        if missing:
            print(f"warning: the policy expects {missing}, which the simulator does not render")

    def chunk(self, obs: Obs, step: int, task: str) -> np.ndarray:
        from lerobot.utils.control_utils import prepare_observation_for_inference
        torch = self._torch
        with torch.inference_mode():
            batch = prepare_observation_for_inference(dict(obs.as_lerobot()), self.device, task, ROBOT_TYPE)
            batch = self.pre(batch)
            actions = self.policy.predict_action_chunk(batch)       # (1, chunk, dim), normalized
            actions = actions[:, : self.n_action_steps]
            out = [self.post(actions[:, i]) for i in range(actions.shape[1])]
        return torch.cat([a.reshape(1, -1) for a in out]).float().cpu().numpy()


def load_policy(spec: str, **kw) -> ChunkPolicy:
    if spec == "expert":
        return ExpertChunkPolicy(n_action_steps=kw.get("n_action_steps") or 50)
    return LeRobotChunkPolicy(spec, device=kw.get("device", "auto"), num_steps=kw.get("num_steps"),
                              n_action_steps=kw.get("n_action_steps"))


# --------------------------------------------------------------------------- rollout


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    steps: int
    time_to_success_s: float | None
    inferences: int
    stall_steps: int
    staleness_ms: list[float] = field(default_factory=list, repr=False)
    task: str = ""
    wrong_object_in_box: bool = False   # another object ended up in the box
    scene: dict = field(default_factory=dict)

    def summary(self) -> dict:
        st = np.asarray(self.staleness_ms) if self.staleness_ms else np.zeros(1)
        d = asdict(self)
        d.pop("staleness_ms")
        d.update(staleness_mean_ms=float(st.mean()), staleness_p95_ms=float(np.percentile(st, 95)))
        return d


class ChunkExecutor:
    """Turns a chunk policy plus a latency model into one action per control step.

    Used by `rollout` for batch evaluation and by the app to drive the arm live.
    Times are in control steps of the simulator; nothing here reads a wall clock.
    """

    def __init__(self, policy: ChunkPolicy, latency: LatencyModel, fps: int, mode: str = "async",
                 threshold: float = 0.5, rng: np.random.Generator | None = None):
        if mode not in ("sync", "async"):
            raise ValueError(f"mode must be sync or async, not {mode!r}")
        self.policy, self.latency, self.fps = policy, latency, fps
        self.mode, self.threshold = mode, threshold
        self.rng = rng or np.random.default_rng()
        self.queue: list[tuple[np.ndarray, int]] = []         # (action, step of its observation)
        self.pending: tuple[int, np.ndarray, int] | None = None  # (arrival step, chunk, obs step)
        self.last_action: np.ndarray | None = None
        self.inferences = self.stalls = 0
        self.staleness_ms: list[float] = []

    def reset(self, start_action: np.ndarray) -> None:
        self.queue, self.pending = [], None
        self.last_action = np.asarray(start_action, dtype=np.float32)
        self.inferences = self.stalls = 0
        self.staleness_ms = []

    def _request(self, step: int, obs: Obs, task: str):
        self.inferences += 1
        ch = self.policy.chunk(obs, step, task)
        delay = int(math.ceil(self.latency.sample_s(self.rng) * self.fps - 1e-9))
        return step + delay, ch, step

    @property
    def waiting(self) -> bool:
        return self.pending is not None and not self.queue

    def next_action(self, step: int, obs: Obs, task: str) -> np.ndarray:
        n = self.policy.n_action_steps
        if self.mode == "sync":
            # Plan only when the queue is empty; the arm holds until the chunk arrives.
            if not self.queue and self.pending is None:
                self.pending = self._request(step, obs, task)
        elif self.pending is None and len(self.queue) <= self.threshold * n:
            self.pending = self._request(step, obs, task)

        if self.pending is not None and step >= self.pending[0]:
            _, ch, obs_step = self.pending
            self.pending = None
            if self.mode == "sync":
                self.queue = [(a, obs_step) for a in ch]
            else:
                # Actions meant for steps that have already passed are dropped.
                fresh = [(a, obs_step) for a in ch[step - obs_step:]]
                if fresh:
                    self.queue = fresh
                if not self.queue:                     # chunk shorter than the latency
                    self.pending = self._request(step, obs, task)

        if self.queue:
            action, src = self.queue.pop(0)
            self.staleness_ms.append((step - src) * 1000.0 / self.fps)
        else:
            action = self.last_action
            self.stalls += 1
        self.last_action = action
        return action


def rollout(env: SO101Env, policy: ChunkPolicy, latency: LatencyModel, seed: int, mode: str = "async",
            threshold: float = 0.5, rng: np.random.Generator | None = None, on_step=None,
            render: bool = True) -> EpisodeResult:
    """One episode. on_step(env, obs, action) is called after every control step."""
    if isinstance(policy, ExpertChunkPolicy):
        policy.seed, policy.mode = seed, mode
    obs = env.reset(seed=seed, render=render)
    policy.reset(env)
    ex = ChunkExecutor(policy, latency, env.cfg.fps, mode, threshold,
                       rng or np.random.default_rng(seed + 10_000))
    ex.reset(env.state())
    hold_needed = int(round(SUCCESS_HOLD_S * env.cfg.fps))
    in_box = 0
    success_step = None
    step = 0
    while step < env.cfg.max_steps:
        action = ex.next_action(step, obs, env.task)
        obs = env.step(action, render=render)
        step += 1
        if on_step:
            on_step(env, obs, action)
        in_box = in_box + 1 if env.is_success() else 0
        if in_box >= hold_needed:
            success_step = step
            break
    return EpisodeResult(seed=seed, success=success_step is not None, steps=step,
                         time_to_success_s=None if success_step is None else success_step / env.cfg.fps,
                         inferences=ex.inferences, stall_steps=ex.stalls, staleness_ms=ex.staleness_ms,
                         task=env.task, wrong_object_in_box=env.distractor_in_box(),
                         scene={k: v for k, v in env.episode_info().items() if k not in ("task", "noise")})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def evaluate(policy_spec: str, cfg: SimConfig, episodes: int, latency: LatencyModel, mode: str = "async",
             threshold: float = 0.5, seed: int = 100_000, device: str = "auto", num_steps: int | None = None,
             n_action_steps: int | None = None, out: str | Path | None = None,
             joint_map: JointMap | None = None, log=print) -> dict:
    """Evaluate a policy over `episodes` seeded layouts and write a JSON record.

    Seeds start at 100000 by default so evaluation layouts never repeat the
    layouts that `generate` (seeds from 0) put into a training set.
    """
    env = SO101Env(cfg, joint_map=joint_map, render=True)
    policy = load_policy(policy_spec, device=device, num_steps=num_steps, n_action_steps=n_action_steps)
    results = []
    t0 = time.perf_counter()
    try:
        for i in range(episodes):
            # The expert ignores images, so skip rendering for it.
            r = rollout(env, policy, latency, seed + i, mode=mode, threshold=threshold,
                        render=not isinstance(policy, ExpertChunkPolicy))
            results.append(r)
            k = sum(x.success for x in results)
            log(f"episode {i + 1:3d}/{episodes}  seed {r.seed}  {'success' if r.success else 'fail   '}  "
                f"{r.steps / cfg.fps:5.1f}s  inferences {r.inferences:3d}  stalls {r.stall_steps:4d}  "
                f"running {k}/{i + 1}" + ("  WRONG OBJECT IN BOX" if r.wrong_object_in_box else "")
                + ("" if env.original_cube else f"  [{r.task}]"))
    finally:
        env.close()
    k = sum(r.success for r in results)
    lo, hi = wilson(k, len(results))
    stale = np.concatenate([np.asarray(r.staleness_ms) for r in results if r.staleness_ms] or [np.zeros(1)])
    record = {
        "kind": "so101_sim_eval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"policy": policy_spec, "mode": mode, "threshold": threshold, "episodes": episodes,
                   "first_seed": seed, "randomize": cfg.randomize, "start_spread": cfg.start_spread, "fps": cfg.fps,
                   "episode_s": cfg.episode_s, "cameras": list(cfg.cameras),
                   "camera1_pos": list(cfg.camera1_pos), "camera1_target": list(cfg.camera1_target),
                   "objects": list(cfg.objects), "scene": cfg.scene, "noise": cfg.noise,
                   "n_action_steps": policy.n_action_steps, "num_steps": num_steps,
                   "latency": asdict(latency), "joint_map": asdict(env.jmap)},
        "success": {"k": k, "n": len(results), "rate": k / max(len(results), 1), "wilson95": [lo, hi]},
        "staleness_ms": {"mean": float(stale.mean()), "p50": float(np.percentile(stale, 50)),
                         "p95": float(np.percentile(stale, 95)), "max": float(stale.max())},
        "stall_fraction": float(sum(r.stall_steps for r in results) / max(sum(r.steps for r in results), 1)),
        "wrong_object_in_box": int(sum(r.wrong_object_in_box for r in results)),
        "episodes": [r.summary() for r in results],
        "host": {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
                 "git_commit": git_commit(Path(__file__).resolve().parent)},
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    log(f"success {k}/{len(results)} = {record['success']['rate']:.1%}  (Wilson 95% {lo:.1%}-{hi:.1%}), "
        f"staleness mean {record['staleness_ms']['mean']:.0f} ms, stalls {record['stall_fraction']:.1%}")
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")
        log(f"wrote {out}")
    return record
