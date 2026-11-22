"""A scripted expert that picks the cube up and puts it in the box.

It reads the true cube and box poses from the simulator once, at the start of the
episode, and plays a fixed sequence of moves. Its actions are joint targets in
LeRobot units, the same thing a leader arm sends, so its demonstrations can be
recorded and trained on exactly like teleoperated ones.

Each episode varies the speed, the approach height and small offsets, so that a
policy trained on many expert demos does not see one memorised trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .env import SO101Env
from .ik import ROLL, ArmIK
from .units import GRIPPER_LO

GRIP_OPEN = 0.65          # rad, about 65 mm between the jaws
GRIP_CLOSED = GRIPPER_LO  # command fully closed; the cube stops the jaw
CUBE_HALF_REF = 0.015     # m, the half width TCP_OFFSET was placed for (the 3 cm cube)
EXTRA_CLEARANCE = 0.003   # m, more room past the fixed jaw for objects other than the original cube
# Lowest tool point, the height the 3 cm cube is grasped at. Lower, the moving jaw's
# tip hits the table as it swings shut and the gripper jams half open.
MIN_GRASP_HEIGHT = 0.015


# The expert draws from its own stream of the episode seed. The environment draws
# the layout and appearance from default_rng(seed) and the object, noise and
# colours from [seed, 1..4]. Before 2026-11-21 the expert used default_rng(seed)
# as well, so its speed was the same random number as the first light's
# brightness, and its heights followed the next lighting draws.
EXPERT_STREAM = 5


def expert_rng(seed: int) -> np.random.Generator:
    """The expert's generator for an episode seed, independent of the scene's draws."""
    return np.random.default_rng([int(seed), EXPERT_STREAM])


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


@dataclass
class Segment:
    kind: str                  # "joint" (interpolate joints) or "cart" (straight tool line)
    duration: float            # seconds
    grip: float                # gripper target at the end, rad
    end_pos: np.ndarray | None = None
    end_q: np.ndarray | None = None
    pitch: float = math.pi / 2  # "cart": jaw pitch at the end, below horizontal


class ScriptedExpert:
    def __init__(self, env: SO101Env, rng: np.random.Generator | None = None,
                 speed: float | None = None):
        self.env = env
        self.ik = ArmIK(env.model, env.qadr, env.dadr)
        self.rng = rng or np.random.default_rng()
        self.speed = speed if speed is not None else float(self.rng.uniform(0.8, 1.25))
        self.dt = 1.0 / env.cfg.fps
        self._plan()

    # ------------------------------------------------------------------ planning

    def _plan(self) -> None:
        env, rng = self.env, self.rng
        cube, yaw = env.cube_pose()
        box = env.box_pos()
        jitter = lambda s: rng.uniform(-s, s, 3) * np.array([1, 1, 0])  # noqa: E731
        obj = env.obj
        hover = rng.uniform(0.06, 0.09)
        carry = rng.uniform(0.09, 0.12)
        release = env.box_height + obj.half_height + rng.uniform(0.012, 0.025)

        grasp = cube + jitter(0.002)
        # Low objects (a lying marker, a small ball) are grasped a little above their
        # centre so the jaw tips stay clear of the table.
        grasp[2] = cube[2] if env.original_cube else max(cube[2], MIN_GRASP_HEIGHT)
        above_cube = grasp + [0, 0, hover]
        above_box = box + jitter(0.01)
        above_box[2] = carry
        drop = above_box.copy()
        drop[2] = release
        lift = grasp.copy()
        lift[2] = carry

        # The jaws close along the object's jaw axis (a cube's face normal, a block's
        # short side); round objects take any direction, so the wrist keeps its roll.
        jaw_dir = None
        if obj.jaw_axis is not None:
            c, s = math.cos(yaw), math.sin(yaw)
            ax, ay, _ = obj.jaw_axis
            jaw_dir = np.array([c * ax - s * ay, s * ax + c * ay, 0.0])
        q_start = env.qpos_arm()
        q_hover, err = self.ik.solve_topdown(q_start, above_cube, jaw_dir, roll_prefer=0.0,
                                             symmetric=max(obj.symmetry, 1))
        # The tool point is placed so a 3 cm cube clears the fixed jaw by 4 mm (see
        # TCP_OFFSET). A wider object would land on the fixed jaw and a narrower one be
        # pushed further, so shift the grasp along the jaw axis by the difference.
        # Objects other than the original cube get 3 mm more clearance (7 mm): with
        # noise on, backlash and gain errors put the jaws a few mm off, and a fixed jaw
        # landing on a ball sends it rolling away.
        shift = CUBE_HALF_REF - obj.grip_half - (0.0 if env.original_cube else EXTRA_CLEARANCE)
        if abs(shift) > 1e-9:
            jaw_axis_world = self.ik.fk(q_hover)[1][:, 2]
            jaw_axis_world = jaw_axis_world - jaw_axis_world[2] * np.array([0, 0, 1.0])
            jaw_axis_world /= max(np.linalg.norm(jaw_axis_world), 1e-9)
            grasp = grasp + shift * jaw_axis_world
            above_cube = grasp + [0, 0, hover]
            lift = grasp.copy()
            lift[2] = carry
            q_hover, err = self.ik.solve_topdown(q_hover, above_cube, None)
        if err > 1e-3:
            # Straight down at hover height needs more wrist_flex than the joint has
            # (far cubes), and solve_topdown's compromise misses the point by up to
            # 2-3 cm. The descent would then come in at a slant and the fixed jaw land
            # on the cube's edge, so hover exactly over the cube, tilted as little as
            # needed. Starting from the compromise keeps the same arm configuration.
            q_hover, _, _ = self.ik.solve_steepest(q_hover, above_cube)
        self.roll = q_hover[ROLL]
        q_home = env.home.copy()

        s = 1.0 / self.speed
        self.segments = [
            Segment("joint", 1.6 * s, GRIP_OPEN, end_q=q_hover),
            Segment("cart", 1.0 * s, GRIP_OPEN, end_pos=grasp),
            Segment("cart", 0.6 * s, GRIP_CLOSED, end_pos=grasp),        # close
            Segment("cart", 0.8 * s, GRIP_CLOSED, end_pos=lift),
            Segment("cart", 1.3 * s, GRIP_CLOSED, end_pos=above_box),
            Segment("cart", 0.6 * s, GRIP_CLOSED, end_pos=drop),
            Segment("cart", 0.5 * s, GRIP_OPEN, end_pos=drop),           # release
            Segment("cart", 0.5 * s, GRIP_OPEN, end_pos=above_box),
            Segment("joint", 1.5 * s, float(q_home[5]), end_q=q_home),
            Segment("joint", 0.5, float(q_home[5]), end_q=q_home),       # settle
        ]
        # Carry and drop points over a far box are out of reach with the jaws
        # straight down; tilt them as little as needed so the cube goes where planned.
        q = q_hover.copy()
        for seg in self.segments:
            if seg.kind == "cart":
                q, seg.pitch, _ = self.ik.solve_steepest(q, seg.end_pos)
            else:
                q = seg.end_q.copy()
        self._seg = 0
        self._t = 0.0
        self._q = q_start.copy()
        self._seg_start_grip = float(q_start[5])
        self._start_segment(q_start)

    def _start_segment(self, q: np.ndarray) -> None:
        self._seg_start_q = q.copy()
        self._seg_start_pos, R = self.ik.fk(q)
        self._seg_start_pitch = self.ik.pitch_of(R, self.ik._radial(self._seg_start_pos))

    def _solve(self, q_init: np.ndarray, pos: np.ndarray, pitch: float = math.pi / 2) -> tuple[np.ndarray, float]:
        q = np.array(q_init, dtype=float)
        q[ROLL] = self.roll
        return self.ik.solve(q, pos, pitch)

    # ------------------------------------------------------------------ acting

    @property
    def done(self) -> bool:
        return self._seg >= len(self.segments)

    @property
    def total_steps(self) -> int:
        return int(math.ceil(sum(seg.duration for seg in self.segments) / self.dt))

    def act(self) -> np.ndarray:
        """Next action in LeRobot units. Holds the last target once done."""
        if self.done:
            return self.env.action_for(self._q)
        seg = self.segments[self._seg]
        self._t += self.dt
        s = min_jerk(self._t / seg.duration)
        if seg.kind == "joint":
            q = self._seg_start_q + s * (seg.end_q - self._seg_start_q)
        else:
            pos = self._seg_start_pos + s * (seg.end_pos - self._seg_start_pos)
            dp = seg.pitch - self._seg_start_pitch
            pitch = seg.pitch if abs(dp) < 2e-3 else self._seg_start_pitch + s * dp
            q, _ = self._solve(self._q, pos, pitch)
        q[5] = self._seg_start_grip + s * (seg.grip - self._seg_start_grip)
        self._q = q
        if self._t >= seg.duration - 1e-9:
            self._seg += 1
            self._t = 0.0
            self._start_segment(q)
            self._seg_start_grip = float(q[5])
        return self.env.action_for(q)


def run_expert_episode(env: SO101Env, seed: int, render: bool = False, extra_s: float = 0.0):
    """Reset with seed and let the expert play. Yields (obs_before, action) per step.

    The caller observes after each step through env; this keeps rendering optional.
    """
    obs = env.reset(seed=seed, render=render)
    expert = ScriptedExpert(env, expert_rng(seed))
    n = expert.total_steps + int(extra_s * env.cfg.fps)
    for _ in range(n):
        action = expert.act()
        yield obs, action
        obs = env.step(action, render=render)


__all__ = ["ScriptedExpert", "expert_rng", "run_expert_episode", "GRIP_OPEN", "GRIP_CLOSED"]
