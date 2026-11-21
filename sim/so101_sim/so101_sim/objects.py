"""The objects the arm picks up, and how to grasp each one.

One free body ("cube" in the scene XML) carries one geom. At each reset the
environment changes that geom's shape, size, mass and inertia to the episode's
object, so every object shares the same joint, contact settings and success check.

| kind     | shape                    | size range                               | grasp                    |
| -------- | ------------------------ | ---------------------------------------- | ------------------------ |
| cube     | box                      | 2.4-3.6 cm sides                         | across two faces         |
| block    | box lying flat           | 4.4-6 cm long, 2.4-3 cm wide and high    | across the short side    |
| cylinder | cylinder standing up     | 2.4-3.4 cm across, 3.6-6 cm tall         | any direction            |
| ball     | sphere                   | 2.8-3.6 cm across                        | any direction            |
| marker   | capsule lying flat       | 2-2.6 cm across, 7.2-9.4 cm long         | across the body          |

All fit the open jaws (about 65 mm) with room to spare and the box (98 mm inside).
Long objects (block, marker) lie within 45 degrees of across the arm's reach: one
pointing straight at the arm near its base needs the wrist rolled 90 degrees, and
the wrist camera mount then hits the shoulder.
Mass comes from the volume at 1100 kg/m^3 (solid plastic), so the original 3 cm
cube weighs 30 g as before. With randomize "none" each object has its middle size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

DENSITY = 1100.0                     # kg/m^3
MASS_RANGE = (0.01, 0.08)            # kg

# A capsule lies along the body's x axis: its own axis (z) turned 90 degrees about y.
_LYING = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
_UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])

# Size ranges per kind: (low, high) for each of the geom's size parameters, metres.
SIZE_RANGES = {
    "cube": [(0.012, 0.018)],                                     # half side
    "block": [(0.022, 0.030), (0.012, 0.015), (0.012, 0.015)],    # half x (long), y, z
    "cylinder": [(0.012, 0.017), (0.018, 0.030)],                 # radius, half height
    "ball": [(0.014, 0.018)],                                     # radius
    "marker": [(0.010, 0.013), (0.026, 0.034)],                   # radius, half length of the shaft
}

NAMES = {"cube": "cube", "block": "block", "cylinder": "cylinder", "ball": "ball", "marker": "marker"}

# Named colours for scenes where the instruction names one. Chosen to stay apart
# from each other, the brown box and the beige table under randomized lighting.
COLORS = {
    "red": (0.85, 0.12, 0.10),
    "green": (0.15, 0.65, 0.20),
    "blue": (0.12, 0.30, 0.85),
    "yellow": (0.95, 0.82, 0.10),
    "purple": (0.55, 0.22, 0.72),
    "white": (0.92, 0.92, 0.92),
}


@dataclass(frozen=True)
class ObjectSpec:
    kind: str
    geom_type: int
    size: tuple[float, float, float]     # MuJoCo geom size
    geom_quat: tuple[float, ...]
    half_height: float                   # centre above the table when resting
    grip_half: float                     # half width the jaws close across
    extent: float                        # largest horizontal half extent from the centre
    footprint: float                     # horizontal bounding radius
    jaw_axis: tuple[float, float, float] | None   # body-frame direction to close along; None = any
    symmetry: int                        # equivalent jaw directions (4 for a cube, 2 for a block)
    mass: float
    inertia: tuple[float, float, float]  # principal moments in the body frame
    condim: int                          # 6 adds rolling friction for things that roll

    @property
    def name(self) -> str:
        return NAMES[self.kind]


def make_object(kind: str, size: list[float]) -> ObjectSpec:
    if kind == "cube":
        (h,) = size
        vol = (2 * h) ** 3
        m = _mass(vol)
        i = m * (2 * h) ** 2 / 6
        return ObjectSpec(kind, mujoco.mjtGeom.mjGEOM_BOX, (h, h, h), tuple(_UPRIGHT), h, h, h,
                          h * math.sqrt(2), (1.0, 0.0, 0.0), 4, m, (i, i, i), 4)
    if kind == "block":
        hx, hy, hz = size
        m = _mass(8 * hx * hy * hz)
        inertia = (m * (hy ** 2 + hz ** 2) / 3, m * (hx ** 2 + hz ** 2) / 3, m * (hx ** 2 + hy ** 2) / 3)
        return ObjectSpec(kind, mujoco.mjtGeom.mjGEOM_BOX, (hx, hy, hz), tuple(_UPRIGHT), hz, hy, hx,
                          math.hypot(hx, hy), (0.0, 1.0, 0.0), 2, m, inertia, 4)
    if kind == "cylinder":
        r, hh = size
        m = _mass(math.pi * r * r * 2 * hh)
        ixy = m * (3 * r * r + (2 * hh) ** 2) / 12
        return ObjectSpec(kind, mujoco.mjtGeom.mjGEOM_CYLINDER, (r, hh, 0.0), tuple(_UPRIGHT), hh, r, r, r,
                          None, 0, m, (ixy, ixy, m * r * r / 2), 6)
    if kind == "ball":
        (r,) = size
        m = _mass(4 / 3 * math.pi * r ** 3)
        i = 0.4 * m * r * r
        return ObjectSpec(kind, mujoco.mjtGeom.mjGEOM_SPHERE, (r, 0.0, 0.0), tuple(_UPRIGHT), r, r, r, r,
                          None, 0, m, (i, i, i), 6)
    if kind == "marker":
        r, hl = size
        m = _mass(math.pi * r * r * 2 * hl + 4 / 3 * math.pi * r ** 3)
        length = 2 * (hl + r)
        i_perp = m * (3 * r * r + length ** 2) / 12
        # Lying along body x, so the jaws close along body y.
        return ObjectSpec(kind, mujoco.mjtGeom.mjGEOM_CAPSULE, (r, hl, 0.0), tuple(_LYING), r, r, hl + r,
                          hl + r, (0.0, 1.0, 0.0), 2, m, (m * r * r / 2, i_perp, i_perp), 6)
    raise ValueError(f"unknown object {kind!r}; choose from {', '.join(SIZE_RANGES)}")


def _mass(volume: float) -> float:
    return float(np.clip(DENSITY * volume, *MASS_RANGE))


def nominal(kind: str) -> ObjectSpec:
    """The middle of the kind's size range."""
    return make_object(kind, [0.5 * (lo + hi) for lo, hi in SIZE_RANGES[kind]])


def sample(kinds: tuple[str, ...], rng: np.random.Generator, vary_size: bool = True) -> ObjectSpec:
    kind = kinds[int(rng.integers(len(kinds)))] if len(kinds) > 1 else kinds[0]
    if not vary_size:
        return nominal(kind)
    return make_object(kind, [float(rng.uniform(lo, hi)) for lo, hi in SIZE_RANGES[kind]])


def sample_scene(scene: str, kinds: tuple[str, ...], rng: np.random.Generator, vary_size: bool = True,
                 count: int = 3) -> tuple[list[ObjectSpec], list[str | None], str]:
    """Objects for one episode, the object to pick up first.

    Returns (objects, colour names, label), where the label is how the instruction
    names the object to pick up: "ball", or "red cube" in the colors scene.
    - single: one object of a kind drawn from kinds; colour left to randomization.
    - colors: count objects of one kind (sizes drawn separately), each a different colour.
    - kinds:  count different kinds (all kinds if fewer than two are given), one colour.
    """
    if scene == "single":
        obj = sample(kinds, rng, vary_size)
        return [obj], [None], obj.name
    if scene == "colors":
        kind = kinds[int(rng.integers(len(kinds)))]
        objs = [sample((kind,), rng, vary_size) for _ in range(count)]
        colors = [str(c) for c in rng.choice(list(COLORS), size=count, replace=False)]
        return objs, colors, f"{colors[0]} {objs[0].name}"
    if scene == "kinds":
        pool = list(dict.fromkeys(kinds))
        if len(pool) < 2:
            pool = list(SIZE_RANGES)
        chosen = [str(k) for k in rng.choice(pool, size=min(count, len(pool)), replace=False)]
        objs = [sample((k,), rng, vary_size) for k in chosen]
        color = str(rng.choice(list(COLORS)))
        return objs, [color] * len(objs), objs[0].name
    raise ValueError(f"unknown scene {scene!r}")


def color_rgb(name: str, rng: np.random.Generator, jitter: float = 0.0) -> np.ndarray:
    rgb = np.asarray(COLORS[name]) + rng.uniform(-jitter, jitter, 3)
    return np.clip(rgb, 0.0, 1.0)


def apply(model: mujoco.MjModel, body: int, geom: int, obj: ObjectSpec) -> None:
    """Turn the object geom and its body into obj. Call mj_setConst afterwards."""
    model.geom_type[geom] = obj.geom_type
    model.geom_size[geom] = obj.size
    model.geom_quat[geom] = obj.geom_quat
    model.geom_condim[geom] = obj.condim
    model.geom_rbound[geom] = _rbound(obj)
    model.geom_aabb[geom] = np.r_[np.zeros(3), _aabb_half(obj)]
    model.body_mass[body] = obj.mass
    model.body_inertia[body] = obj.inertia
    model.body_iquat[body] = _UPRIGHT
    model.body_ipos[body] = 0.0


def _rbound(obj: ObjectSpec) -> float:
    a, b, c = obj.size
    if obj.geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return math.sqrt(a * a + b * b + c * c)
    if obj.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return a
    if obj.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return math.hypot(a, b)
    return a + b                                          # capsule


def _aabb_half(obj: ObjectSpec) -> np.ndarray:
    """Half sizes of the geom's box in the geom's own frame."""
    a, b, c = obj.size
    if obj.geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([a, b, c])
    if obj.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.array([a, a, a])
    if obj.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return np.array([a, a, b])
    return np.array([a, a, a + b])                        # capsule


__all__ = ["ObjectSpec", "make_object", "nominal", "sample", "sample_scene", "color_rgb", "apply",
           "SIZE_RANGES", "NAMES", "COLORS"]
