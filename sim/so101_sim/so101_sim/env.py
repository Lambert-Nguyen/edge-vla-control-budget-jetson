"""The SO-101 cube-to-box task in MuJoCo.

Actions and the robot state use LeRobot's SO-101 units (see units.py), so a policy
trained on data from this simulator sees the same kind of numbers it would see on
the real follower, and the other way round.

    env = SO101Env(SimConfig())
    obs = env.reset(seed=0)
    obs = env.step(action)        # 6 values: 5 joints in degrees, gripper 0-100
    env.is_success()

Every public method runs on the calling thread. Rendering uses an OpenGL context,
so on macOS create and use the environment on the main thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import JOINT_NAMES, SCENE_XML, START_MIN_HEIGHT, START_SPREAD, SimConfig
from .units import JointMap

# Dataset camera key -> camera name in the MJCF.
CAMERA_SOURCES = {"camera1": "camera1", "camera2": "wrist_cam"}

# The tool point the expert and keyboard teleop steer, in the gripperframe site's
# frame. The site's x axis points out of the jaws, its z axis runs across them
# from the fixed jaw (z = -20 mm) to the moving one. This point sits 8 mm back from
# the site and 1 mm toward the moving jaw: a 3 cm cube centred here clears the fixed
# jaw by 4 mm on the way down, and the closing moving jaw pushes it the last 4 mm.
TCP_OFFSET = np.array([-0.008, 0.0, -0.001])

GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1", "camera_mount")


@dataclass
class Obs:
    state: np.ndarray                 # float32[6], LeRobot units
    images: dict[str, np.ndarray]     # camera key -> HxWx3 uint8 RGB

    def as_lerobot(self) -> dict[str, np.ndarray]:
        """Observation dict with LeRobot's keys, as predict_action and add_frame expect."""
        out = {"observation.state": self.state}
        for key, img in self.images.items():
            out[f"observation.images.{key}"] = img
        return out


def look_at_quat(pos, target) -> np.ndarray:
    """Quaternion for a MuJoCo camera at pos looking at target, with +z world up."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:          # looking straight down
        right = np.array([0.0, -1.0, 0.0])
    right /= np.linalg.norm(right)
    z = -fwd                                   # MuJoCo cameras look along their -z
    y = np.cross(z, right)
    mat = np.column_stack([right, y, z])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    return quat


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class SO101Env:
    def __init__(self, cfg: SimConfig | None = None, joint_map: JointMap | None = None,
                 render: bool = True, seed: int | None = None):
        self.cfg = cfg or SimConfig()
        self.jmap = joint_map or JointMap()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.model.opt.timestep = 1.0 / (self.cfg.fps * self.cfg.substeps)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.render_enabled = render
        self._renderer: mujoco.Renderer | None = None

        m = self.model
        name2id = lambda kind, name: mujoco.mj_name2id(m, kind, name)  # noqa: E731
        jids = [name2id(mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINT_NAMES]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.act = np.array([name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINT_NAMES])
        # LeRobot's so101_follower caps the gripper servo at 50% torque (Max_Torque_Limit
        # 500 of 1000), so the simulated gripper gets half the arm servos' force range.
        grip_act = self.act[-1]
        m.actuator_forcerange[grip_act] = 0.5 * m.actuator_forcerange[grip_act]
        self.ctrl_lo = m.actuator_ctrlrange[self.act, 0].copy()
        self.ctrl_hi = m.actuator_ctrlrange[self.act, 1].copy()
        cube_j = name2id(mujoco.mjtObj.mjOBJ_JOINT, "cube")
        self.cube_qadr = m.jnt_qposadr[cube_j]
        self.cube_dadr = m.jnt_dofadr[cube_j]
        self.cube_body = name2id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.box_body = name2id(mujoco.mjtObj.mjOBJ_BODY, "box")
        self.table_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.site = name2id(mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.gripper_body = name2id(mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.cam_ids = {k: name2id(mujoco.mjtObj.mjOBJ_CAMERA, v) for k, v in CAMERA_SOURCES.items()}
        self.cam1 = self.cam_ids["camera1"]
        gb = {name2id(mujoco.mjtObj.mjOBJ_BODY, b) for b in GRIPPER_BODIES}
        self.gripper_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in gb}
        box_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.box_body]
        self.box_geoms = box_geoms

        # Box interior from its geoms: walls at +-x and +-y, floor at the bottom.
        walls = [g for g in box_geoms if "wall" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        self.box_inner_half = float(min(abs(m.geom_pos[g][0]) - m.geom_size[g][0]
                                        for g in walls if abs(m.geom_pos[g][0]) > 1e-6))
        self.box_outer_half = float(max(abs(m.geom_pos[g][0]) + m.geom_size[g][0] for g in walls))
        self.box_height = float(max(m.geom_pos[g][2] + m.geom_size[g][2] for g in walls))
        self.cube_half = float(m.geom_size[self.cube_geom][0])

        # Defaults that randomization perturbs, kept so that "none" restores them.
        self._default_rgba = m.geom_rgba.copy()
        self._default_mat_rgba = m.mat_rgba.copy()
        self._default_light = (m.light_diffuse.copy(), m.light_pos.copy(), m.light_dir.copy())
        self._default_cube_mass = float(m.body_mass[self.cube_body])
        self._default_cube_inertia = m.body_inertia[self.cube_body].copy()
        self._default_cube_friction = m.geom_friction[self.cube_geom].copy()
        self._cube_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "cube")
        self._box_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "box")
        self._table_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "table")

        self.home = self.jmap.to_sim(np.array(self.cfg.home_pose, dtype=float))
        self.start = self.home.copy()       # this episode's start pose, sim radians
        self._fixed_bodies = {0, self.box_body, self.cube_body}
        self.step_count = 0
        self.task = self.cfg.task.instruction
        self._aim_camera1(self.cfg.camera1_pos, self.cfg.camera1_target, self.cfg.camera1_fovy)

    # ----------------------------------------------------------------- rendering

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.cfg.height, self.cfg.width)
        return self._renderer

    def render_camera(self, key: str) -> np.ndarray:
        r = self.renderer
        r.update_scene(self.data, camera=self.cam_ids[key])
        return r.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ----------------------------------------------------------------- state

    def qpos_arm(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def state(self) -> np.ndarray:
        return self.jmap.to_lerobot(self.qpos_arm())

    def observe(self, render: bool | None = None) -> Obs:
        render = self.render_enabled if render is None else render
        images = {k: self.render_camera(k) for k in self.cfg.cameras} if render else {}
        return Obs(self.state(), images)

    def cube_pose(self) -> tuple[np.ndarray, float]:
        """Cube centre (m) and yaw (rad) in the world frame."""
        pos = self.data.xpos[self.cube_body].copy()
        q = self.data.xquat[self.cube_body]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return pos, yaw

    def box_pos(self) -> np.ndarray:
        return self.model.body_pos[self.box_body].copy()

    def tcp_pos(self) -> np.ndarray:
        R = self.data.site_xmat[self.site].reshape(3, 3)
        return self.data.site_xpos[self.site] + R @ TCP_OFFSET

    def cube_in_gripper_contact(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if self.cube_geom in pair and pair & self.gripper_geoms:
                return True
        return False

    def cube_in_box(self) -> bool:
        pos, _ = self.cube_pose()
        rel = pos - self.box_pos()
        margin = self.box_inner_half - 0.5 * self.cube_half
        return bool(abs(rel[0]) < margin and abs(rel[1]) < margin and pos[2] < self.box_height)

    def is_success(self) -> bool:
        """Cube resting inside the box and released by the gripper."""
        return self.cube_in_box() and not self.cube_in_gripper_contact()

    # ----------------------------------------------------------------- control

    def step(self, action: np.ndarray, render: bool | None = None) -> Obs:
        """Apply one action (LeRobot units) for one control period and observe."""
        self.apply_action(action)
        return self.observe(render)

    def apply_action(self, action: np.ndarray) -> None:
        target = self.jmap.to_sim(np.asarray(action, dtype=np.float64))
        if self.cfg.max_step_deg is not None:
            cur = self.qpos_arm()
            lim = math.radians(self.cfg.max_step_deg)
            target = np.clip(target, cur - lim, cur + lim)
        self.data.ctrl[self.act] = np.clip(target, self.ctrl_lo, self.ctrl_hi)
        mujoco.mj_step(self.model, self.data, nstep=self.cfg.substeps)
        self.step_count += 1

    def hold(self, seconds: float) -> None:
        """Advance physics with the current targets, without counting control steps."""
        n = int(round(seconds / self.model.opt.timestep))
        mujoco.mj_step(self.model, self.data, nstep=max(n, 1))

    # ----------------------------------------------------------------- reset

    def reset(self, seed: int | None = None, render: bool | None = None) -> Obs:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        level = self.cfg.randomize
        mass_before = float(m.body_mass[self.cube_body])
        self._randomize_appearance(level)
        if float(m.body_mass[self.cube_body]) != mass_before:
            # Masses feed constants the solver precomputes at compile time
            # (invweight0 and friends). Recompute them, or a lighter cube makes the
            # contact solver blow up. mj_setConst uses d as scratch, so call it
            # before the episode state is written.
            mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        cube_xy, cube_yaw, box_xy = self._sample_layout(level)
        m.body_pos[self.box_body][:2] = box_xy

        c = self.cube_qadr
        d.qpos[c:c + 3] = [cube_xy[0], cube_xy[1], self.cube_half + 0.0005]
        d.qpos[c + 3:c + 7] = yaw_quat(cube_yaw)
        # Drawn after the layout, so a seed gives the same layout at any spread.
        self.start = self._sample_start(level)
        d.qpos[self.qadr] = self.start
        d.ctrl[self.act] = self.start
        mujoco.mj_forward(m, d)
        self.hold(0.3)                       # let the cube and the arm settle
        d.time = 0.0
        self.step_count = 0
        return self.observe(render)

    def _sample_layout(self, level: str):
        t = self.cfg.task
        if level == "none":
            return np.array([0.21, -0.06]), 0.0, np.array([0.20, 0.16])
        rng = self.rng
        side = rng.choice([-1.0, 1.0])
        box_xy = np.array([rng.uniform(*t.box_x), side * rng.uniform(*t.box_abs_y)])
        keep_out = self.box_outer_half + self.cube_half * math.sqrt(2) + t.clearance
        for _ in range(200):
            cube_xy = np.array([rng.uniform(*t.cube_x), rng.uniform(*t.cube_y)])
            if np.max(np.abs(cube_xy - box_xy)) > keep_out:
                break
        else:   # fall back to the far side of the spawn area
            cube_xy = np.array([np.mean(t.cube_x), -side * max(abs(v) for v in t.cube_y)])
        yaw = math.radians(rng.uniform(*t.cube_yaw_deg))
        return cube_xy, yaw, box_xy

    def _sample_start(self, level: str) -> np.ndarray:
        """Start pose near home: jaws well over the table, arm clear of cube and box."""
        spread = 0.0 if level == "none" else self.cfg.start_spread
        if spread <= 0:
            return self.home.copy()
        m, d = self.model, self.data
        home = self.jmap.to_lerobot(self.home).astype(np.float64)
        lo, hi = self.jmap.to_lerobot(self.ctrl_lo), self.jmap.to_lerobot(self.ctrl_hi)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)     # a mirrored joint map swaps them
        width = spread * np.asarray(START_SPREAD)
        for _ in range(100):
            q = self.jmap.to_sim(np.clip(home + self.rng.uniform(-width, width), lo, hi))
            d.qpos[self.qadr] = q
            mujoco.mj_forward(m, d)
            if self.tcp_pos()[2] >= START_MIN_HEIGHT and not self._arm_touches_something():
                return q
        return self.home.copy()

    def _arm_touches_something(self) -> bool:
        fixed = self._fixed_bodies
        body = self.model.geom_bodyid
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if body[c.geom1] not in fixed or body[c.geom2] not in fixed:
                return True
        return False

    def _randomize_appearance(self, level: str) -> None:
        m, rng = self.model, self.rng
        m.geom_rgba[:] = self._default_rgba
        m.mat_rgba[:] = self._default_mat_rgba
        diffuse, lpos, ldir = self._default_light
        m.light_diffuse[:] = diffuse
        m.light_pos[:] = lpos
        m.light_dir[:] = ldir
        m.body_mass[self.cube_body] = self._default_cube_mass
        m.body_inertia[self.cube_body] = self._default_cube_inertia
        m.geom_friction[self.cube_geom] = self._default_cube_friction
        pos, target, fovy = self.cfg.camera1_pos, self.cfg.camera1_target, self.cfg.camera1_fovy

        if level == "light":
            m.light_diffuse[:] = diffuse * rng.uniform(0.85, 1.15)
        elif level == "full":
            m.light_diffuse[:] = diffuse * rng.uniform(0.6, 1.3, size=(m.nlight, 1))
            m.light_pos[:] = lpos + rng.uniform(-0.15, 0.15, size=lpos.shape)
            m.light_dir[:] = ldir + rng.uniform(-0.15, 0.15, size=ldir.shape)
            m.mat_rgba[self._cube_mat, :3] = rng.uniform(0.05, 0.95, 3)
            m.mat_rgba[self._box_mat, :3] = np.clip(self._default_mat_rgba[self._box_mat, :3]
                                                    + rng.uniform(-0.15, 0.15, 3), 0, 1)
            m.mat_rgba[self._table_mat, :3] = np.clip(rng.uniform(0.6, 1.0) * np.ones(3)
                                                      + rng.uniform(-0.08, 0.08, 3), 0.4, 1.0)
            mass = rng.uniform(0.02, 0.05)
            m.body_mass[self.cube_body] = mass
            m.body_inertia[self.cube_body] = self._default_cube_inertia * mass / self._default_cube_mass
            m.geom_friction[self.cube_geom, 0] = rng.uniform(0.7, 1.2)
            pos = tuple(np.asarray(pos) + rng.uniform(-0.02, 0.02, 3))
            target = tuple(np.asarray(target) + rng.uniform(-0.02, 0.02, 3))
            fovy = fovy + rng.uniform(-3, 3)
        self._aim_camera1(pos, target, fovy)

    def _aim_camera1(self, pos, target, fovy) -> None:
        self.model.cam_pos[self.cam1] = pos
        self.model.cam_quat[self.cam1] = look_at_quat(pos, target)
        self.model.cam_fovy[self.cam1] = fovy
