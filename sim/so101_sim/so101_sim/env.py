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
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from . import objects
from .config import JOINT_NAMES, SCENE_XML, SCENES, START_MIN_HEIGHT, START_SPREAD, SimConfig
from .noise import EpisodeNoise
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

# Extra objects in the scene XML for the multi-object scenes.
DISTRACTOR_NAMES = ("distractor1", "distractor2")
# Closest two objects' footprints may come, metres: room for the open jaws (about
# 3 cm either side of the object to pick up) to come down without touching another.
OBJECT_GAP = 0.035
# Where distractors may stand, metres: wider than the pick-up object's spawn area,
# still within the arm's reach and both cameras' view.
DISTRACTOR_X = (0.13, 0.28)
DISTRACTOR_Y = (-0.15, 0.15)


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

        # The object this episode uses. With objects ("cube",) the scene's own 3 cm
        # cube is never touched, so datasets made before objects existed regenerate
        # exactly; otherwise reset() reshapes the object geom (objects.py).
        unknown = set(self.cfg.objects) - set(objects.SIZE_RANGES)
        if unknown or not self.cfg.objects:
            raise ValueError(f"unknown objects {sorted(unknown)}; choose from {', '.join(objects.SIZE_RANGES)}")
        if self.cfg.scene not in SCENES:
            raise ValueError(f"unknown scene {self.cfg.scene!r}; choose from {', '.join(SCENES)}")
        self.original_cube = tuple(self.cfg.objects) == ("cube",) and self.cfg.scene == "single"
        self.obj = replace(objects.make_object("cube", [self.cube_half]),
                           mass=float(m.body_mass[self.cube_body]),
                           inertia=tuple(float(v) for v in m.body_inertia[self.cube_body]))
        self.scene = "single" if self.cfg.scene == "mixed" else self.cfg.scene   # this episode's
        self.target_label = "cube"          # how the instruction names the object to pick up
        self.target_color: str | None = None

        # Distractors for the "colors" and "kinds" scenes, parked under the table
        # (weightless, no contacts) until placed.
        self.distractors = []
        for name in DISTRACTOR_NAMES:
            b = name2id(mujoco.mjtObj.mjOBJ_BODY, name)
            j = m.body_jntadr[b]
            self.distractors.append({"body": b, "geom": name2id(mujoco.mjtObj.mjOBJ_GEOM, name),
                                     "qadr": m.jnt_qposadr[j],
                                     "mat": name2id(mujoco.mjtObj.mjOBJ_MATERIAL, name),
                                     "spec": None, "color": None})

        # Noise (noise.py) perturbs the actuator gains and the wrist camera mount; keep
        # the model's values to restore them.
        self.noise: EpisodeNoise | None = None
        self._default_kp = m.actuator_gainprm[self.act, 0].copy()
        self._wrist_cam = self.cam_ids["camera2"]
        self._default_wrist_cam = (m.cam_pos[self._wrist_cam].copy(), m.cam_quat[self._wrist_cam].copy())

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
        self._fixed_bodies = {0, self.box_body, self.cube_body} | {d["body"] for d in self.distractors}
        self.step_count = 0
        self.instruction = self.cfg.task.instruction
        self.task = self.instruction
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
        """Joint readings in LeRobot units, with encoder noise when noise is on."""
        if self.noise is None:
            return self.jmap.to_lerobot(self.qpos_arm())
        reading = self.noise.quantize(self.jmap.to_lerobot(self.noise.reading(self.qpos_arm())))
        # Same dtype as the clean path: MPS (Apple GPUs) has no float64 tensors.
        return reading.astype(self.jmap.to_lerobot(self.home).dtype)

    def action_for(self, q: np.ndarray) -> np.ndarray:
        """The action (LeRobot units) that puts the arm at joint angles q (sim radians).

        With noise on, this undoes the calibration offset, as a person steering by
        eye would; backlash and jitter remain. The scripted expert acts through it.
        """
        if self.noise is not None:
            q = self.noise.compensate(q)
        return self.jmap.to_lerobot(q)

    def set_instruction(self, text: str) -> None:
        """Instruction for the task; 'the cube' in it becomes this episode's object."""
        self.instruction = text
        self.task = self._task_text()

    def _task_text(self) -> str:
        if self.original_cube:
            return self.instruction
        return self.instruction.replace("the cube", f"the {self.target_label}")

    def observe(self, render: bool | None = None) -> Obs:
        render = self.render_enabled if render is None else render
        images = {}
        if render:
            for k in self.cfg.cameras:
                img = self.render_camera(k)
                images[k] = img if self.noise is None else self.noise.image(k, img)
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
        # The original cube keeps its original check; a long object lying in the box
        # can sit further off centre, so other objects use their narrow side.
        half = self.cube_half if self.original_cube else self.obj.grip_half
        margin = self.box_inner_half - 0.5 * half
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
            # Like max_relative_target: relative to the position the arm reports.
            cur = self.qpos_arm() if self.noise is None else self.qpos_arm() - self.noise.offset
            lim = math.radians(self.cfg.max_step_deg)
            target = np.clip(target, cur - lim, cur + lim)
        if self.noise is not None:
            target = self.noise.command(target)
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
        # The object and the noise come from their own generators, seeded from the
        # episode seed, so the layout drawn from self.rng is the same whatever
        # objects or noise scale are chosen.
        base = 0
        if not self.original_cube or self.cfg.noise > 0:
            base = int(seed) if seed is not None else int(self.rng.integers(2 ** 31))
        mass_before = float(m.body_mass[self.cube_body])
        changed = False
        if not self.original_cube:
            orng = np.random.default_rng([base, 1])
            self.scene = self.cfg.scene
            if self.scene == "mixed":
                self.scene = str(orng.choice(["single", "colors", "kinds"]))
            objs, colors, self.target_label = objects.sample_scene(
                self.scene, self.cfg.objects, orng, vary_size=level != "none")
            self.obj, self.target_color = objs[0], colors[0]
            objects.apply(m, self.cube_body, self.cube_geom, self.obj)
            for i, dist in enumerate(self.distractors):
                k = i + 1
                dist["spec"], dist["color"] = (objs[k], colors[k]) if k < len(objs) else (None, None)
                if dist["spec"] is not None:
                    objects.apply(m, dist["body"], dist["geom"], dist["spec"])
                self._set_active(dist, dist["spec"] is not None)
            changed = True
        self._randomize_appearance(level)
        if not self.original_cube and self.scene != "single":
            crng = np.random.default_rng([base, 4])
            jitter = 0.05 if level == "full" else 0.0
            m.mat_rgba[self._cube_mat, :3] = objects.color_rgb(self.target_color, crng, jitter)
            for dist in self.distractors:
                if dist["color"] is not None:
                    m.mat_rgba[dist["mat"], :3] = objects.color_rgb(dist["color"], crng, jitter)
        changed |= self._apply_noise(np.random.default_rng([base, 2]))
        if changed or float(m.body_mass[self.cube_body]) != mass_before:
            # Masses and shapes feed constants the solver precomputes at compile time
            # (invweight0 and friends). Recompute them, or a lighter cube makes the
            # contact solver blow up. mj_setConst uses d as scratch, so call it
            # before the episode state is written.
            mujoco.mj_setConst(m, d)
        self.task = self._task_text()
        mujoco.mj_resetData(m, d)
        cube_xy, cube_yaw, box_xy = self._sample_layout(level)
        m.body_pos[self.box_body][:2] = box_xy

        c = self.cube_qadr
        d.qpos[c:c + 3] = [cube_xy[0], cube_xy[1], self.obj.half_height + 0.0005]
        d.qpos[c + 3:c + 7] = yaw_quat(cube_yaw)
        if not self.original_cube:
            self._place_distractors(np.random.default_rng([base, 3]), cube_xy, box_xy)
        # Drawn after the layout, so a seed gives the same layout at any spread.
        self.start = self._sample_start(level)
        d.qpos[self.qadr] = self.start
        d.ctrl[self.act] = self.start
        if self.noise is not None:
            self.noise._held = self.start.copy()     # the gears rest at the start pose
        mujoco.mj_forward(m, d)
        self.hold(0.3)                       # let the cube and the arm settle
        d.time = 0.0
        self.step_count = 0
        return self.observe(render)

    def _set_active(self, dist: dict, active: bool) -> None:
        """Bring a distractor into the scene, or park it: weightless, without contacts."""
        m = self.model
        flag = int(m.geom_contype[self.cube_geom]) if active else 0
        m.geom_contype[dist["geom"]] = flag
        m.geom_conaffinity[dist["geom"]] = int(m.geom_conaffinity[self.cube_geom]) if active else 0
        m.body_gravcomp[dist["body"]] = 0.0 if active else 1.0

    def _place_distractors(self, rng: np.random.Generator, target_xy: np.ndarray, box_xy: np.ndarray) -> None:
        """Put the active distractors on the table, clear of the box and each other.

        They spread over a wider area than the object to pick up (which keeps its
        original spawn area and layout). A distractor that finds no room is parked.
        """
        t = self.cfg.task
        placed = [(np.asarray(target_xy), self.obj.footprint)]
        d = self.data
        for dist in self.distractors:
            spec = dist["spec"]
            if spec is None:
                continue
            keep_out = self.box_outer_half + spec.footprint + t.clearance
            for _ in range(400):
                xy = np.array([rng.uniform(*DISTRACTOR_X), rng.uniform(*DISTRACTOR_Y)])
                if np.max(np.abs(xy - box_xy)) <= keep_out:
                    continue
                if all(np.linalg.norm(xy - p) >= spec.footprint + f + OBJECT_GAP for p, f in placed):
                    break
            else:
                dist["spec"] = dist["color"] = None
                self._set_active(dist, False)
                continue
            yaw = math.radians(rng.uniform(*t.cube_yaw_deg)) + (math.pi / 2 if spec.symmetry == 2 else 0.0)
            q = dist["qadr"]
            d.qpos[q:q + 3] = [xy[0], xy[1], spec.half_height + 0.0005]
            d.qpos[q + 3:q + 7] = yaw_quat(yaw)
            placed.append((xy, spec.footprint))

    def distractor_in_box(self) -> bool:
        """True if any other object ended up in the box: the wrong one was picked."""
        box = self.box_pos()
        for dist in self.distractors:
            if dist["spec"] is None:
                continue
            pos = self.data.xpos[dist["body"]]
            rel = pos - box
            if abs(rel[0]) < self.box_inner_half and abs(rel[1]) < self.box_inner_half and pos[2] < self.box_height:
                return True
        return False

    def episode_info(self) -> dict:
        """What this episode put on the table, for dataset logs and eval records."""
        info = {"scene": self.scene, "target": self.target_label, "task": self.task,
                "object": {"kind": self.obj.kind, "size_m": [round(v, 4) for v in self.obj.size if v]}}
        others = [{"kind": dist["spec"].kind, "color": dist["color"]} for dist in self.distractors if dist["spec"]]
        if others:
            info["distractors"] = others
        if self.noise is not None:
            info["noise"] = self.noise.summary()
        return info

    def _apply_noise(self, rng: np.random.Generator) -> bool:
        """Draw this episode's noise, or restore the perfect robot. True if anything changed."""
        m = self.model
        was_noisy = self.noise is not None
        m.actuator_gainprm[self.act, 0] = self._default_kp
        m.actuator_biasprm[self.act, 1] = -self._default_kp
        m.cam_pos[self._wrist_cam], m.cam_quat[self._wrist_cam] = (v.copy() for v in self._default_wrist_cam)
        if self.cfg.noise <= 0:
            self.noise = None
            return was_noisy
        gripper_rad_per_unit = float(self.ctrl_hi[5] - self.ctrl_lo[5]) / 100.0
        self.noise = EpisodeNoise(self.cfg.noise, rng, self.cfg.cameras, gripper_rad_per_unit)
        kp = self._default_kp * self.noise.kp_scale
        m.actuator_gainprm[self.act, 0] = kp
        m.actuator_biasprm[self.act, 1] = -kp
        m.cam_pos[self._wrist_cam] += self.noise.wrist_cam_pos
        rot = self.noise.wrist_cam_rot
        angle = float(np.linalg.norm(rot))
        if angle > 0:
            dq = np.empty(4)
            mujoco.mju_axisAngle2Quat(dq, rot / angle, angle)
            q = np.empty(4)
            mujoco.mju_mulQuat(q, m.cam_quat[self._wrist_cam], dq)
            m.cam_quat[self._wrist_cam] = q
        return True

    def _sample_layout(self, level: str):
        t = self.cfg.task
        if level == "none":
            return np.array([0.21, -0.06]), 0.0, np.array([0.20, 0.16])
        rng = self.rng
        side = rng.choice([-1.0, 1.0])
        box_xy = np.array([rng.uniform(*t.box_x), side * rng.uniform(*t.box_abs_y)])
        keep_out = self.box_outer_half + self.obj.footprint + t.clearance
        for _ in range(200):
            cube_xy = np.array([rng.uniform(*t.cube_x), rng.uniform(*t.cube_y)])
            if np.max(np.abs(cube_xy - box_xy)) > keep_out:
                break
        else:   # fall back to the far side of the spawn area
            cube_xy = np.array([np.mean(t.cube_x), -side * max(abs(v) for v in t.cube_y)])
        yaw = math.radians(rng.uniform(*t.cube_yaw_deg))
        if self.obj.symmetry == 2:
            yaw += math.pi / 2          # long side across the reach (objects.py says why)
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
        base_mass, base_inertia = self._default_cube_mass, self._default_cube_inertia
        if not self.original_cube:
            base_mass, base_inertia = self.obj.mass, np.asarray(self.obj.inertia)
        m.body_mass[self.cube_body] = base_mass
        m.body_inertia[self.cube_body] = base_inertia
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
            # 20-50 g for the original 30 g cube; the same -33%/+67% for other objects.
            if self.original_cube:
                mass = rng.uniform(0.02, 0.05)
            else:
                mass = base_mass * rng.uniform(2 / 3, 5 / 3)
            m.body_mass[self.cube_body] = mass
            m.body_inertia[self.cube_body] = base_inertia * mass / base_mass
            m.geom_friction[self.cube_geom, 0] = rng.uniform(0.7, 1.2)
            pos = tuple(np.asarray(pos) + rng.uniform(-0.02, 0.02, 3))
            target = tuple(np.asarray(target) + rng.uniform(-0.02, 0.02, 3))
            fovy = fovy + rng.uniform(-3, 3)
        self._aim_camera1(pos, target, fovy)

    def _aim_camera1(self, pos, target, fovy) -> None:
        self.model.cam_pos[self.cam1] = pos
        self.model.cam_quat[self.cam1] = look_at_quat(pos, target)
        self.model.cam_fovy[self.cam1] = fovy
