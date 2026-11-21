"""Inverse kinematics for the 5-DoF SO-101.

The arm has five joints, so it cannot reach an arbitrary 6-D pose. What it can
reach is a tool position, the tool's pitch in the arm's vertical plane, and a roll
about the tool axis. The solver therefore splits the problem:

- shoulder_pan, shoulder_lift, elbow_flex and wrist_flex move the tool point to a
  target position at a target pitch (damped least squares, 4 equations in 4
  unknowns);
- wrist_roll is set directly, either given or chosen to line the jaws up with a
  horizontal direction (for a top-down grasp).

Pitch is measured below the horizontal: 90 degrees points the jaws straight down.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .env import TCP_OFFSET

PAN, LIFT, ELBOW, WFLEX, ROLL, GRIP = range(6)


class ArmIK:
    def __init__(self, model: mujoco.MjModel, qadr: np.ndarray, dadr: np.ndarray):
        self.m = model
        self.d = mujoco.MjData(model)          # scratch data, never stepped
        self.qadr, self.dadr = qadr, dadr
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.body = model.site_bodyid[self.site]
        jids = [model.dof_jntid[a] for a in dadr]
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self._roll_sign = self._measure_roll_sign()

    # ------------------------------------------------------------------ helpers

    def _fk(self, q: np.ndarray):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3).copy()
        p = self.d.site_xpos[self.site] + R @ TCP_OFFSET
        return p, R

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool point (m) and site rotation for joint angles q (radians, 6 values)."""
        return self._fk(np.asarray(q, float))

    @staticmethod
    def pitch_of(R: np.ndarray, radial: np.ndarray) -> float:
        a = R[:, 0]                             # approach axis: out of the jaws
        return math.atan2(-a[2], float(a[:2] @ radial))

    def _radial(self, p: np.ndarray) -> np.ndarray:
        r = p[:2].copy()
        n = np.linalg.norm(r)
        return r / n if n > 1e-6 else np.array([1.0, 0.0])

    def _measure_roll_sign(self) -> float:
        """+1 if a positive wrist_roll turns the jaw axis counter-clockwise about the approach axis."""
        q = np.zeros(6)
        q[WFLEX] = 1.2
        _, R0 = self._fk(q)
        q[ROLL] = 0.1
        _, R1 = self._fk(q)
        a, j0, j1 = R0[:, 0], R0[:, 2], R1[:, 2]
        return 1.0 if float(np.cross(j0, j1) @ a) > 0 else -1.0

    # ------------------------------------------------------------------ solvers

    def solve(self, q_init: np.ndarray, target: np.ndarray, pitch: float,
              iters: int = 60, tol: float = 5e-4, damping: float = 0.02,
              pitch_weight: float = 0.15) -> tuple[np.ndarray, float]:
        """Move the tool point to target at the given pitch. Roll and gripper are kept.

        Returns (q, position error in metres).
        """
        q = np.array(q_init, dtype=float)
        idx = [PAN, LIFT, ELBOW, WFLEX]
        cols = self.dadr[idx]
        err = np.inf
        for _ in range(iters):
            p, R = self._fk(q)
            radial = self._radial(p)
            e_pos = target - p
            e_pitch = math.atan2(math.sin(pitch - self.pitch_of(R, radial)),
                                 math.cos(pitch - self.pitch_of(R, radial)))
            err = float(np.linalg.norm(e_pos))
            if err < tol and abs(e_pitch) < 2e-3:
                break
            mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, p, self.body)
            Jp = self.jacp[:, cols]
            # d(pitch)/dq from the angular Jacobian: the approach axis turns by w x a.
            a = R[:, 0]
            u, v = -a[2], float(a[:2] @ radial)
            da = np.cross(self.jacr[:, cols].T, a).T            # 3 x 4
            du, dv = -da[2], radial @ da[:2]
            Jt = (v * du - u * dv) / max(u * u + v * v, 1e-9)
            Jt[0] = 0.0                                       # pan does not change pitch
            J = np.vstack([Jp, pitch_weight * Jt])
            e = np.concatenate([e_pos, [pitch_weight * e_pitch]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(4), e)
            step = np.linalg.norm(dq)
            if step > 0.25:
                dq *= 0.25 / step
            q[idx] = np.clip(q[idx] + dq, self.lo[idx], self.hi[idx])
        return q, err

    def solve_steepest(self, q_init: np.ndarray, target: np.ndarray, pitch: float = math.pi / 2,
                       min_pitch: float = math.radians(30), tol: float = 1e-3,
                       iters: int = 10) -> tuple[np.ndarray, float, float]:
        """Reach target exactly at the steepest pitch up to `pitch` the joint limits allow.

        Pointing the jaws straight down at a point high above, or far from, the base
        needs more wrist_flex than the joint has. solve() then returns a compromise
        that misses the position by centimetres. This tilts the jaws instead: it
        lowers the pitch in small steps, each solve starting from the previous one
        so the arm stays in the same configuration (a cold solve at a shallow pitch
        can flip to another one, e.g. shoulder_lift at its limit), then bisects the
        last step for the steepest pitch with a position error under tol.

        Returns (q, pitch, position error). If even min_pitch cannot reach the
        target, returns the closest attempt.
        """
        q, err = self.solve(q_init, target, pitch)
        if err < tol:
            return q, pitch, err
        best = (q, pitch, err)
        hi, q_hi = pitch, q
        p = pitch
        while p > min_pitch + 1e-9:
            p = max(p - math.radians(5), min_pitch)
            q_p, err_p = self.solve(q_hi, target, p)
            if err_p < tol:
                lo, best = p, (q_p, p, err_p)
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    q_mid, err_mid = self.solve(q_hi, target, mid)
                    if err_mid < tol:
                        lo, best = mid, (q_mid, mid, err_mid)
                    else:
                        hi = mid
                return best
            if err_p < best[2]:
                best = (q_p, p, err_p)
            hi, q_hi = p, q_p
        return best

    def roll_for_jaw_axis(self, q: np.ndarray, direction: np.ndarray,
                          symmetric: int = 4, prefer: float | None = None) -> float:
        """wrist_roll that lines the jaw axis up with a horizontal direction.

        symmetric=4 treats the direction as a cube face normal, so any of the four
        face normals will do; the one closest to `prefer` (default: current roll)
        and inside the joint limits wins.
        """
        q = np.array(q, dtype=float)
        _, R = self._fk(q)
        a, jaw = R[:, 0], R[:, 2]
        d = np.asarray(direction, float)
        d = d - (d @ a) * a
        d /= max(np.linalg.norm(d), 1e-9)
        prefer = q[ROLL] if prefer is None else prefer
        best, best_cost = q[ROLL], np.inf
        for k in range(symmetric):
            ang = 2 * math.pi * k / symmetric
            dk = d * math.cos(ang) + np.cross(a, d) * math.sin(ang)
            phi = math.atan2(float(np.cross(jaw, dk) @ a), float(jaw @ dk))
            for wrap in (-2 * math.pi, 0.0, 2 * math.pi):
                cand = q[ROLL] + self._roll_sign * phi + wrap
                if self.lo[ROLL] <= cand <= self.hi[ROLL] and abs(cand - prefer) < best_cost:
                    best, best_cost = cand, abs(cand - prefer)
        return best

    def solve_topdown(self, q_init: np.ndarray, target: np.ndarray, jaw_dir: np.ndarray | None,
                      roll_prefer: float | None = None, iters: int = 3,
                      symmetric: int = 4) -> tuple[np.ndarray, float]:
        """Jaws pointing straight down at target; roll set from jaw_dir if given.

        symmetric is how many jaw directions are equivalent (4 for a cube, 2 for a block).
        """
        q = np.array(q_init, dtype=float)
        err = np.inf
        for _ in range(iters):
            q, err = self.solve(q, target, math.pi / 2)
            if jaw_dir is None:
                break
            q[ROLL] = self.roll_for_jaw_axis(q, jaw_dir, symmetric=symmetric, prefer=roll_prefer)
        q, err = self.solve(q, target, math.pi / 2)
        return q, err
