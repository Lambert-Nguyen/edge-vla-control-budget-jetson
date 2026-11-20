"""Where the SO-101 gripper is, from the joint angles. Plain NumPy, no simulator.

The link lengths and joint frames come from the SO-101 model that Lam's
simulator uses (sim/so101_sim/.../so101.xml, from the MuJoCo Menagerie model by
TheRobotStudio). The model's frame has z up and its origin under the base, at
the bottom of the arm.

LeRobot measures each joint from the middle of its calibrated range, which is
close to the model's zero, but not the same on every arm. So:

    model angle = LeRobot degrees + offset_deg[joint]

The offsets and the table height come from the table calibration
(hardware/table_calibration.py), one set per arm.
"""

from __future__ import annotations

import numpy as np

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# (position, quaternion w x y z) of each body in its parent, from so101.xml.
# Every joint turns about the z axis of its own body.
_CHAIN = [
    ((0.0388353, 0.0, 0.0624), (0.0, 0.0, -1.0, 0.0)),                                   # shoulder (pan)
    ((-0.0303992, -0.0182778, -0.0542), (1.0, -1.0, -1.0, -1.0)),                        # upper_arm (lift)
    ((-0.11257, -0.028, 0.0), (1.0, 0.0, 0.0, 1.0)),                                     # lower_arm (elbow)
    ((-0.1349, 0.0052, 0.0), (1.0, 0.0, 0.0, -1.0)),                                     # wrist (flex)
    ((5.55112e-17, -0.0611, 0.0181), (0.0172091, -0.0172091, 0.706897, 0.706897)),       # gripper (roll)
]
# Points that can touch the table, in the gripper frame.
GRIPPER_POINTS = {
    "fixed_jaw_tip": (-0.0081, 0.0, -0.101),
    "gripper_body": (0.0, 0.0, -0.04),
    "camera_mount": (-0.001, 0.06, -0.04),
}
# The moving jaw: its body in the gripper frame, then its tip in its own frame.
_MOVING_JAW = ((0.0202, 0.0188, -0.0234), (1.0, 1.0, 0.0, 0.0))
_MOVING_JAW_TIP = (-0.012, -0.078, 0.0192)
GRIPPER_RAD = (-0.174533, 1.7453292)  # model range of the gripper joint: closed .. open


def _quat(w, x, y, z) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_CHAIN_R = [(np.array(p), _quat(*q)) for p, q in _CHAIN]
_JAW_R = (np.array(_MOVING_JAW[0]), _quat(*_MOVING_JAW[1]))


def joint_points(model_deg) -> dict:
    """Positions (m) of each joint and the gripper points, for model angles in degrees.

    model_deg: the five arm joints, then optionally the gripper (0-100, LeRobot units).
    """
    R, p, out = np.eye(3), np.zeros(3), {}
    for name, (pos, rot), a in zip(ARM, _CHAIN_R, np.radians(np.asarray(model_deg[:5], dtype=float))):
        p = p + R @ pos
        R = R @ rot
        out[name] = p.copy()
        out[name + "_axis"] = R[:, 2].copy()  # the joint turns about this axis
        R = R @ _rotz(a)
    for name, q in GRIPPER_POINTS.items():
        out[name] = p + R @ np.array(q)
    grip = float(model_deg[5]) if len(model_deg) > 5 else 0.0
    ga = GRIPPER_RAD[0] + (min(max(grip, 0.0), 100.0) / 100.0) * (GRIPPER_RAD[1] - GRIPPER_RAD[0])
    jaw_p = p + R @ _JAW_R[0]
    jaw_R = R @ _JAW_R[1] @ _rotz(ga)
    out["moving_jaw_tip"] = jaw_p + jaw_R @ np.array(_MOVING_JAW_TIP)
    return out


TABLE_POINTS = ("fixed_jaw_tip", "moving_jaw_tip", "gripper_body", "camera_mount", "wrist_roll")


def lowest_z(lerobot: dict, offsets_deg: dict) -> float:
    """Height (m, model frame) of the lowest gripper point, for a LeRobot pose dict."""
    deg = [lerobot[j] + offsets_deg.get(j, 0.0) for j in ARM] + [lerobot.get("gripper", 0.0)]
    pts = joint_points(deg)
    return min(float(pts[k][2]) for k in TABLE_POINTS)


FIT_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")


def _residuals(params: np.ndarray, touches: list[dict], upright: dict | None) -> np.ndarray:
    off = dict(zip(FIT_JOINTS, params[:3]))
    table = params[3]
    res = []
    for t in touches:
        deg = [t[j] + off.get(j, 0.0) for j in ARM]
        res.append(joint_points(deg)["fixed_jaw_tip"][2] - table)
    if upright is not None:
        deg = [upright[j] + off.get(j, 0.0) for j in ARM]
        pts = joint_points(deg)
        # Straight up means: elbow, wrist and tip are right above the shoulder_lift
        # joint, seen along the arm's forward direction (perpendicular to that
        # joint's axis). Sideways the links are offset by design, so not that way.
        base = pts["shoulder_lift"]
        fwd = np.cross([0.0, 0.0, 1.0], pts["shoulder_lift_axis"])
        fwd = fwd / np.linalg.norm(fwd)
        for k in ("elbow_flex", "wrist_flex", "fixed_jaw_tip"):
            res.append(float(np.dot(pts[k] - base, fwd)))
    return np.array(res, dtype=float)


def fit_table(touches: list[dict], upright: dict | None, iters: int = 50):
    """Fit the angle offsets of shoulder_lift, elbow_flex, wrist_flex and the table height.

    touches: LeRobot poses with the fixed jaw tip touching the table.
    upright: the LeRobot pose with the whole arm straight up (or None).
    Returns (offsets_deg, table_z_m, residuals_m). Gauss-Newton, no SciPy.
    """
    p = np.zeros(4)
    p[3] = float(np.mean([joint_points([t[j] for j in ARM])["fixed_jaw_tip"][2] for t in touches]))
    for _ in range(iters):
        r = _residuals(p, touches, upright)
        J = np.empty((len(r), 4))
        for i, h in enumerate((0.01, 0.01, 0.01, 1e-5)):
            dp = np.zeros(4)
            dp[i] = h
            J[:, i] = (_residuals(p + dp, touches, upright) - r) / h
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + step
        if np.abs(step[:3]).max() < 1e-4 and abs(step[3]) < 1e-7:
            break
    r = _residuals(p, touches, upright)
    return {j: float(v) for j, v in zip(FIT_JOINTS, p[:3])}, float(p[3]), r
