"""Tests of the gripper height check (S-11): the arm model, the table fit and the rule."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot_robot_so101_safe.kinematics import ARM, FIT_JOINTS, fit_table, joint_points, lowest_z
from lerobot_robot_so101_safe.rules import SafetyConfig, check_height, filter_action

# Model angles with the whole arm straight up (found with the same definition as the fit).
UPRIGHT_MODEL = {"shoulder_lift": -13.97, "elbow_flex": -73.82, "wrist_flex": -5.06}


def pose(**kw):
    p = {j: 0.0 for j in ARM} | {"gripper": 30.0}
    p.update(kw)
    return p


def test_upright_model_pose_is_vertical():
    pts = joint_points([0.0, *UPRIGHT_MODEL.values(), 0.0])
    base = pts["shoulder_lift"]
    fwd = np.cross([0, 0, 1], pts["shoulder_lift_axis"])
    fwd /= np.linalg.norm(fwd)
    for k in ("elbow_flex", "wrist_flex", "fixed_jaw_tip"):
        assert abs(np.dot(pts[k] - base, fwd)) < 1e-3
    assert pts["fixed_jaw_tip"][2] > 0.45  # about 53 cm up


def test_fit_recovers_known_offsets_and_table():
    true = {"shoulder_lift": -12.0, "elbow_flex": 8.0, "wrist_flex": -6.0}
    table = 0.03
    rng = np.random.default_rng(0)
    touches = []
    while len(touches) < 6:
        p = {"shoulder_pan": rng.uniform(-30, 30), "shoulder_lift": rng.uniform(10, 45),
             "wrist_flex": rng.uniform(-20, 40), "wrist_roll": 0.0}

        def h(e):
            deg = [p["shoulder_pan"], p["shoulder_lift"] + true["shoulder_lift"], e + true["elbow_flex"],
                   p["wrist_flex"] + true["wrist_flex"], 0.0]
            return joint_points(deg)["fixed_jaw_tip"][2] - table
        lo, hi = -80.0, 90.0
        if h(lo) * h(hi) > 0:
            continue
        for _ in range(60):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if h(lo) * h(m) > 0 else (lo, m)
        p["elbow_flex"] = 0.5 * (lo + hi)
        touches.append(p)
    upright = {"shoulder_pan": 5.0, "wrist_roll": 0.0} | {j: UPRIGHT_MODEL[j] - true[j] for j in FIT_JOINTS}
    off, tz, res = fit_table(touches, upright)
    for j in FIT_JOINTS:
        assert off[j] == pytest.approx(true[j], abs=0.1)
    assert tz == pytest.approx(table, abs=5e-4)
    assert np.abs(res).max() < 1e-3


def cfg(table_z):
    return SafetyConfig(height_check=True, table_z_m=table_z, table_margin_m=0.01, model_offsets_deg={})


def test_height_allows_goals_above_the_floor():
    high = pose(shoulder_lift=-14, elbow_flex=-74)            # straight up, tip about 53 cm
    goal = dict(high, elbow_flex=-71)
    out, ev = check_height(goal, high, high, cfg(0.0))
    assert out == goal and ev == []


def test_height_holds_a_goal_that_goes_into_the_table():
    present = pose(shoulder_lift=30, elbow_flex=10)
    table = lowest_z(present, {}) - 0.012                     # floor 2 mm under the arm
    goal = dict(present, shoulder_lift=33, elbow_flex=13)      # tip and elbow go lower
    assert lowest_z(goal, {}) < table + 0.01
    out, ev = check_height(goal, present, present, cfg(table))
    assert ev == [("clip_height", "arm")]
    assert out["shoulder_lift"] == 30 and out["elbow_flex"] == 10


def test_height_lets_the_arm_lift_off_the_table():
    low = pose(shoulder_lift=30, elbow_flex=20)
    table = lowest_z(low, {})                                  # the arm is already at the table
    goal = dict(low, shoulder_lift=27)                         # tips back, gripper goes up
    assert lowest_z(goal, {}) > lowest_z(low, {})
    out, ev = check_height(goal, low, low, cfg(table))
    assert out == goal and ev == []


def test_height_lets_the_base_turn_near_the_table():
    low = pose(shoulder_lift=30, elbow_flex=20)
    table = lowest_z(low, {})
    goal = dict(low, shoulder_pan=3, shoulder_lift=33)         # turn, and also try to go lower
    out, ev = check_height(goal, low, low, cfg(table))
    assert ev == [("clip_height", "arm")]
    assert out["shoulder_pan"] == 3 and out["shoulder_lift"] == 30


def test_filter_action_runs_the_height_check():
    present = pose(shoulder_lift=30, elbow_flex=10)
    table = lowest_z(present, {}) - 0.012
    limits = {j: (-150.0, 150.0) for j in ARM} | {"gripper": (0.0, 100.0)}
    c = cfg(table)
    action = {f"{j}.pos": v for j, v in present.items()} | {"shoulder_lift.pos": 33.0}
    goal, ev = filter_action(action, present, limits, c, last_goal=present)
    assert ("clip_height", "arm") in ev and goal["shoulder_lift"] == 30


def test_height_off_by_default():
    assert SafetyConfig().height_check is False
