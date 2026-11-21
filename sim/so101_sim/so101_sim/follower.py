"""Mirror the simulated arm onto a real SO-101 follower over USB.

Whatever drives the simulator (keyboard, sliders, the leader arm, the expert, a
policy) also drives the real follower: each simulator step, the follower gets the
same joint targets, in the same LeRobot units. With the leader arm as the control
mode this is teleop through the simulator; with the expert or a policy it shows
on the real arm what a run in the simulator asked for, cube or no cube.

Uses LeRobot's own SO101Follower driver, so the follower must already be
calibrated on this computer with

    lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id=<ID>

Safety, in this order:

- no target moves more than max_step_deg from the last one per step (3 degrees,
  as MAX_STEP in hardware/so101_lab.sh);
- after a jump (on connect, or when a new scene puts the simulated arm back at
  its start pose) the follower catches up at catch_up_deg per step instead,
  about 30 degrees a second, until it has reached the simulated arm;
- LeRobot's max_relative_target caps every target against the arm's present
  position as well.

Stopping the mirror disconnects the follower, and LeRobot turns its torque off as
it does: the arm goes limp. Support it, or bring it to rest first.
"""

from __future__ import annotations

import numpy as np

from .config import FEATURE_NAMES

MAX_STEP_DEG = 3.0       # per step while following, as MAX_STEP in so101_lab.sh
CATCH_UP_DEG = 1.0       # per step while catching up after a jump
JUMP_DEG = 10.0          # a simulated target this far from the last one is a jump


def _follower_classes():
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        try:   # LeRobot versions with a separate so101_follower package
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError as e:
            raise RuntimeError("The follower mirror needs LeRobot with the Feetech extra: "
                               "pip install 'lerobot[feetech]>=0.4.4'") from e
    return SO101Follower, SO101FollowerConfig


class FollowerMirror:
    def __init__(self, port: str, follower_id: str, max_step_deg: float = MAX_STEP_DEG,
                 catch_up_deg: float = CATCH_UP_DEG, robot=None):
        if robot is None:
            SO101Follower, SO101FollowerConfig = _follower_classes()
            robot = SO101Follower(SO101FollowerConfig(port=port, id=follower_id,
                                                      max_relative_target=max_step_deg))
            if not robot.calibration:           # loaded from the calibration file, if there is one
                raise RuntimeError(
                    f"No calibration for follower '{follower_id}' on this computer. Run:\n"
                    f"  lerobot-calibrate --robot.type=so101_follower --robot.port={port} "
                    f"--robot.id={follower_id}")
            robot.connect(calibrate=False)
            if not robot.is_calibrated:         # same as pressing ENTER at lerobot-calibrate's prompt
                robot.bus.write_calibration(robot.calibration)
        self.dev = robot
        self.port, self.id = port, follower_id
        self.max_step, self.catch_up = float(max_step_deg), float(catch_up_deg)
        self.sent = self.read()                 # start from where the real arm is
        self.last_want: np.ndarray | None = None
        self.catching_up = True
        self.gap = 0.0                          # largest joint gap to the simulator, LeRobot units

    def read(self) -> np.ndarray:
        obs = self.dev.get_observation()
        return np.array([obs[k] for k in FEATURE_NAMES], dtype=np.float64)

    def send(self, action: np.ndarray) -> np.ndarray:
        """Move one capped step toward the simulator's joint targets. Returns the target sent."""
        want = np.asarray(action, dtype=np.float64)
        if self.last_want is not None and np.max(np.abs(want - self.last_want)) > JUMP_DEG:
            self.catching_up = True
        self.last_want = want
        step = self.catch_up if self.catching_up else self.max_step
        target = self.sent + np.clip(want - self.sent, -step, step)
        self.gap = float(np.max(np.abs(want - target)))
        if self.gap < 1e-6:
            self.catching_up = False            # caught up: follow at the normal rate
        self.dev.send_action({k: float(v) for k, v in zip(FEATURE_NAMES, target)})
        self.sent = target
        return target

    def close(self) -> None:
        try:
            self.dev.disconnect()
        except Exception:
            pass
