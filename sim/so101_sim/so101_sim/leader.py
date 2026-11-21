"""Drive the simulated follower with a real SO-101 leader arm over USB.

Uses LeRobot's own SO101Leader driver, so the leader must already be calibrated
on this computer with

    lerobot-calibrate --teleop.type=so101_leader --teleop.port=<PORT> --teleop.id=<ID>

The leader reports LeRobot units (degrees, gripper 0-100), which are exactly the
simulator's action units, so its readings go straight to the simulated follower.

Ports: macOS /dev/tty.usbmodem..., Windows COM3 and the like, Linux /dev/ttyACM0.
`lerobot-find-port` tells you which one.
"""

from __future__ import annotations

import numpy as np

from .config import FEATURE_NAMES


class LeaderArm:
    def __init__(self, port: str, leader_id: str):
        try:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        except ImportError:
            try:   # LeRobot versions that used a separate so101_leader package
                from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
            except ImportError as e:
                raise RuntimeError("Leader-arm teleop needs LeRobot with the Feetech extra: "
                                   "pip install 'lerobot[feetech]>=0.4.4'") from e
        self.dev = SO101Leader(SO101LeaderConfig(port=port, id=leader_id))
        if not self.dev.calibration:            # loaded from the calibration file, if there is one
            raise RuntimeError(f"No calibration for leader '{leader_id}' on this computer. Run:\n"
                               f"  lerobot-calibrate --teleop.type=so101_leader --teleop.port={port} "
                               f"--teleop.id={leader_id}")
        self.dev.connect(calibrate=False)
        if not self.dev.is_calibrated:          # same as pressing ENTER at lerobot-calibrate's prompt
            self.dev.bus.write_calibration(self.dev.calibration)

    def read(self) -> np.ndarray:
        a = self.dev.get_action()
        return np.array([a[k] for k in FEATURE_NAMES], dtype=np.float32)

    def close(self) -> None:
        try:
            self.dev.disconnect()
        except Exception:
            pass
