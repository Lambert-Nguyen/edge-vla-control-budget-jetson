from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.robots.config import RobotConfig
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION


@RobotConfig.register_subclass("so101_safe_follower")
@dataclass
class SafeFollowerConfig(SOFollowerRobotConfig):
    # One YAML file per arm, in configs/safety/.
    safety_config: str = "configs/safety/lab_follower.yaml"
    # Overrides max_step_deg from the YAML file when set (MAX_STEP in so101_lab.sh).
    max_step_deg: float | None = None
    # Event log and counters go here, as <run>.jsonl and <run>.counters.json.
    log_dir: str = "logs/safety"
    run_name: str | None = None

    def __post_init__(self):
        super().__post_init__()
        # LeRobot looks for the calibration in a folder named after the robot
        # type. Use the normal follower's folder, so the same file is used.
        if self.calibration_dir is None:
            self.calibration_dir = Path(HF_LEROBOT_CALIBRATION) / "robots" / "so_follower"
