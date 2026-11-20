"""Record an arm's rest pose into its safety config.

Put the follower by hand in the pose it should park in, with the torque off.
This reads the joints once and writes them as rest_pose_deg in the YAML file.
It never writes to the servos.

    python -m lerobot_robot_so101_safe.rest_pose --port PORT --id team_follower \
        --config configs/safety/team_follower.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .rules import JOINTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True, help="calibration name, e.g. team_follower")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    if not robot.calibration:
        raise SystemExit(f"No calibration file for {args.id}.")
    robot.bus.connect()
    try:
        pos = robot.bus.sync_read("Present_Position")
    finally:
        robot.bus.disconnect(disable_torque=False)
    pose = "{" + ", ".join(f"{j}: {pos[j]:.1f}" for j in JOINTS) + "}"
    text = args.config.read_text()
    line = f"rest_pose_deg: {pose}"
    if not re.search(r"^rest_pose_deg:.*$", text, flags=re.M):
        raise SystemExit(f"{args.config} has no rest_pose_deg line.")
    text = re.sub(r"^rest_pose_deg:.*$", line, text, count=1, flags=re.M)
    args.config.write_text(text)
    print(f"wrote to {args.config}:\n  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
