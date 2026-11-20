#!/usr/bin/env python3
"""Table calibration for the gripper height check (S-11). Only reads the joints.

The height check needs two things per arm: the table height, and small angle
offsets between LeRobot's degrees and the arm model. You move the follower by
hand, with the torque off, and touch the tip of the FIXED jaw (the one that
does not move) to the table at 6 or more spots. The tool fits the offsets and
the table height, shows how well they fit, and writes them into the safety
config, with height_check on.

    systemctl --user stop so101-teleop        # the follower goes limp, hold it
    source ~/team_arm.env
    python3 hardware/table_calibration.py --port "$FOLLOWER_PORT" --id "$FOLLOWER_ID" \\
        --config configs/safety/team_follower.yaml

It never writes to the servos. The touches are saved in logs/safety/, and
--refit FILE fits them again without the arm.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# The script folder (hardware/) is first on sys.path. Put the package folder first,
# so the installed package is found, not the project folder of the same name.
sys.path.insert(0, str(Path(__file__).resolve().parent / "lerobot_robot_so101_safe"))

MIN_TOUCHES = 6
MAX_RMS_MM = 5.0
MAX_OFFSET_DEG = 30.0


def read_pose(robot) -> dict:
    pos = robot.bus.sync_read("Present_Position")
    return {k: float(v) for k, v in pos.items()}


def fit_and_report(touches: list[dict], upright: dict | None):
    import numpy as np
    from lerobot_robot_so101_safe.kinematics import fit_table

    offsets, table_z, res = fit_table(touches, upright)
    n = len(touches)
    touch_mm = 1000 * res[:n]
    rms = float(np.sqrt(np.mean(touch_mm ** 2)))
    print(f"\nFit from {n} touches" + (" and the upright pose" if upright else "") + ":")
    print("  offsets (model = LeRobot + offset): " + ", ".join(f"{j} {v:+.1f} deg" for j, v in offsets.items()))
    print(f"  table height in the model frame: {100 * table_z:+.1f} cm")
    print("  each touch, tip height above the fitted table (mm): " + ", ".join(f"{x:+.1f}" for x in touch_mm))
    if upright is not None:
        print("  upright pose, elbow / wrist / tip off the vertical (mm): "
              + ", ".join(f"{1000 * x:+.1f}" for x in res[n:]))
    print(f"  rms of the touches: {rms:.1f} mm")
    ok = rms <= MAX_RMS_MM and all(abs(v) <= MAX_OFFSET_DEG for v in offsets.values())
    if not ok:
        print(f"[warn] the fit is not good enough (rms over {MAX_RMS_MM} mm, or an offset over "
              f"{MAX_OFFSET_DEG} deg). Check that the FIXED jaw tip touched each time, add touches "
              "at more different poses, and run again.")
    return offsets, table_z, ok


def write_config(path: Path, offsets: dict, table_z: float) -> None:
    text = path.read_text()
    values = {
        "height_check": "true",
        "table_z_m": f"{table_z:.4f}",
        "model_offsets_deg": "{" + ", ".join(f"{j}: {v:.2f}" for j, v in offsets.items()) + "}",
    }
    stamp = datetime.now().strftime("%Y-%m-%d")
    for key, val in values.items():
        line = f"{key}: {val}"
        if re.search(rf"^{key}:.*$", text, flags=re.M):
            text = re.sub(rf"^{key}:.*$", line, text, count=1, flags=re.M)
        else:
            text = text.replace("rest_pose_deg:", f"{line}\nrest_pose_deg:", 1)
    if "# Table calibration" not in text:
        text = text.replace("height_check:", f"# Table calibration ({stamp}, hardware/table_calibration.py).\n"
                            "height_check:", 1)
    path.write_text(text)
    print(f"\nwrote height_check, table_z_m and model_offsets_deg to {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--id")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--refit", type=Path, help="fit saved touches again, without the arm")
    ap.add_argument("--log-dir", type=Path, default=Path("logs/safety"))
    args = ap.parse_args()

    from lerobot_robot_so101_safe.rules import SafetyConfig

    cfg = SafetyConfig.load(args.config)
    upright = cfg.upright_pose_deg

    if args.refit:
        saved = json.loads(args.refit.read_text())
        offsets, table_z, ok = fit_and_report(saved["touches"], saved.get("upright", upright))
    else:
        if not args.port or not args.id:
            ap.error("--port and --id are needed, or use --refit")
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
        if not robot.calibration:
            raise SystemExit(f"No calibration file for {args.id}.")
        robot.bus.connect()
        touches: list[dict] = []
        try:
            print(__doc__.split("\n\n")[1])
            print(f"\nUpright pose from the config: {upright}" if upright else
                  "\nNo upright pose in the config. Type u to record it (whole arm straight up).")
            print("Enter = record a touch, d = delete the last one, u = record the upright pose, q = done")
            while True:
                cmd = input(f"[{len(touches)} touches] > ").strip().lower()
                if cmd == "q":
                    if len(touches) >= MIN_TOUCHES:
                        break
                    print(f"  need at least {MIN_TOUCHES} touches")
                elif cmd == "d":
                    if touches:
                        touches.pop()
                        print("  deleted the last touch")
                elif cmd == "u":
                    upright = read_pose(robot)
                    print("  upright: " + ", ".join(f"{k} {v:.1f}" for k, v in upright.items()))
                else:
                    pose = read_pose(robot)
                    touches.append(pose)
                    print("  touch: " + ", ".join(f"{k} {v:.1f}" for k, v in pose.items()))
        finally:
            robot.bus.disconnect(disable_torque=False)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        saved = args.log_dir / f"table-touches-{datetime.now():%Y%m%d-%H%M%S}.json"
        saved.write_text(json.dumps({"arm": args.id, "touches": touches, "upright": upright}, indent=2) + "\n")
        print(f"saved the touches to {saved}")
        offsets, table_z, ok = fit_and_report(touches, upright)

    if not ok:
        return 1
    if input("\nWrite this into the safety config and turn the height check on? [y/n] ").strip().lower() == "y":
        write_config(args.config, offsets, table_z)
        print("Run bench case B-9 next: hardware/safety_test.py --cases B-9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
