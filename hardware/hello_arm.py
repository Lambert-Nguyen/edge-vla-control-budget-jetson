#!/usr/bin/env python3
"""Smallest safe motion on the SO-101 follower arm.

Connects, reads the current pose, nudges one joint by a small amount in slow
interpolated steps, returns it to where it started, and reports the tracking
error between what was commanded and what the servo actually reached.

Keep a hand on the servo power switch the first time you run this.

    # look, do not touch
    python3 hardware/hello_arm.py --port /dev/so101_follower --dry-run

    # move the safest joint a little
    python3 hardware/hello_arm.py --port /dev/so101_follower --joint wrist_roll

Units are LeRobot normalized position, not degrees. After calibration each
joint spans roughly -100 to 100 (the gripper spans 0 to 100), where the range
maps to the mechanical limits recorded during calibration. A delta of 5 is a
small, visible nudge.
"""

from __future__ import annotations

import argparse
import sys
import time

# Safe default. wrist_roll cannot drive the arm into the table and carries no
# load, so it is the right joint for a first power-on.
DEFAULT_JOINT = "wrist_roll"

# Hard ceiling on a single command from this script, whatever the user passes.
MAX_DELTA = 15.0

# Joints that can swing the whole arm. Refuse to move these without --force.
HEAVY_JOINTS = {"shoulder_pan", "shoulder_lift", "elbow_flex"}


def import_robot():
    """LeRobot moved these modules between 0.4.x and 0.5.x. Try both."""
    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        return SO101Follower, SO101FollowerConfig, "lerobot.robots"
    except ImportError:
        pass
    try:
        from lerobot.common.robots.so101_follower import (
            SO101Follower,
            SO101FollowerConfig,
        )
        return SO101Follower, SO101FollowerConfig, "lerobot.common.robots"
    except ImportError as exc:
        print(f"[FAIL] cannot import the SO-101 driver: {exc}", file=sys.stderr)
        print(
            "       Activate the arm environment first:\n"
            "         source ~/lerobot-py312/bin/activate\n"
            "       and confirm it was installed with the feetech extra:\n"
            '         pip install "lerobot[feetech]"',
            file=sys.stderr,
        )
        raise SystemExit(1)


def read_positions(robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {k: v for k, v in obs.items() if k.endswith(".pos")}


def print_pose(label: str, pose: dict[str, float]) -> None:
    print(f"\n{label}")
    for key in sorted(pose):
        print(f"  {key:<24} {pose[key]:8.2f}")


def glide(robot, start: dict[str, float], key: str, target: float, steps: int, hz: float):
    """Interpolate to the target so the servo never sees a step command.

    A single large jump makes the servo move at whatever speed it can, which is
    fast enough to be alarming and to shake an unclamped base.
    """
    period = 1.0 / hz
    begin = start[key]
    for i in range(1, steps + 1):
        action = dict(start)
        action[key] = begin + (target - begin) * (i / steps)
        robot.send_action(action)
        time.sleep(period)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/so101_follower",
                        help="serial port or udev symlink for the follower")
    parser.add_argument("--robot-id", default="so101_follower",
                        help="calibration id, must match what lerobot-calibrate wrote")
    parser.add_argument("--joint", default=DEFAULT_JOINT)
    parser.add_argument("--delta", type=float, default=5.0,
                        help="normalized position change, capped at %(default)s")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--dwell", type=float, default=0.75,
                        help="seconds to hold at the target before returning")
    parser.add_argument("--dry-run", action="store_true",
                        help="connect and read, command nothing")
    parser.add_argument("--force", action="store_true",
                        help="allow a shoulder or elbow joint")
    args = parser.parse_args()

    delta = max(-MAX_DELTA, min(MAX_DELTA, args.delta))
    if delta != args.delta:
        print(f"[warn] clamping delta {args.delta} to {delta}")

    if args.joint in HEAVY_JOINTS and not args.force and not args.dry_run:
        print(f"[FAIL] {args.joint} can swing the whole arm. Test {DEFAULT_JOINT} "
              f"first, then re-run with --force if you really mean it.",
              file=sys.stderr)
        return 1

    SO101Follower, SO101FollowerConfig, module = import_robot()
    print(f"[ok]   driver from {module}")

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))

    print(f"[..]   connecting to {args.port} as '{args.robot_id}'")
    robot.connect()
    print("[ok]   connected")

    try:
        start = read_positions(robot)
        if not start:
            print("[FAIL] no '.pos' keys in the observation. Is the arm calibrated?",
                  file=sys.stderr)
            return 1
        print_pose("current pose", start)

        key = f"{args.joint}.pos"
        if key not in start:
            print(f"\n[FAIL] unknown joint '{args.joint}'. Available: "
                  f"{', '.join(sorted(k[:-4] for k in start))}", file=sys.stderr)
            return 1

        if args.dry_run:
            print("\n[ok]   dry run, nothing commanded. "
                  "Re-run without --dry-run to move.")
            return 0

        target = start[key] + delta
        print(f"\n[..]   {args.joint}: {start[key]:.2f} -> {target:.2f} "
              f"over {args.steps} steps at {args.hz:.0f} Hz")
        print("       hand on the power switch")

        glide(robot, start, key, target, args.steps, args.hz)
        time.sleep(args.dwell)

        reached = read_positions(robot)
        error = reached[key] - target
        print(f"[ok]   reached {reached[key]:.2f}, tracking error {error:+.2f}")

        print(f"[..]   returning to {start[key]:.2f}")
        glide(robot, reached, key, start[key], args.steps, args.hz)
        time.sleep(args.dwell)

        final = read_positions(robot)
        drift = final[key] - start[key]
        print(f"[ok]   back at {final[key]:.2f}, drift from start {drift:+.2f}")

        print("\nTracking error and drift are the floor on how precisely any "
              "policy can be executed. Note them in the team channel.")
        return 0

    finally:
        # Always disconnect. Leaving the bus open blocks the next run and can
        # leave servos holding torque.
        robot.disconnect()
        print("[ok]   disconnected")


if __name__ == "__main__":
    sys.exit(main())
