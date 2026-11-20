#!/usr/bin/env python3
"""E-8 bench test of the safety layer on the real follower. No policy runs.

It sends scripted bad actions through the safety layer (robot type
so101_safe_follower) and checks cases B-1 to B-7 of the safety layer design
spec. THE ARM MOVES. Before you start:

- The arm is clamped to the table, and nothing is in its reach.
- Two people are there. One runs the test, one stands at the servo power switch.
- The teleop service is stopped (stop teleop), so the port is free.
- The rest pose is recorded (so101_lab.sh rest-pose).

    source ~/team_arm.env
    python3 hardware/safety_test.py --port "$FOLLOWER_PORT" --id "$FOLLOWER_ID" \\
        --config configs/safety/team_follower.yaml --wrong-config configs/safety/lab_follower.yaml

It asks before each case. The report goes to logs/safety/bench-<time>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HZ = 30.0


def ask(msg: str) -> bool:
    try:
        return input(f"\n{msg} [Enter = run, s = skip] ").strip().lower() != "s"
    except EOFError:
        return False


def present(robot) -> dict:
    with robot.bus_lock:
        return dict(robot.bus.sync_read("Present_Position"))


def pos_action(goal: dict) -> dict:
    return {f"{j}.pos": v for j, v in goal.items()}


def drive(robot, goal: dict, steps: int, log=None) -> None:
    """Send the same goal for a number of steps at 30 Hz."""
    for _ in range(steps):
        t = time.perf_counter()
        robot.send_action(pos_action(goal))
        if log is not None:
            log.append(present(robot))
        time.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t)))


def watch(robot, seconds: float) -> list[dict]:
    out = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        out.append(present(robot))
        time.sleep(0.02)
    return out


def moved(samples: list[dict], joints) -> float:
    return max(max(s[j] for s in samples) - min(s[j] for s in samples) for j in joints)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--config", required=True, help="this arm's safety config")
    ap.add_argument("--wrong-config", help="another arm's config, for B-7")
    args = ap.parse_args()

    from lerobot_robot_so101_safe import SafeFollower, SafeFollowerConfig, SafetyCheckError
    from lerobot_robot_so101_safe.rules import JOINTS, State

    name = "bench-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    results: dict[str, dict] = {}

    def record(case: str, ok: bool, **detail) -> None:
        results[case] = {"pass": ok, **detail}
        print(f"  {case}: {'PASS' if ok else 'FAIL'} {detail}")

    print(__doc__.split("\n\n")[1])
    if not ask("Arm clamped, area clear, a second person at the power switch?"):
        return 1

    # B-7 first: the wrong profile must be refused before any torque goes on.
    if args.wrong_config and ask("B-7: start with the other arm's config. The arm must not move."):
        bad = SafeFollower(SafeFollowerConfig(port=args.port, id=args.id, safety_config=args.wrong_config,
                                              run_name=name + "-b7"))
        bad.safety.arm = args.id  # test the voltage profile, not the name check
        try:
            bad.connect()
            record("B-7", False, reason="connect did not refuse")
            bad.disconnect()
        except SafetyCheckError as e:
            record("B-7", True, refused=str(e))

    robot = SafeFollower(SafeFollowerConfig(port=args.port, id=args.id, safety_config=args.config,
                                            run_name=name))
    robot.connect()
    lim = robot.limits
    home = present(robot)
    try:
        if ask("B-1: each joint gets a goal 30 degrees past its soft limit, then goes back."):
            worst = {}
            for j in JOINTS:
                if j == "wrist_roll":
                    continue  # full turn joint, its limit is not a collision limit
                for side in (0, 1):
                    edge = lim[j][side]
                    goal = dict(home)
                    goal[j] = edge + (30 if side else -30) * (0.5 if j == "gripper" else 1)
                    log: list[dict] = []
                    drive(robot, goal, steps=int(abs(goal[j] - home[j]) / 2) + 30, log=log)
                    past = max((s[j] - edge) if side else (edge - s[j]) for s in log)
                    worst[f"{j}.{'hi' if side else 'lo'}"] = round(past, 2)
                    drive(robot, home, steps=int(abs(edge - home[j]) / 2) + 30)
            record("B-1", max(worst.values()) <= 1.0, past_limit_deg=worst)

        if ask("B-2: elbow_flex jumps 45 degrees."):
            start = present(robot)
            goal = dict(start)
            goal["elbow_flex"] = max(min(start["elbow_flex"] + 45, lim["elbow_flex"][1]), lim["elbow_flex"][0])
            log = [start]
            drive(robot, goal, steps=40, log=log)
            steps = [abs(b["elbow_flex"] - a["elbow_flex"]) for a, b in zip(log, log[1:])]
            n_moving = sum(1 for s in steps if s > 0.3)
            record("B-2", max(steps) <= 3.5 and n_moving >= 15,
                   max_step_deg=round(max(steps), 2), moving_steps=n_moving)
            drive(robot, start, steps=40)

        if ask("B-3: one action with NaN in wrist_flex."):
            drive(robot, present(robot), steps=10)
            a = pos_action(present(robot))
            a["wrist_flex.pos"] = float("nan")
            robot.send_action(a)
            state = robot.core.state
            after = watch(robot, 1.0)
            record("B-3", state is State.HOLD and moved(after, JOINTS) <= 0.5,
                   state=state.value, moved_deg=round(moved(after, JOINTS), 2))
            drive(robot, present(robot), steps=5)  # back to RUN

        if ask("B-4: shoulder_pan moves slowly, then the actions stop for 1 s."):
            start = present(robot)
            goal = dict(start)
            for i in range(20):
                goal["shoulder_pan"] = start["shoulder_pan"] + 0.5 * (i + 1)
                drive(robot, goal, steps=1)
            after = watch(robot, 1.0)
            holds = [e for e in json.loads("[" + ",".join(Path(robot.log_path).read_text().splitlines()) + "]")
                     if e["event"] == "hold" and e.get("reason") == "watchdog"]
            gap = holds[-1]["gap_ms"] if holds else None
            settle = after[len(after) // 4:]  # after the servo reached the hold goal
            ok = gap is not None and gap < 200 and moved(settle, ["shoulder_pan"]) <= 1.0
            record("B-4", ok, hold_after_ms=gap, moved_after_hold_deg=round(moved(settle, ["shoulder_pan"]), 2))
            drive(robot, start, steps=30)

        if ask("B-5: 5 bad actions in a row, then good actions that must be ignored."):
            for _ in range(5):
                a = pos_action(present(robot))
                a["elbow_flex.pos"] = float("nan")
                robot.send_action(a)
            before = present(robot)
            goal = dict(before)
            goal["elbow_flex"] = before["elbow_flex"] + 10
            log: list[dict] = []
            drive(robot, goal, steps=30, log=log)
            record("B-5", robot.core.state is State.STOP and moved([before] + log, JOINTS) <= 0.5,
                   state=robot.core.state.value, moved_deg=round(moved([before] + log, JOINTS), 2))
    except KeyboardInterrupt:
        print("\nCtrl+C: parking (this is B-6).")
    finally:
        print("\nB-6: the arm now parks at its rest pose, then the torque goes off. Watch that it does not fall.")
        robot.disconnect()
        rest = robot.safety.rest_pose_deg
        b6 = input("Did the arm reach its rest pose and stay there without falling? [y/n] ").strip().lower()
        record("B-6", b6 == "y", operator="confirmed" if b6 == "y" else "not confirmed", rest_pose=rest)
        b8 = input("B-8: was a second person at the power switch for the whole test? [y/n] ").strip().lower()
        record("B-8", b8 == "y")
        out = Path(robot.config.log_dir) / f"{name}.json"
        out.write_text(json.dumps({"config": args.config, "cases": results,
                                   "counters": robot.core.counters.as_dict()}, indent=2) + "\n")
        print(f"\nwrote {out}")
        failed = [c for c, r in results.items() if not r["pass"]]
        print("E-8 bench test:", "PASS" if not failed and len(results) >= 8 else f"not passed ({failed or 'skipped cases'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
