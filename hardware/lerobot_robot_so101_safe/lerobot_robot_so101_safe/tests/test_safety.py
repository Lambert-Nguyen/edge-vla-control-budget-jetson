"""Unit tests 3 to 14 of the safety layer design spec. No arm is needed.

They use a fake bus that records every read and write, and a fake clock, so
the watchdog tests do not wait in real time (except test 14, which checks
the real watchdog thread).

    PYTHONNOUSERSITE=1 ~/lerobot-py310-cuda/bin/python -m pytest hardware/lerobot_robot_so101_safe/tests -q
"""

from __future__ import annotations

import json
import math
import threading
import time
from types import SimpleNamespace

import pytest

from lerobot_robot_so101_safe.rules import (JOINTS, SafetyConfig, SafetyCore, State, filter_action,
                                            park_path, soft_limits)

# Lab follower calibration from the spec (Appendix B), in ticks.
LAB_CAL = {
    "shoulder_pan": (1142, 3202), "shoulder_lift": (786, 3126), "elbow_flex": (937, 3129),
    "wrist_flex": (830, 3225), "wrist_roll": (0, 4095), "gripper": (1490, 2977),
}
CAL = {j: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": lo, "range_max": hi}
       for i, (j, (lo, hi)) in enumerate(LAB_CAL.items())}
REST = {"shoulder_pan": 0.0, "shoulder_lift": -95.0, "elbow_flex": 90.0, "wrist_flex": 70.0,
        "wrist_roll": 0.0, "gripper": 5.0}


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms / 1000


def cfg(**kw):
    return SafetyConfig(arm="lab_follower", rest_pose_deg=dict(REST), **kw)


def zeros():
    return {j: 0.0 for j in JOINTS}


def act(**kw):
    a = {f"{j}.pos": 0.0 for j in JOINTS}
    a["gripper.pos"] = 50.0
    a.update({f"{k}.pos": v for k, v in kw.items()})
    return a


# ---- rules ---------------------------------------------------------------------


def test_soft_limits_match_spec_table():
    lim = soft_limits(CAL, cfg())
    assert lim["shoulder_pan"][1] == pytest.approx(85.5, abs=0.1)
    assert lim["shoulder_lift"][1] == pytest.approx(97.9, abs=0.1)
    assert lim["elbow_flex"][1] == pytest.approx(91.4, abs=0.1)
    assert lim["wrist_flex"][1] == pytest.approx(100.3, abs=0.1)
    assert lim["wrist_roll"][1] == pytest.approx(175.0, abs=0.1)
    assert lim["gripper"] == (2.0, 98.0)


def test_limit_overrides_only_tighten():
    lim = soft_limits(CAL, cfg(limit_overrides={"shoulder_lift": [-75, None], "elbow_flex": [-200, 200]}))
    assert lim["shoulder_lift"][0] == -75.0
    assert lim["shoulder_lift"][1] == pytest.approx(97.9, abs=0.1)  # null keeps it
    assert lim["elbow_flex"] == pytest.approx((-91.4, 91.4), abs=0.1)  # cannot widen
    with pytest.raises(ValueError):
        soft_limits(CAL, cfg(limit_overrides={"shoulder_lift": [99, None]}))


def test_limit_clamp():  # 3
    lim = soft_limits(CAL, cfg())
    present = zeros() | {"shoulder_pan": 84.0, "gripper": 50.0}
    goal, events = filter_action(act(shoulder_pan=120), present, lim, cfg())
    assert goal["shoulder_pan"] == pytest.approx(lim["shoulder_pan"][1])
    assert goal["shoulder_pan"] == pytest.approx(85.5, abs=0.1)
    assert ("clip_limit", "shoulder_pan") in events


def test_step_cap():  # 4
    lim = soft_limits(CAL, cfg())
    present = zeros() | {"gripper": 50.0}
    goal, _ = filter_action(act(elbow_flex=20), present, lim, cfg())
    assert goal["elbow_flex"] == 3.0
    goal, _ = filter_action(act(elbow_flex=-20), present, lim, cfg())
    assert goal["elbow_flex"] == -3.0


def test_step_cap_follows_last_goal():
    # Under load the arm lags its goal. The speed cap counts from the last goal,
    # so the goal keeps moving 3 degrees per step and the servo can push harder.
    lim = soft_limits(CAL, cfg())
    present = zeros() | {"elbow_flex": 4.0, "gripper": 50.0}
    last = present | {"elbow_flex": 10.0}
    goal, _ = filter_action(act(elbow_flex=30), present, lim, cfg(), last)
    assert goal["elbow_flex"] == 13.0


def test_follow_cap():
    # A blocked arm: the goal never gets more than max_follow_deg ahead of it.
    lim = soft_limits(CAL, cfg())
    present = zeros() | {"elbow_flex": 5.0, "gripper": 50.0}
    last = present | {"elbow_flex": 14.0}
    goal, events = filter_action(act(elbow_flex=30), present, lim, cfg(), last)
    assert goal["elbow_flex"] == 15.0 and ("clip_follow", "elbow_flex") in events


def test_core_restarts_from_present_after_hold(make_robot):
    r = ready(make_robot())
    r.bus.present = zeros() | {"gripper": 50.0}
    r.send_action(act(elbow_flex=30))           # goal 3, the fake arm follows
    r.send_action(act(elbow_flex=float("nan")))  # hold at present
    r.send_action(act(elbow_flex=30))
    assert r.bus.writes[-1]["elbow_flex"] == pytest.approx(r.bus.writes[-2]["elbow_flex"] + 3.0)


def test_limit_before_step():  # 5
    lim = soft_limits(CAL, cfg())
    present = zeros() | {"shoulder_pan": 95.0, "gripper": 50.0}
    goal, events = filter_action(act(shoulder_pan=95), present, lim, cfg())
    # Clamped to 85.5 first, then the step cap allows 3 degrees back: 92.0.
    assert goal["shoulder_pan"] == pytest.approx(92.0)
    assert ("clip_limit", "shoulder_pan") in events and ("clip_step", "shoulder_pan") in events


def test_gripper_units():  # 6
    lim = soft_limits(CAL, cfg())
    present = zeros() | {"gripper": 96.0}
    goal, _ = filter_action(act(gripper=100), present, lim, cfg())
    assert goal["gripper"] == 98.0  # limit, in 0-100 units
    present["gripper"] = 50.0
    goal, _ = filter_action(act(gripper=0), present, lim, cfg())
    assert goal["gripper"] == 45.0  # step of 5 units, not 3 degrees


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "x"])
def test_bad_value_is_dropped(bad):
    lim = soft_limits(CAL, cfg())
    a = act()
    a["wrist_flex.pos"] = bad
    goal, events = filter_action(a, zeros() | {"gripper": 50.0}, lim, cfg())
    assert goal is None and events == [("bad_action", "wrist_flex")]


def test_park_path_steps():
    present = zeros() | {"gripper": 50.0}
    path = park_path(present, REST, 2.0)
    prev = present
    for g in path:
        assert all(abs(g[j] - prev[j]) <= 2.0 + 1e-9 for j in JOINTS)
        prev = g
    assert path[-1] == pytest.approx(REST)


# ---- the robot with a fake bus -------------------------------------------------


class FakeBus:
    """Records reads and writes. Fails the test if two threads use it at once."""

    def __init__(self, volts=50, temps=30, delay_s=0.0):
        self.motors = {j: SimpleNamespace(id=i + 1) for i, j in enumerate(JOINTS)}
        self.present = zeros() | {"gripper": 50.0}
        self.volts, self.temps = volts, temps
        self.delay_s = delay_s
        self.writes: list[dict] = []
        self.torque_off_at_disconnect = None
        self.is_calibrated = True
        self.is_connected = True
        self._busy = threading.Lock()
        self.overlaps = 0

    def _enter(self):
        if not self._busy.acquire(blocking=False):
            self.overlaps += 1
            self._busy.acquire()
        if self.delay_s:
            time.sleep(self.delay_s)

    def _leave(self):
        self._busy.release()

    def connect(self, *a, **k):
        pass

    def broadcast_ping(self, num_retry=0):
        return {m.id: 777 for m in self.motors.values()}

    def sync_read(self, name, normalize=True):
        self._enter()
        try:
            if name == "Present_Position":
                return dict(self.present)
            if name == "Present_Voltage":
                return {j: self.volts for j in JOINTS}
            if name == "Present_Temperature":
                return {j: self.temps for j in JOINTS}
            raise KeyError(name)
        finally:
            self._leave()

    def sync_write(self, name, values):
        self._enter()
        try:
            assert name == "Goal_Position"
            self.writes.append(dict(values))
            self.present = dict(values)  # the fake arm reaches each goal at once
        finally:
            self._leave()

    def disconnect(self, disable_torque=True):
        self.torque_off_at_disconnect = disable_torque


@pytest.fixture
def make_robot(tmp_path):
    from lerobot_robot_so101_safe import SafeFollower, SafeFollowerConfig

    def _make(profile="lab", clock=None, **bus_kw):
        cal_dir = tmp_path / "cal"
        cal_dir.mkdir(exist_ok=True)
        (cal_dir / "lab_follower.json").write_text(json.dumps(CAL))
        volt = [4.5, 5.5] if profile == "lab" else [11.0, 12.8]
        yml = tmp_path / f"{profile}.yaml"
        rest = "{" + ", ".join(f"{j}: {v}" for j, v in REST.items()) + "}"
        yml.write_text(f"arm: lab_follower\nvoltage_v: {volt}\nrest_pose_deg: {rest}\n")
        conf = SafeFollowerConfig(port="/dev/null", id="lab_follower", calibration_dir=cal_dir,
                                  safety_config=str(yml), log_dir=str(tmp_path / "logs"))
        robot = SafeFollower(conf, clock=clock or Clock())
        robot.bus = FakeBus(**bus_kw)
        return robot

    return _make


def ready(robot):
    """What connect() does, without configure() and the watchdog thread."""
    robot.safety_check()
    robot.core = SafetyCore(robot.safety, robot.limits, robot.clock)
    return robot


def test_bad_action_holds(make_robot):  # 7
    r = ready(make_robot())
    r.send_action(act(elbow_flex=2))
    n = len(r.bus.writes)
    sent = r.send_action(act(elbow_flex=float("nan")))
    assert r.core.state is State.HOLD
    # The only write is the hold: the present position, not the bad goal.
    assert len(r.bus.writes) == n + 1 and r.bus.writes[-1] == r.bus.present
    assert all(math.isfinite(v) for v in sent.values())
    r.send_action(act(elbow_flex=4))
    assert r.core.state is State.RUN


def test_five_bad_actions_stop(make_robot):  # 8
    r = ready(make_robot())
    for _ in range(5):
        r.send_action(act(gripper=float("nan")))
    assert r.core.state is State.STOP
    assert r.core.counters.stop_reason == "bad_actions"


def test_watchdog_trips(make_robot):  # 9
    clock = Clock()
    r = ready(make_robot(clock=clock))
    r.send_action(act(elbow_flex=1))
    n = len(r.bus.writes)
    for _ in range(14):  # 140 ms, checked every 10 ms
        clock.advance(10)
        assert not r._watchdog_tick()
    clock.advance(10)  # 150 ms after the last action
    assert r._watchdog_tick()
    held_at_ms = (clock() - r.core.last_action_t) * 1e3
    assert held_at_ms < 200
    assert len(r.bus.writes) == n + 1 and r.core.state is State.HOLD
    clock.advance(100)
    assert not r._watchdog_tick()  # one hold write, not one every 10 ms
    assert r.core.counters.holds_watchdog == 1


def test_watchdog_quiet_at_30hz(make_robot):  # 10
    import random

    clock = Clock()
    r = ready(make_robot(clock=clock))
    rng = random.Random(0)
    for _ in range(300):
        r.send_action(act())
        gap = 33 + rng.uniform(0, 60)  # up to 60 ms of jitter: at most 93 ms
        waited = 0
        while waited + 10 <= gap:
            clock.advance(10)
            waited += 10
            r._watchdog_tick()
        clock.advance(gap - waited)
    assert r.core.counters.holds_watchdog == 0


def test_stop_ignores_actions(make_robot):  # 11
    r = ready(make_robot())
    r.core.stop("test")
    n = len(r.bus.writes)
    for _ in range(10):
        r.send_action(act(elbow_flex=2))
    assert len(r.bus.writes) == n


def test_voltage_check(make_robot):  # 12
    from lerobot_robot_so101_safe import SafetyCheckError

    r = make_robot(profile="pro", volts=50)  # 5.0 V read with the 12 V profile
    with pytest.raises(SafetyCheckError, match="voltage"):
        r.connect()
    assert r.bus.torque_off_at_disconnect is True
    assert r.bus.writes == []  # nothing was written, the torque never went on
    ok = make_robot(profile="pro", volts=119)
    ok.safety_check()  # 11.9 V is fine for the Pro kit follower


def test_park_steps(make_robot):  # 13
    r = ready(make_robot())
    r.bus.present = {"shoulder_pan": 20.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                     "wrist_flex": 0.0, "wrist_roll": 10.0, "gripper": 50.0}
    start = dict(r.bus.present)
    r.safety.control_hz = 1000  # do not wait 33 ms per step in a unit test
    r.disconnect()
    prev = start
    for g in r.bus.writes:
        assert all(abs(g[j] - prev[j]) <= 2.0 + 1e-9 for j in JOINTS)
        prev = g
    assert r.bus.writes[-1] == pytest.approx(REST)
    assert r.bus.torque_off_at_disconnect is True
    counters = json.loads(r.counters_path.read_text())
    assert counters["steps"] == 0


def test_bus_lock(make_robot):  # 14
    r = ready(make_robot(clock=time.monotonic, delay_s=0.002))
    r.safety.watchdog_ms = 20
    r.safety.watchdog_check_ms = 1
    r._watchdog = threading.Thread(target=r._watchdog_loop, daemon=True)
    r._watchdog.start()
    try:
        for i in range(60):
            r.send_action(act())
            time.sleep(0.03 if i % 3 else 0.001)  # some gaps trip the watchdog
    finally:
        r._watchdog_stop.set()
        r._watchdog.join()
    assert r.core.counters.holds_watchdog > 0  # the watchdog did run
    assert r.bus.overlaps == 0


def test_bus_failure_returns_the_held_goal(make_robot):
    # lerobot-record stores what send_action returns as the action, so a failed
    # read must still give every joint: the goal the servos keep.
    r = ready(make_robot())
    first = r.send_action(act(elbow_flex=2))
    n = len(r.bus.writes)

    def broken(*a, **k):
        raise OSError("no status packet")

    r.bus.sync_read = broken
    sent = r.send_action(act(elbow_flex=4))
    assert set(sent) == {f"{j}.pos" for j in JOINTS}
    assert sent == first
    assert len(r.bus.writes) == n  # nothing new was written
    assert r.core.counters.bus_retries == 1


def test_disconnect_closes_the_bus_when_park_fails(make_robot):
    r = ready(make_robot())
    r.safety.control_hz = 1000

    def broken(*a, **k):
        raise OSError("port gone")

    r.bus.sync_write = broken
    r.bus.present = zeros() | {"shoulder_pan": 20.0}
    r.log_path.parent.mkdir(parents=True, exist_ok=True)
    r._log_file = open(r.log_path, "a")
    r.disconnect()
    # PARK gave up after a retry, logged it, and the torque still went off.
    assert r.bus.torque_off_at_disconnect is True
    assert r.core.counters.bus_retries == 1
    events = [json.loads(line)["event"] for line in r.log_path.read_text().splitlines()]
    assert events[-1] == "park_failed"
