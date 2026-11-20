"""Rules of the SO-101 safety layer. No input or output here.

Everything in this file is a pure function or a small state machine with an
injected clock, so it can be tested on any computer without an arm. The
hardware part is in robot.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TICKS_MAX = 4095  # STS3215: 4096 positions per turn, LeRobot uses 4095 in its formula


class State(str, Enum):
    CHECK = "CHECK"
    RUN = "RUN"
    HOLD = "HOLD"
    STOP = "STOP"
    PARK = "PARK"


@dataclass
class SafetyConfig:
    """One arm's settings. Loaded from configs/safety/<arm>.yaml."""

    arm: str = "lab_follower"
    control_hz: float = 30.0
    max_step_deg: float = 3.0
    gripper_max_step: float = 5.0
    soft_margin_deg: float = 5.0
    gripper_limits: tuple[float, float] = (2.0, 98.0)
    watchdog_ms: float = 150.0
    watchdog_check_ms: float = 10.0
    bad_actions_to_stop: int = 5
    voltage_v: tuple[float, float] = (4.5, 5.5)
    temp_warn_c: float = 55.0
    temp_stop_c: float = 65.0
    health_every_steps: int = 15
    bus_fail_steps_to_stop: int = 3
    park_step_deg: float = 2.0
    rest_pose_deg: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown safety config keys: {sorted(unknown)}")
        d = dict(d)
        for key in ("gripper_limits", "voltage_v"):
            if key in d and d[key] is not None:
                d[key] = tuple(float(x) for x in d[key])
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "SafetyConfig":
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})


def soft_limits(calibration: dict, cfg: SafetyConfig) -> dict[str, tuple[float, float]]:
    """Soft limit of each joint, in LeRobot's units.

    In degree mode LeRobot puts 0 degrees in the middle of the calibrated
    range: degrees = (ticks - middle) * 360 / 4095. So the range is
    symmetric, and the soft limit keeps soft_margin_deg away from each end.
    The gripper uses LeRobot's 0 to 100 scale and its own limits.
    """
    limits = {}
    for j in JOINTS:
        if j == "gripper":
            limits[j] = (float(cfg.gripper_limits[0]), float(cfg.gripper_limits[1]))
            continue
        c = calibration[j]
        lo_t, hi_t = (c["range_min"], c["range_max"]) if isinstance(c, dict) else (c.range_min, c.range_max)
        half = (hi_t - lo_t) / 2 * 360 / TICKS_MAX
        edge = half - cfg.soft_margin_deg
        if edge <= 0:
            raise ValueError(f"{j}: calibrated range {lo_t}-{hi_t} is smaller than the margin")
        limits[j] = (-edge, edge)
    return limits


def filter_action(action: dict, present: dict, limits: dict, cfg: SafetyConfig):
    """Check one action. Returns (goal, events), or (None, events) for a bad action.

    The order matters. The soft limit comes first, then the step cap, so a
    clamp can never make a jump bigger than the step cap.
    """
    goal, events = {}, []
    for j in JOINTS:
        v = action.get(f"{j}.pos")
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = None
        if v is None or not math.isfinite(v):
            return None, [("bad_action", j)]
        lo, hi = limits[j]
        if not lo <= v <= hi:
            v = min(max(v, lo), hi)
            events.append(("clip_limit", j))
        cap = cfg.gripper_max_step if j == "gripper" else cfg.max_step_deg
        if abs(v - present[j]) > cap:
            v = present[j] + math.copysign(cap, v - present[j])
            events.append(("clip_step", j))
        goal[j] = v
    return goal, events


def check_voltage(volts: dict, cfg: SafetyConfig) -> str | None:
    """None if every servo's voltage is in the arm's range, else the reason."""
    lo, hi = cfg.voltage_v
    bad = {j: v for j, v in volts.items() if not lo <= v <= hi}
    if bad:
        worst = ", ".join(f"{j} {v:.1f} V" for j, v in bad.items())
        return f"voltage out of range {lo}-{hi} V: {worst}. Is this the right arm and config?"
    return None


def park_path(present: dict, rest: dict, step: float) -> list[dict]:
    """Goals from present to rest. No joint moves more than step per goal."""
    n = max(math.ceil(abs(rest[j] - present[j]) / step) for j in JOINTS) if present else 0
    path = []
    for i in range(1, n + 1):
        path.append({j: present[j] + (rest[j] - present[j]) * i / n for j in JOINTS})
    return path


@dataclass
class Counters:
    steps: int = 0
    clip_limit: dict = field(default_factory=lambda: {j: 0 for j in JOINTS})
    clip_step: dict = field(default_factory=lambda: {j: 0 for j in JOINTS})
    bad_actions: int = 0
    holds_watchdog: int = 0
    holds_bad: int = 0
    longest_gap_ms: float = 0.0
    stops: int = 0
    stop_reason: str | None = None
    volt_min: float | None = None
    volt_max: float | None = None
    temp_max: float | None = None
    bus_retries: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class SafetyCore:
    """The state machine. The caller gives it the clock and the bus values."""

    def __init__(self, cfg: SafetyConfig, limits: dict, clock):
        self.cfg = cfg
        self.limits = limits
        self.clock = clock
        self.state = State.RUN
        self.counters = Counters()
        self.events: list[dict] = []
        self.last_action_t: float | None = None
        self.bad_in_row = 0
        self.bus_fail_in_row = 0

    def log(self, event: str, **kw) -> None:
        self.events.append({"t": round(self.clock(), 4), "event": event, **kw})

    def _hold(self, reason: str, **kw) -> None:
        if self.state is State.RUN:
            self.state = State.HOLD
        if reason == "watchdog":
            self.counters.holds_watchdog += 1
        else:
            self.counters.holds_bad += 1
        self.log("hold", reason=reason, **kw)

    def stop(self, reason: str, **kw) -> None:
        if self.state in (State.STOP, State.PARK):
            return
        self.state = State.STOP
        self.counters.stops += 1
        self.counters.stop_reason = reason
        self.log("stop", reason=reason, **kw)

    def on_action(self, action: dict, present: dict):
        """One action from the policy. Returns the goal to write, or None.

        None means: write nothing new. In HOLD and STOP the arm keeps its
        last goal, which the hold already set to where it stood.
        """
        now = self.clock()
        if self.last_action_t is not None:
            gap = (now - self.last_action_t) * 1e3
            self.counters.longest_gap_ms = max(self.counters.longest_gap_ms, gap)
        self.last_action_t = now
        self.counters.steps += 1
        if self.state in (State.STOP, State.PARK, State.CHECK):
            return None

        goal, events = filter_action(action, present, self.limits, self.cfg)
        if goal is None:
            self.counters.bad_actions += 1
            self.bad_in_row += 1
            joint = events[0][1]
            self.log("bad_action", joint=joint)
            if self.bad_in_row >= self.cfg.bad_actions_to_stop:
                self.stop("bad_actions", count=self.bad_in_row)
            else:
                self._hold("bad_action", joint=joint)
            return None

        self.bad_in_row = 0
        if self.state is State.HOLD:
            self.state = State.RUN
            self.log("run")
        for kind, j in events:
            getattr(self.counters, kind)[j] += 1
            want = action.get(f"{j}.pos")
            self.log(kind, joint=j, want=round(float(want), 2), sent=round(goal[j], 2))
        return goal

    def watchdog_due(self) -> bool:
        """True when the arm must be held now: RUN and no action for watchdog_ms."""
        if self.state is not State.RUN or self.last_action_t is None:
            return False
        return (self.clock() - self.last_action_t) * 1e3 >= self.cfg.watchdog_ms

    def on_watchdog_hold(self) -> None:
        gap = (self.clock() - self.last_action_t) * 1e3
        self._hold("watchdog", gap_ms=round(gap, 1))

    def on_bus_result(self, ok: bool) -> None:
        if ok:
            self.bus_fail_in_row = 0
            return
        self.bus_fail_in_row += 1
        if self.bus_fail_in_row >= self.cfg.bus_fail_steps_to_stop:
            self.stop("bus", fails=self.bus_fail_in_row)

    def on_health(self, volts: dict | None = None, temps: dict | None = None) -> None:
        c = self.counters
        if volts:
            vals = list(volts.values())
            c.volt_min = min(vals + ([c.volt_min] if c.volt_min is not None else []))
            c.volt_max = max(vals + ([c.volt_max] if c.volt_max is not None else []))
            reason = check_voltage(volts, self.cfg)
            if reason:
                self.stop("voltage", detail=reason)
        if temps:
            hot_j = max(temps, key=temps.get)
            hot = temps[hot_j]
            c.temp_max = hot if c.temp_max is None else max(c.temp_max, hot)
            if hot >= self.cfg.temp_stop_c:
                self.stop("temperature", joint=hot_j, value=hot)
            elif hot >= self.cfg.temp_warn_c:
                self.log("temp_warn", joint=hot_j, value=hot)
