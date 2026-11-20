"""SafeFollower: the SO-101 follower with the safety layer in front of the servos.

This is the hardware part. It reads and writes the bus, runs the watchdog
thread, checks health and parks the arm. The rules are in rules.py.
Every bus read or write holds self.bus_lock, because the main thread and the
watchdog thread share the serial bus.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from lerobot.robots.so_follower import SOFollower

from .config import SafeFollowerConfig
from .rules import JOINTS, SafetyConfig, SafetyCore, State, check_voltage, park_path, soft_limits

logger = logging.getLogger(__name__)


class SafetyCheckError(RuntimeError):
    """CHECK failed. The torque stays off."""


class SafeFollower(SOFollower):
    config_class = SafeFollowerConfig
    name = "so101_safe_follower"

    def __init__(self, config: SafeFollowerConfig, clock=time.monotonic):
        # The layer does the step cap itself, so the bus is read once per step.
        config.max_relative_target = None
        super().__init__(config)
        self.config = config
        self.clock = clock
        self.safety = SafetyConfig.load(config.safety_config)
        if config.max_step_deg is not None:
            self.safety.max_step_deg = float(config.max_step_deg)
        self.bus_lock = threading.Lock()
        self.core: SafetyCore | None = None
        self.limits: dict | None = None
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._health_turn = 0
        self._log_file = None
        self._last_print: dict[str, float] = {}
        name = config.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = Path(config.log_dir) / f"{name}.jsonl"
        self.counters_path = Path(config.log_dir) / f"{name}.counters.json"

    # ---- CHECK -------------------------------------------------------------

    def safety_check(self) -> None:
        """Run before the torque goes on. Raises SafetyCheckError on any problem."""
        if self.safety.arm != self.id:
            raise SafetyCheckError(f"safety config is for '{self.safety.arm}', but the robot id is '{self.id}'")
        if self.safety.rest_pose_deg is None:
            raise SafetyCheckError("no rest pose in the safety config. Record it with: so101_lab.sh rest-pose")
        missing = set(JOINTS) - set(self.safety.rest_pose_deg)
        if missing:
            raise SafetyCheckError(f"rest pose has no value for {sorted(missing)}")
        if not self.calibration:
            raise SafetyCheckError(f"no calibration file {self.calibration_fpath}")
        with self.bus_lock:
            found = self.bus.broadcast_ping(num_retry=2) or {}
            answer = [m for m in JOINTS if self.bus.motors[m].id in found]
            if len(answer) < len(JOINTS):
                raise SafetyCheckError(f"only {len(answer)} of 6 servos answer: missing "
                                       f"{sorted(set(JOINTS) - set(answer))}")
            if not self.bus.is_calibrated:
                raise SafetyCheckError("the servos' calibration does not match the calibration file")
            volts = {m: v / 10 for m, v in self.bus.sync_read("Present_Voltage", normalize=False).items()}
            temps = dict(self.bus.sync_read("Present_Temperature", normalize=False))
        reason = check_voltage(volts, self.safety)
        if reason:
            raise SafetyCheckError(reason)
        hot = max(temps.values())
        if hot >= self.safety.temp_stop_c:
            raise SafetyCheckError(f"a servo is at {hot} C, the stop limit is {self.safety.temp_stop_c} C")
        self.limits = soft_limits(self.calibration, self.safety)
        self._check_volts, self._check_temps = volts, temps

    def connect(self, calibrate: bool = True) -> None:
        # Like the parent, but CHECK runs before configure(), which turns the
        # torque on. A failed CHECK leaves the torque off.
        self.bus.connect()
        try:
            self.safety_check()
        except SafetyCheckError:
            self.bus.disconnect(disable_torque=True)
            raise
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        self.core = SafetyCore(self.safety, self.limits, self.clock)
        self.core.on_health(self._check_volts, self._check_temps)
        self.core.log("check_ok", volts=self._check_volts, limits=self.limits)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "a")
        self._flush_events()
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()
        logger.info(f"{self} connected, safety config {self.config.safety_config}")

    # ---- RUN ---------------------------------------------------------------

    def _read_present(self) -> dict | None:
        """Present position, with one retry. Caller holds bus_lock."""
        for attempt in range(2):
            try:
                return dict(self.bus.sync_read("Present_Position"))
            except Exception:
                if attempt == 0:
                    self.core.counters.bus_retries += 1
        return None

    def _write_goal(self, goal: dict) -> None:
        self.bus.sync_write("Goal_Position", goal)

    def send_action(self, action: dict) -> dict:
        with self.bus_lock:
            present = self._read_present()
            self.core.on_bus_result(present is not None)
            if present is None:
                self._flush_events()
                return {}
            before = self.core.state
            goal = self.core.on_action(action, present)
            if goal is not None:
                self._write_goal(goal)
            self._health_step()
            if before is State.RUN and self.core.state in (State.HOLD, State.STOP):
                # A bad action, a health fault or a bus fault: hold the arm where it is.
                self._write_goal(present)
                goal = None
            sent = goal if goal is not None else present
        self._flush_events()
        return {f"{j}.pos": v for j, v in sent.items()}

    def _health_step(self) -> None:
        """Every health_every_steps steps, read voltage or temperature, in turns."""
        if self.core.counters.steps % self.safety.health_every_steps:
            return
        self._health_turn ^= 1
        try:
            if self._health_turn:
                raw = self.bus.sync_read("Present_Voltage", normalize=False)
                self.core.on_health(volts={m: v / 10 for m, v in raw.items()})
            else:
                self.core.on_health(temps=dict(self.bus.sync_read("Present_Temperature", normalize=False)))
        except Exception:
            self.core.counters.bus_retries += 1

    def _watchdog_loop(self) -> None:
        period = self.safety.watchdog_check_ms / 1000
        while not self._watchdog_stop.wait(period):
            self._watchdog_tick()

    def _watchdog_tick(self) -> bool:
        """One watchdog check. Returns True when it wrote a hold."""
        if not self.core.watchdog_due():
            return False
        with self.bus_lock:
            if not self.core.watchdog_due():  # an action came in meanwhile
                return False
            present = self._read_present()
            if present is not None:
                self._write_goal(present)
            self.core.on_watchdog_hold()
        self._flush_events()
        return True

    # ---- events --------------------------------------------------------------

    def _flush_events(self) -> None:
        if self.core is None:
            return
        events, self.core.events = self.core.events, []
        for e in events:
            if self._log_file:
                self._log_file.write(json.dumps(e) + "\n")
            kind = e["event"]
            if kind in ("clip_limit", "clip_step", "hold", "bad_action", "stop", "temp_warn"):
                now = time.monotonic()
                if now - self._last_print.get(kind, -1e9) >= 1.0:  # at most once a second per type
                    self._last_print[kind] = now
                    logger.warning("safety %s", {k: v for k, v in e.items() if k != "t"})
        if self._log_file:
            self._log_file.flush()

    # ---- PARK ----------------------------------------------------------------

    def park(self) -> None:
        """Move slowly to the rest pose. The caller turns the torque off after."""
        self.core.state = State.PARK
        self.core.log("park")
        rest = {j: float(self.safety.rest_pose_deg[j]) for j in JOINTS}
        with self.bus_lock:
            present = self._read_present()
        if present is None:
            self.core.log("park_failed", reason="bus")
            return
        period = 1.0 / self.safety.control_hz
        for goal in park_path(present, rest, self.safety.park_step_deg):
            with self.bus_lock:
                self._write_goal(goal)
            time.sleep(period)
        time.sleep(0.5)  # let the servos reach the last goal

    def disconnect(self) -> None:
        if self.core is not None:
            self._watchdog_stop.set()
            if self._watchdog:
                self._watchdog.join(timeout=1)
            try:
                self.park()
            finally:
                self._flush_events()
                c = self.core.counters.as_dict()
                self.counters_path.parent.mkdir(parents=True, exist_ok=True)
                self.counters_path.write_text(json.dumps(c, indent=2) + "\n")
                logger.info("safety counters: %s", json.dumps(c))
                if self._log_file:
                    self._log_file.close()
                    self._log_file = None
        super().disconnect()
