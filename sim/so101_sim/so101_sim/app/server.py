"""The simulator app: physics and rendering run here, the controls open in a browser.

Why a browser: the same page works on Windows, macOS and Linux with no GUI toolkit
to install, and the simulator keeps all OpenGL work on the main thread, which is
what macOS requires. The HTTP server only moves JPEG frames and small JSON
messages between the two, on 127.0.0.1 unless --host says otherwise.

Control modes:
  keyboard  steer the jaws in space with the keyboard (inverse kinematics)
  sliders   set each joint directly
  leader    a real SO-101 leader arm on USB drives the simulated follower
  expert    the scripted expert does the task, once or N times while recording
  policy    a LeRobot checkpoint drives the arm, with the Jetson's latency injected
"""

from __future__ import annotations

import io
import json
import math
import queue
import threading
import time
import traceback
import webbrowser
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import mujoco
import numpy as np
from PIL import Image

from ..config import FEATURE_NAMES, SimConfig
from ..env import SO101Env
from ..expert import GRIP_CLOSED, GRIP_OPEN, ScriptedExpert
from ..ik import ROLL, ArmIK
from ..units import JointMap

STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                "style.css": "text/css; charset=utf-8"}
STREAMS = ("view", "camera1", "camera2")
KEY_TIMEOUT_S = 0.6           # forget held keys if the browser stops reporting them
# Rendering is most of a tick (the arm meshes are ~400k triangles; resolution barely
# matters). camera1/camera2 are rendered every step only when something uses them,
# recording or a policy; otherwise the page gets them as previews at this rate.
CAM_PREVIEW_HZ = 10.0
VIEW_HZ_RECORDING = 10.0      # the 3D view's rate while recording, to keep 30 Hz
WORKSPACE = (np.array([0.02, -0.38, 0.0]), np.array([0.42, 0.38, 0.38]))


def lerobot_available() -> bool:
    try:
        import lerobot  # noqa: F401
        return True
    except ImportError:
        return False


class FrameHub:
    """Latest JPEG per stream, and a condition the streaming threads wait on."""

    def __init__(self):
        self.cond = threading.Condition()
        self.frames: dict[str, bytes] = {}
        self.seq = 0
        self.closed = False

    def publish(self, frames: dict[str, bytes]) -> None:
        with self.cond:
            self.frames.update(frames)
            self.seq += 1
            self.cond.notify_all()

    def wait(self, last_seq: int, timeout: float = 1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or self.closed, timeout)
            return self.seq, dict(self.frames)

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()


def jpeg(img: np.ndarray, size: tuple[int, int] | None = None, quality: int = 80) -> bytes:
    im = Image.fromarray(img)
    if size and im.size != size:
        im = im.resize(size, Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class SimApp:
    def __init__(self, cfg: SimConfig, joint_map: JointMap | None = None, dataset: str | None = None,
                 root: str | None = None, task: str | None = None, seed: int = 0,
                 leader_port: str | None = None, leader_id: str = "my_leader",
                 policy: str | None = None, latency_ms: float = 0.0, policy_mode: str = "async",
                 threshold: float = 0.5, num_steps: int | None = None, ui_hz: float = 30.0,
                 view_size: tuple[int, int] = (960, 720)):
        if task:
            cfg = cfg.with_(task=replace(cfg.task, instruction=task))
        self.cfg = cfg
        self.env = SO101Env(cfg, joint_map=joint_map, render=True)
        self.ik = ArmIK(self.env.model, self.env.qadr, self.env.dadr)
        self.cmds: queue.Queue = queue.Queue()
        self.hub = FrameHub()
        self.status_lock = threading.Lock()
        self.status: dict = {}
        self.log_lines: deque[str] = deque(maxlen=14)
        self.running = True
        self.paused = False
        self.seed = seed
        self.step = 0
        self.success_streak = 0
        self.mode = "keyboard"
        self.keys: set[str] = set()
        self.keys_t = 0.0
        self.kb: dict | None = None
        self.slider: np.ndarray | None = None
        self.leader = None
        self.leader_port, self.leader_id = leader_port, leader_id
        self.expert: ScriptedExpert | None = None
        self.expert_tail = 0
        self.batch_left = 0
        self.batch_stats = [0, 0]                # saved, attempted
        self.fast = False
        self.policy_spec = policy
        self.policy_cfg = dict(latency_ms=latency_ms, mode=policy_mode, threshold=threshold, num_steps=num_steps)
        self.policy = None
        self.executor = None
        self.rec = None
        self.dataset_name = dataset
        self.dataset_root = root
        self.recording = False
        self.ep_frames = 0
        self.saved_episodes = 0
        self.ui_period = 1.0 / max(ui_hz, 1.0)
        self._last_ui = 0.0
        self._last_cam_preview = 0.0
        self._slow_warned = False
        self._tick_times: deque[float] = deque(maxlen=60)
        self.last_action = self.env.state()
        self.cam = mujoco.MjvCamera()
        self.reset_view()
        self.view_size = view_size
        self._view_renderer: mujoco.Renderer | None = None
        self.has_lerobot = lerobot_available()
        lo = self.env.jmap.to_lerobot(self.env.ctrl_lo)
        hi = self.env.jmap.to_lerobot(self.env.ctrl_hi)
        self.limits = [[round(float(min(a, b)), 1), round(float(max(a, b)), 1)] for a, b in zip(lo, hi)]

    # ------------------------------------------------------------------ helpers

    def log(self, msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        self.log_lines.append(line)
        print(line, flush=True)

    def reset_view(self) -> None:
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0.17, 0.0, 0.06]
        self.cam.distance = 0.78
        self.cam.azimuth = 145.0
        self.cam.elevation = -28.0

    # ------------------------------------------------------------------ scene

    def new_scene(self, seed: int | None = None) -> None:
        if self.recording:
            self.discard_episode(reset=False)
        self.seed = self.seed + 1 if seed is None else seed
        self.env.reset(seed=self.seed, render=False)
        self.step = 0
        self.success_streak = 0
        self.last_action = self.env.state()
        self.enter_mode(self.mode, quiet=True)

    # ------------------------------------------------------------------ modes

    def enter_mode(self, mode: str, quiet: bool = False) -> None:
        if mode != "leader" and self.leader is not None:
            self.leader.close()
            self.leader = None
        if mode != "expert":
            self.batch_left = 0
        q = self.env.qpos_arm()
        try:
            if mode == "keyboard":
                p, R = self.ik.fk(q)
                self.kb = dict(pos=p, pitch=self.ik.pitch_of(R, self.ik._radial(p)), roll=float(q[ROLL]),
                               grip=float(q[5]), closed=bool(q[5] < 0.0), q=q.copy())
            elif mode == "sliders":
                self.slider = self.env.state().astype(np.float64)
            elif mode == "leader":
                if self.leader is None:
                    if not self.leader_port:
                        raise RuntimeError("Enter the leader arm's serial port first "
                                           "(COM3, /dev/tty.usbmodem..., /dev/ttyACM1).")
                    from ..leader import LeaderArm
                    self.log(f"connecting to the leader on {self.leader_port} ...")
                    self.leader = LeaderArm(self.leader_port, self.leader_id)
                    self.log("leader connected")
            elif mode == "expert":
                self.expert = ScriptedExpert(self.env, np.random.default_rng(self.seed))
                self.expert_tail = int(0.5 * self.cfg.fps)
            elif mode == "policy":
                self.load_policy()
                self.policy.reset(self.env)
                self.executor.reset(self.env.state())
            elif mode != "idle":
                raise ValueError(f"unknown mode {mode}")
            self.mode = mode
            if not quiet:
                self.log(f"mode: {mode}")
        except Exception as e:
            self.log(f"could not switch to {mode}: {e}")
            if mode != "keyboard":
                self.enter_mode("keyboard", quiet=True)

    def load_policy(self) -> None:
        from ..policy_eval import ChunkExecutor, LatencyModel, load_policy
        if not self.policy_spec:
            raise RuntimeError("Enter a policy: a checkpoint folder (.../pretrained_model) or 'expert'.")
        key = (self.policy_spec, self.policy_cfg["num_steps"])
        if self.policy is None or getattr(self, "_policy_key", None) != key:
            self.log(f"loading policy {self.policy_spec} ...")
            self.policy = load_policy(self.policy_spec, num_steps=self.policy_cfg["num_steps"])
            self._policy_key = key
            self.log(f"policy loaded, {self.policy.n_action_steps} actions per chunk")
        if hasattr(self.policy, "seed"):
            self.policy.seed, self.policy.mode = self.seed, self.policy_cfg["mode"]
        self.executor = ChunkExecutor(self.policy, LatencyModel.constant(self.policy_cfg["latency_ms"]),
                                      self.cfg.fps, self.policy_cfg["mode"], self.policy_cfg["threshold"])

    def control(self, obs) -> np.ndarray:
        dt = 1.0 / self.cfg.fps
        if self.mode == "keyboard":
            return self.keyboard_action(dt)
        if self.mode == "sliders":
            return self.slider.astype(np.float32)
        if self.mode == "leader":
            return self.leader.read()
        if self.mode == "expert":
            return self.expert.act()
        if self.mode == "policy":
            return self.executor.next_action(self.step, obs, self.env.task)
        return self.last_action

    def keyboard_action(self, dt: float) -> np.ndarray:
        kb, k = self.kb, self.keys
        if time.monotonic() - self.keys_t > KEY_TIMEOUT_S:
            k = set()
        fine = "fine" in k
        v = 0.04 if fine else 0.12                       # m/s
        w = math.radians(30 if fine else 90)             # rad/s
        axis = lambda a, b: float((a in k) - (b in k))   # noqa: E731
        kb["pos"] = np.clip(kb["pos"] + v * dt * np.array([axis("fwd", "back"), axis("left", "right"),
                                                           axis("up", "down")]), *WORKSPACE)
        kb["pitch"] = float(np.clip(kb["pitch"] + w * dt * axis("pitch_down", "pitch_up"),
                                    math.radians(-20), math.radians(110)))
        kb["roll"] = float(np.clip(kb["roll"] + w * dt * axis("roll_left", "roll_right"),
                                   self.ik.lo[ROLL], self.ik.hi[ROLL]))
        if kb.get("toggled"):                            # hold the current opening until Space
            target = GRIP_CLOSED if kb["closed"] else GRIP_OPEN
            kb["grip"] += float(np.clip(target - kb["grip"], -3.0 * dt, 3.0 * dt))
        q = kb["q"].copy()
        q[ROLL] = kb["roll"]
        q, err = self.ik.solve(q, kb["pos"], kb["pitch"], iters=25)
        if err > 0.01:                                   # out of reach: stop where the arm got to
            kb["pos"], _ = self.ik.fk(q)
        q[5] = kb["grip"]
        kb["q"] = q
        return self.env.jmap.to_lerobot(q)

    # ------------------------------------------------------------------ recording

    def ensure_recorder(self) -> bool:
        if self.rec is not None:
            return True
        if not self.has_lerobot:
            self.log("recording needs LeRobot: pip install 'lerobot>=0.4.4'")
            return False
        from ..recorder import Recorder, lerobot_home
        name = self.dataset_name or "sim_cube_box"
        self.dataset_name = name
        repo = name if "/" in name else f"local/{name}"
        root = Path(self.dataset_root) if self.dataset_root else lerobot_home() / repo
        try:
            self.rec = Recorder(name, self.cfg, self.env.task, root=root, resume=root.exists())
            self.saved_episodes = self.rec.num_episodes
            self.log(f"dataset {self.rec.root} ({self.saved_episodes} episodes so far)")
            return True
        except Exception as e:
            self.log(f"cannot open dataset {name}: {e}")
            self.rec = None
            return False

    def start_episode(self) -> None:
        if self.recording or not self.ensure_recorder():
            return
        self.new_scene()
        self.recording = True
        self.ep_frames = 0
        self._slow_warned = False
        self.log(f"recording episode {self.saved_episodes} (seed {self.seed})")

    def save_episode(self, reset: bool = True) -> bool:
        """Save the episode being recorded. Returns whether one was written."""
        if not self.recording:
            return False
        self.recording = False
        if self.ep_frames == 0:
            return False
        try:
            self.saved_episodes = self.rec.save_episode()
        except Exception as e:
            traceback.print_exc()
            self.log(f"could not save the episode ({e}); it was discarded, the dataset is unchanged")
            if reset:
                self.new_scene()
            return False
        # Finalize after every episode so the dataset on disk is always valid, even
        # if this window is closed or the process killed. The next episode reopens it.
        self.rec.finalize()
        self.rec = None
        self.log(f"saved episode {self.saved_episodes - 1} ({self.ep_frames} frames); "
                 f"{self.saved_episodes} in {self.dataset_name}")
        if reset:
            self.new_scene()
        return True

    def discard_episode(self, reset: bool = True) -> None:
        if not self.recording:
            return
        self.recording = False
        self.rec.discard()
        self.log("episode discarded")
        if reset:
            self.new_scene()

    def start_batch(self, n: int) -> None:
        if not self.ensure_recorder():
            return
        self.batch_left = int(n)
        self.batch_stats = [0, 0]
        self.mode = "expert"
        self.log(f"recording {n} expert demos ({'fast' if self.fast else 'real time'})")
        self.start_episode()

    def after_step(self) -> None:
        """Episode bookkeeping once per control step."""
        if self.recording:
            self.ep_frames += 1
        in_box = self.env.is_success()
        self.success_streak = self.success_streak + 1 if in_box else 0
        if self.mode == "expert" and self.expert is not None and self.expert.done:
            self.expert_tail -= 1
            if self.expert_tail <= 0 and self.batch_left > 0 and self.recording:
                ok = self.env.is_success()
                self.batch_stats[1] += 1
                if ok:
                    if self.save_episode(reset=False):
                        self.batch_stats[0] += 1
                        self.batch_left -= 1
                else:
                    self.discard_episode(reset=False)
                    self.log("expert missed; demo discarded")
                if self.batch_left > 0:
                    self.start_episode()
                else:
                    self.log(f"batch done: {self.batch_stats[0]} saved of {self.batch_stats[1]} tries")
                    self.new_scene()
                return
        if self.recording and self.batch_left == 0 and self.ep_frames >= self.cfg.max_steps:
            self.log(f"time limit {self.cfg.episode_s:g}s reached")
            self.save_episode()          # what lerobot-record does when an episode runs out of time

    # ------------------------------------------------------------------ commands

    def handle(self, c: dict) -> None:
        t = c.get("type")
        if t == "keys":
            self.keys = set(c.get("keys", []))
            self.keys_t = time.monotonic()
        elif t == "toggle_grip" and self.mode == "keyboard":
            self.kb["closed"] = not self.kb["closed"]
            self.kb["toggled"] = True
        elif t == "mode":
            if c.get("leader_port"):
                self.leader_port = c["leader_port"]
            if c.get("leader_id"):
                self.leader_id = c["leader_id"]
            if c.get("policy"):
                self.policy_spec = c["policy"]
            for key in ("latency_ms", "threshold"):
                if c.get(key) is not None:
                    self.policy_cfg[key] = float(c[key])
            if c.get("policy_mode"):
                self.policy_cfg["mode"] = c["policy_mode"]
            if "num_steps" in c:
                self.policy_cfg["num_steps"] = int(c["num_steps"]) if c["num_steps"] else None
            self.enter_mode(c["mode"])
        elif t == "joints" and self.mode == "sliders":
            self.slider = np.asarray(c["values"], dtype=np.float64)
        elif t == "reset":
            self.new_scene(c.get("seed"))
        elif t == "randomize":
            self.cfg = self.cfg.with_(randomize=c["level"])
            self.env.cfg = self.cfg
            self.log(f"randomization: {c['level']}")
            self.new_scene()
        elif t == "pause":
            self.paused = bool(c.get("value", not self.paused))
        elif t == "fast":
            self.fast = bool(c.get("value"))
        elif t == "orbit":
            self.cam.azimuth -= 0.35 * float(c.get("dx", 0))
            self.cam.elevation = float(np.clip(self.cam.elevation - 0.35 * float(c.get("dy", 0)), -89, 5))
        elif t == "zoom":
            self.cam.distance = float(np.clip(self.cam.distance * math.exp(0.001 * float(c.get("d", 0))), 0.2, 2.5))
        elif t == "view_reset":
            self.reset_view()
        elif t == "dataset":
            if self.recording:
                self.log("finish the current episode first")
                return
            if self.rec is not None:
                self.rec.finalize()
                self.rec = None
            self.dataset_name = (c.get("name") or "sim_cube_box").strip()
            if c.get("task"):
                self.env.task = c["task"].strip()
            self.ensure_recorder()
        elif t == "record":
            self.start_episode()
        elif t == "save":
            self.save_episode()
        elif t == "discard":
            self.discard_episode()
        elif t == "batch":
            self.start_batch(int(c.get("n", 10)))
        elif t == "stop_batch":
            self.batch_left = 0
            self.discard_episode()
        elif t == "quit":
            self.running = False
        else:
            self.log(f"unknown command {t}")

    # ------------------------------------------------------------------ main loop

    def publish(self, obs) -> None:
        if self._view_renderer is None:
            w, h = self.view_size
            self._view_renderer = mujoco.Renderer(self.env.model, h, w)
        r = self._view_renderer
        r.update_scene(self.env.data, self.cam)
        view = r.render()
        frames = {"view": jpeg(view, quality=82)}
        for key in ("camera1", "camera2"):
            if key in obs.images:
                frames[key] = jpeg(obs.images[key], size=(320, 240), quality=75)
        self.hub.publish(frames)

    def update_status(self, action) -> None:
        now = time.perf_counter()
        hz = (len(self._tick_times) - 1) / (self._tick_times[-1] - self._tick_times[0]) \
            if len(self._tick_times) > 2 else 0.0
        ex = self.executor if self.mode == "policy" else None
        if (self.recording and self.mode == "leader" and not self._slow_warned and not self.fast
                and len(self._tick_times) == self._tick_times.maxlen and hz < 0.9 * self.cfg.fps):
            # Keyboard, expert and policy all move in simulated time, but a person on
            # the leader arm moves in wall time: each tick is stored as 1/fps s, so
            # running slow records their motion sped up.
            self._slow_warned = True
            self.log(f"warning: the simulator runs at {hz:.0f} Hz, below {self.cfg.fps} Hz, so this "
                     f"demo plays back {self.cfg.fps / max(hz, 1):.1f}x faster than you moved. "
                     "Closing other apps that use the GPU helps")
        st = {
            "mode": self.mode, "paused": self.paused, "fast": self.fast,
            "seed": self.seed, "step": self.step, "sim_time": self.step / self.cfg.fps,
            "fps_target": self.cfg.fps, "fps_actual": round(hz, 1), "randomize": self.cfg.randomize,
            "task": self.env.task, "joint_names": list(FEATURE_NAMES), "limits": self.limits,
            "state": [round(float(x), 2) for x in self.env.state()],
            "action": [round(float(x), 2) for x in np.asarray(action)],
            "tcp": [round(float(x), 4) for x in self.env.tcp_pos()],
            "success": self.success_streak >= int(0.5 * self.cfg.fps), "cube_in_box": self.env.cube_in_box(),
            "recording": self.recording, "ep_frames": self.ep_frames, "ep_limit": self.cfg.max_steps,
            "dataset": {"name": self.dataset_name, "episodes": self.saved_episodes,
                        "open": self.rec is not None, "root": str(self.rec.root) if self.rec else None},
            "batch_left": self.batch_left, "batch_stats": self.batch_stats,
            "lerobot": self.has_lerobot,
            "leader": {"port": self.leader_port, "id": self.leader_id, "connected": self.leader is not None},
            "policy": {"spec": self.policy_spec, **self.policy_cfg,
                       "inferences": ex.inferences if ex else 0, "stalls": ex.stalls if ex else 0,
                       "waiting": bool(ex.waiting) if ex else False},
            "keyboard": None if self.kb is None else {
                "pos": [round(float(x), 3) for x in self.kb["pos"]],
                "pitch_deg": round(math.degrees(self.kb["pitch"]), 1), "closed": self.kb["closed"]},
            "log": list(self.log_lines),
            "time": now,
        }
        with self.status_lock:
            self.status = st

    def tick(self) -> None:
        now = time.perf_counter()
        ui_period = 0.25 if self.fast else self.ui_period
        if self.recording:
            ui_period = max(ui_period, 1.0 / VIEW_HZ_RECORDING)
        ui_due = now - self._last_ui >= ui_period
        preview_due = ui_due and now - self._last_cam_preview >= 1.0 / CAM_PREVIEW_HZ
        obs = self.env.observe(render=self.recording or self.mode == "policy" or preview_due)
        if preview_due:
            self._last_cam_preview = now
        action = self.control(obs)
        if self.recording:
            self.rec.add(obs, action)
        self.env.apply_action(action)
        self.last_action = action
        self.step += 1
        self.after_step()
        self._tick_times.append(time.perf_counter())
        if ui_due:
            self._last_ui = now
            self.publish(obs)
            self.update_status(action)

    def loop(self) -> None:
        dt = 1.0 / self.cfg.fps
        next_t = time.perf_counter()
        while self.running:
            try:
                while True:
                    self.handle(self.cmds.get_nowait())
            except queue.Empty:
                pass
            try:
                if self.paused:
                    self.publish(self.env.observe(render=True))
                    self.update_status(self.last_action)
                    time.sleep(0.05)
                    next_t = time.perf_counter()
                    continue
                self.tick()
            except Exception as e:                       # keep the app alive, report the error
                self.log(f"error: {e}")
                traceback.print_exc()
                if self.mode != "keyboard":
                    self.enter_mode("keyboard", quiet=True)
                time.sleep(0.2)
            if self.fast:
                next_t = time.perf_counter()
                continue
            next_t += dt
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:                          # fell behind: do not try to catch up
                next_t = time.perf_counter()

    def shutdown(self) -> None:
        self.running = False
        if self.recording:
            self.discard_episode(reset=False)
        if self.rec is not None:
            self.rec.finalize()
            self.rec = None
        if self.leader is not None:
            self.leader.close()
        self.hub.close()
        if self._view_renderer is not None:
            self._view_renderer.close()
        self.env.close()

    def snapshot(self) -> dict:
        with self.status_lock:
            return dict(self.status)


def make_handler(app: SimApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):        # keep the terminal for the app's own messages
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), STATIC_FILES["index.html"])
                elif path.startswith("/static/") and path[8:] in STATIC_FILES:
                    name = path[8:]
                    self._send(200, (STATIC / name).read_bytes(), STATIC_FILES[name])
                elif path == "/api/status":
                    self._send(200, json.dumps(app.snapshot()).encode(), "application/json")
                elif path.startswith("/stream/") and path[8:] in STREAMS:
                    self._stream(path[8:])
                else:
                    self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            if urlparse(self.path).path != "/api/cmd":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                cmd = json.loads(self.rfile.read(n) or b"{}")
                app.cmds.put(cmd)
                self._send(200, b'{"ok": true}', "application/json")
            except (ValueError, json.JSONDecodeError):
                self._send(400, b'{"ok": false}', "application/json")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _stream(self, name: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            while app.running and not app.hub.closed:
                seq, frames = app.hub.wait(seq)
                jpg = frames.get(name)
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")

    return Handler


def serve(app: SimApp, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, p), make_handler(app))
            port = p
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"No free port between {port} and {port + 19}.")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
    app.log(f"SO-101 simulator running at {url}  (Ctrl+C here, or Quit in the page, to stop)")
    if host not in ("127.0.0.1", "localhost"):
        app.log(f"listening on {host}: anyone on this network can open the page")
    app.new_scene(app.seed)
    app.enter_mode("keyboard", quiet=True)
    if app.policy_spec:
        app.enter_mode("policy")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        app.loop()
    except KeyboardInterrupt:
        app.log("stopping")
    finally:
        app.shutdown()
        server.shutdown()
        server.server_close()
