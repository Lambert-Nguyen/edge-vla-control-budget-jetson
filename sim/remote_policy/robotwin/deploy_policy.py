"""RoboTwin hooks for the remote policy link.

RoboTwin calls eval() once per control step. When the local action queue is
empty, the adapter sends the observation to the policy server on the Jetson
and gets back one action chunk. Then it plays exc_action_size actions of the
chunk, one per step, before it asks again. The round trip times are saved
at the end of each episode, so a simulation run also reports the link.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sim.remote_policy.client import RemotePolicy  # noqa: E402


def shrink(img: np.ndarray, size: int) -> np.ndarray:
    """Scale an image down so its longer side is `size`, keeping its shape.

    No padding here: the policy pads to a square itself (Hy-VLA resize_with_pad),
    so the result is the same as sending the full frame, with fewer bytes on Wi-Fi.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if size <= 0 or scale >= 1:
        return img
    import cv2

    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)


def encode_obs(observation: dict[str, Any], image_size: int = 0) -> dict[str, np.ndarray]:
    """Three uint8 HxWx3 images, the 16-d end-effector state and the 14-d joints."""
    ep = observation["endpose"]
    state = np.array([*ep["left_endpose"], ep["left_gripper"], *ep["right_endpose"], ep["right_gripper"]],
                     dtype=np.float32)
    cams = observation["observation"]
    return {
        "head": shrink(cams["head_camera"]["rgb"], image_size),
        "left": shrink(cams["left_camera"]["rgb"], image_size),
        "right": shrink(cams["right_camera"]["rgb"], image_size),
        "state": state,
        "joints": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    }


class RemoteModel:
    def __init__(self, usr_args: dict[str, Any]):
        self.host = usr_args.get("host", "sjsujetson-36.local")
        self.port = int(usr_args.get("port", 8765))
        self.action_type = usr_args.get("action_type", "ee")
        self.exc = int(usr_args.get("exc_action_size", 7))
        self.image_size = int(usr_args.get("image_size", 224))
        self.log_dir = Path(usr_args.get("log_dir", "eval_result/remote_link"))
        self.client = RemotePolicy(self.host, self.port)
        self.info = self.client.info()
        self.queue: deque = deque()
        self.episode = 0
        print(f"[remote] policy server {self.host}:{self.port}: {self.info}", flush=True)

    def step(self, TASK_ENV, observation) -> None:
        if not self.queue:
            chunk = self.client.infer(encode_obs(observation, self.image_size), TASK_ENV.get_instruction())
            self.queue.extend(np.asarray(chunk)[: self.exc])
        TASK_ENV.take_action(self.queue.popleft(), action_type=self.action_type)

    def reset(self) -> None:
        if self.client.round_trip_ms:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out = self.log_dir / f"episode_{self.episode:04d}.json"
            out.write_text(json.dumps({"host": self.host, "server": self.info,
                                       "round_trip_ms": self.client.round_trip_ms,
                                       "server_ms": self.client.server_ms,
                                       "time": time.time()}) + "\n")
            self.episode += 1
        self.client.round_trip_ms, self.client.server_ms = [], []
        self.queue.clear()
        self.client.reset()


def get_model(usr_args: dict[str, Any]) -> RemoteModel:
    return RemoteModel(usr_args)


def eval(TASK_ENV, model: RemoteModel, observation: dict[str, Any]) -> None:  # noqa: A001
    model.step(TASK_ENV, observation)


def reset_model(model: RemoteModel) -> None:
    model.reset()
