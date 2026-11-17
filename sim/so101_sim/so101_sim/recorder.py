"""Write simulated episodes as LeRobot datasets.

The features match what lerobot-record writes for the real follower in
hardware/so101_lab.sh: ``observation.state`` and ``action`` with the six
``<joint>.pos`` names, ``observation.images.camera1`` / ``camera2`` as 640x480
video, robot_type ``so101_follower``, and the task string. lerobot-train can
therefore read simulated and real datasets the same way.

Datasets go to ``$HF_LEROBOT_HOME/local/<name>`` by default
(``~/.cache/huggingface/lerobot/local/<name>``), the same place the lab script
uses on the Jetson. LeRobot is imported lazily so that the rest of the simulator
runs without it.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import numpy as np

from .config import FEATURE_NAMES, ROBOT_TYPE, SimConfig
from .env import Obs


class LeRobotMissing(RuntimeError):
    pass


def _accepted(fn, kwargs: dict) -> dict:
    """Keep only the keyword arguments this LeRobot version's function accepts."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def require_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    except ImportError as e:
        raise LeRobotMissing(
            "Recording needs LeRobot. Install it into this environment with\n"
            "    pip install 'lerobot>=0.4.4'\n"
            "(the launcher scripts do this for you)."
        ) from e
    import lerobot
    return lerobot


def lerobot_home() -> Path:
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        return Path(HF_LEROBOT_HOME)
    except ImportError:
        import os
        base = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        return Path(os.getenv("HF_LEROBOT_HOME", str(Path(base) / "lerobot"))).expanduser()


def dataset_features(cfg: SimConfig, use_videos: bool = True) -> dict:
    names = list(FEATURE_NAMES)
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
    }
    for cam in cfg.cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (cfg.height, cfg.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class Recorder:
    """Thin wrapper over LeRobotDataset: add frames, then save or discard the episode."""

    def __init__(self, name: str, cfg: SimConfig, task: str, root: str | Path | None = None,
                 resume: bool = False, overwrite: bool = False, use_videos: bool = True,
                 vcodec: str = "libsvtav1"):
        require_lerobot()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.cfg, self.task = cfg, task
        self.repo_id = name if "/" in name else f"local/{name}"
        self.root = Path(root) if root else lerobot_home() / self.repo_id
        if self.root.exists() and not resume:
            if not overwrite:
                raise FileExistsError(f"{self.root} already exists. Pick another name, "
                                      "or pass --resume to add episodes, or --overwrite.")
            shutil.rmtree(self.root)
        threads = 4 * len(cfg.cameras)
        if resume and self.root.exists():
            if hasattr(LeRobotDataset, "resume"):            # LeRobot >= 0.5
                self.ds = LeRobotDataset.resume(**_accepted(LeRobotDataset.resume, dict(
                    repo_id=self.repo_id, root=self.root, vcodec=vcodec, image_writer_threads=threads)))
            else:                                            # LeRobot 0.4.x, as lerobot-record --resume
                self.ds = LeRobotDataset(**_accepted(LeRobotDataset.__init__, dict(
                    repo_id=self.repo_id, root=self.root, vcodec=vcodec)))
                self.ds.start_image_writer(num_threads=threads)
            self._check_compatible(dataset_features(cfg, use_videos))
        else:
            self.ds = LeRobotDataset.create(**_accepted(LeRobotDataset.create, dict(
                repo_id=self.repo_id, fps=cfg.fps, features=dataset_features(cfg, use_videos),
                root=self.root, robot_type=ROBOT_TYPE, use_videos=use_videos,
                image_writer_threads=threads, vcodec=vcodec)))
        self.frames_in_episode = 0

    def _check_compatible(self, features: dict) -> None:
        meta = self.ds.meta
        if meta.fps != self.cfg.fps:
            raise ValueError(f"{self.root} was recorded at {meta.fps} fps, not {self.cfg.fps}")
        for key, ft in features.items():
            have = meta.features.get(key)
            if have is None or tuple(have["shape"]) != tuple(ft["shape"]):
                raise ValueError(f"{self.root} does not match this simulator: feature {key} is "
                                 f"{None if have is None else have['shape']}, expected {ft['shape']}")

    @property
    def num_episodes(self) -> int:
        return int(self.ds.meta.total_episodes)

    def add(self, obs: Obs, action: np.ndarray) -> None:
        frame = {
            "observation.state": np.asarray(obs.state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "task": self.task,
        }
        for cam in self.cfg.cameras:
            frame[f"observation.images.{cam}"] = obs.images[cam]
        self.ds.add_frame(frame)
        self.frames_in_episode += 1

    def save_episode(self) -> int:
        """Encode and write the episode. Returns the number of episodes saved so far.

        Videos are encoded one camera after the other. LeRobot's default spawns a
        process pool per episode, which re-imports torch in every worker: about 50%
        slower for two 640x480 cameras on an M-series Mac, and the pool can die on a
        cold start. If saving fails anyway, the episode is dropped before the error
        is raised: LeRobot has already half-consumed its buffer by then, and the
        next episode would otherwise be appended to the broken one.
        """
        if self.frames_in_episode == 0:
            return self.num_episodes
        try:
            self.ds.save_episode(**_accepted(self.ds.save_episode, dict(parallel_encoding=False)))
        except Exception:
            self.frames_in_episode = 0
            try:
                self.ds.clear_episode_buffer()
            except Exception:
                pass
            raise
        self.frames_in_episode = 0
        return self.num_episodes

    def discard(self) -> None:
        if self.frames_in_episode:
            self.ds.clear_episode_buffer()
        self.frames_in_episode = 0

    def finalize(self) -> None:
        self.discard()
        self.ds.finalize()
