"""What a real SO-101 and its USB cameras add to a perfect simulator.

A noise scale of 1 is our estimate of the lab rig; 0 turns everything off and 2
doubles every amplitude for a stress test. Each episode draws its own errors
(calibration offsets, backlash, gains, camera exposure and delay) from a
generator seeded by the episode's seed, so a layout seed gives the same layout at
any noise scale, and the same noise at the same scale.

Joints, at scale 1:

- reading: the STS3215's 12-bit encoder (0.088 degrees per tick) plus 0.15 degrees
  of reading noise; the gripper 0.3 of its 0-100 units.
- calibration: the real arm's zero differs from the model's by up to 2 degrees per
  joint (2 units on the gripper). The arm moves to command + offset and reports
  position - offset, as a slightly wrong `lerobot-calibrate` would make it.
- backlash: 0.8 degrees of play in the gearbox; the output only follows a command
  once it has taken up the slack.
- servo gains: each joint's position gain is off by up to 15%.
- command jitter: 0.1 degrees per step.

Cameras, at scale 1:

- exposure and white balance: up to 15% brighter or darker, 6% per colour channel,
  plus 2% flicker per frame;
- sensor noise: 3 levels of 255 per pixel;
- MJPEG: frames go through JPEG at quality 70-95, as USB webcams send them;
- delay: frames arrive 1-2 control steps (33-67 ms) late, and 2% of frames are
  dropped (the previous one repeats);
- wrist camera mount: off by up to 2 mm and 2 degrees.

The scripted expert reads the true state and compensates for the calibration
offset, as a person teleoperating by eye would, so its demos still succeed; the
recorded observations carry all of the noise above.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

QUANT_DEG = 360.0 / 4096           # one STS3215 encoder tick

# Amplitudes at scale 1. Angles in degrees, gripper in LeRobot units (0-100).
STATE_SIGMA_DEG = 0.15
STATE_SIGMA_GRIPPER = 0.3
CALIB_DEG = 2.0
CALIB_GRIPPER = 2.0
BACKLASH_DEG = 0.8
KP_SPREAD = 0.15
ACTION_SIGMA_DEG = 0.1
EXPOSURE = 0.15
WHITE_BALANCE = 0.06
FLICKER = 0.02
PIXEL_SIGMA = 3.0
JPEG_QUALITY = (70, 95)
DELAY_STEPS = (0.5, 2.5)            # rounded: 1 or 2 steps at scale 1
DROP_PROB = 0.02
WRIST_CAM_POS_M = 0.002
WRIST_CAM_ROT_DEG = 2.0

try:
    import cv2
except ImportError:                  # JPEG is skipped without OpenCV
    cv2 = None


class EpisodeNoise:
    """One episode's draw of every error, and the functions that apply them.

    gripper_rad_per_unit converts the gripper's 0-100 units to simulator radians.
    """

    def __init__(self, scale: float, rng: np.random.Generator, cameras: tuple[str, ...],
                 gripper_rad_per_unit: float):
        self.scale = s = float(scale)
        self.rng = rng
        deg = math.radians(1.0)
        per_joint = np.array([deg] * 5 + [gripper_rad_per_unit])
        u = lambda amp: rng.uniform(-1.0, 1.0, 6) * amp  # noqa: E731

        self.offset = u(s * np.array([CALIB_DEG] * 5 + [CALIB_GRIPPER])) * per_joint
        self.backlash = s * BACKLASH_DEG * deg * np.r_[np.ones(5), 0.0]
        self.kp_scale = 1.0 + u(s * KP_SPREAD)
        self.kp_scale = np.clip(self.kp_scale, 0.3, None)
        self.action_sigma = s * ACTION_SIGMA_DEG * deg * np.r_[np.ones(5), 0.0]
        self.state_sigma = s * np.array([STATE_SIGMA_DEG] * 5 + [STATE_SIGMA_GRIPPER]) * per_joint
        self.wrist_cam_pos = rng.uniform(-1, 1, 3) * s * WRIST_CAM_POS_M
        self.wrist_cam_rot = rng.uniform(-1, 1, 3) * s * math.radians(WRIST_CAM_ROT_DEG)

        self.cameras = {}
        for key in cameras:
            gain = (1.0 + rng.uniform(-1, 1) * s * EXPOSURE) * (1.0 + rng.uniform(-1, 1, 3) * s * WHITE_BALANCE)
            delay = int(round(rng.uniform(*DELAY_STEPS) * s)) if s > 0 else 0
            quality = int(rng.integers(JPEG_QUALITY[0], JPEG_QUALITY[1] + 1))
            self.cameras[key] = {"gain": gain.astype(np.float32), "delay": delay, "quality": quality,
                                 "frames": deque(maxlen=delay + 1)}
        self._held: np.ndarray | None = None     # backlash: where each gear train sits

    @property
    def active(self) -> bool:
        return self.scale > 0

    # ------------------------------------------------------------------ joints

    def command(self, target: np.ndarray) -> np.ndarray:
        """Simulator joint target a commanded target turns into on the real arm."""
        t = np.asarray(target, float) + self.offset
        if self._held is None:
            self._held = t.copy()
        half = 0.5 * self.backlash
        self._held = np.clip(self._held, t - half, t + half)
        return self._held + self.rng.normal(0.0, 1.0, 6) * self.action_sigma

    def compensate(self, q: np.ndarray) -> np.ndarray:
        """The command that puts the arm at q (ignoring backlash and jitter)."""
        return np.asarray(q, float) - self.offset

    def reading(self, q_true: np.ndarray) -> np.ndarray:
        """What the encoders report for the true joint angles, in simulator radians."""
        return np.asarray(q_true, float) - self.offset + self.rng.normal(0.0, 1.0, 6) * self.state_sigma

    @staticmethod
    def quantize(state_lerobot: np.ndarray) -> np.ndarray:
        out = np.array(state_lerobot, dtype=np.float64)
        out[:5] = np.round(out[:5] / QUANT_DEG) * QUANT_DEG
        return out

    # ------------------------------------------------------------------ cameras

    def image(self, key: str, img: np.ndarray) -> np.ndarray:
        """A rendered frame as the camera would deliver it: exposed, noisy, compressed, late."""
        cam = self.cameras[key]
        frames = cam["frames"]
        if frames and self.rng.random() < DROP_PROB * self.scale:
            frames.append(frames[-1])            # dropped: the previous frame again
        else:
            frames.append(self._degrade(img, cam))
        while len(frames) < frames.maxlen:       # first frame of the episode
            frames.appendleft(frames[0])
        return frames[0]

    def _degrade(self, img: np.ndarray, cam: dict) -> np.ndarray:
        s = self.scale
        x = img.astype(np.float32)
        x *= cam["gain"] * (1.0 + self.rng.uniform(-1, 1) * s * FLICKER)
        x += self.rng.standard_normal(x.shape, dtype=np.float32) * (s * PIXEL_SIGMA)
        out = np.clip(x, 0, 255).astype(np.uint8)
        if cv2 is not None:
            ok, buf = cv2.imencode(".jpg", out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, cam["quality"]])
            if ok:
                out = np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])
        return out

    def summary(self) -> dict:
        """The episode's draw, for logs and result records."""
        return {
            "scale": self.scale,
            "calibration_offset": [round(float(v), 5) for v in self.offset],
            "kp_scale": [round(float(v), 3) for v in self.kp_scale],
            "camera_delay_steps": {k: c["delay"] for k, c in self.cameras.items()},
            "jpeg_quality": {k: c["quality"] for k, c in self.cameras.items()},
        }


__all__ = ["EpisodeNoise", "QUANT_DEG"]
