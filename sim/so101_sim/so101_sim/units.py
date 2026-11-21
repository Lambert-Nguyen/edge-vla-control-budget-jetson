"""Conversion between simulator joint angles and LeRobot's SO-101 units.

LeRobot 0.4.4 and 0.5.1 both default to ``use_degrees=True`` for so101_follower and
so101_leader, so a real dataset stores:

- the five arm joints in degrees, measured from the middle of each joint's
  calibrated range;
- the gripper on a 0-100 scale across its calibrated range.

The Menagerie model puts each arm joint's zero at the middle of its range too, so
arm joints convert as plain degrees. The gripper maps its simulated range linearly
onto 0-100, with 0 at the closed end.

Both assumptions should be checked once against the real arm: put the follower in
a pose, read its values with ``lerobot`` and compare with the simulator in the
same pose. If a joint is mirrored or offset, fix it in a joint-map JSON file
(see ``JointMap.load``) rather than in code.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Simulated gripper range, radians (so101.xml): closed at the low end.
GRIPPER_LO = -0.174533
GRIPPER_HI = 1.7453292


@dataclass
class JointMap:
    sign: list[float] = field(default_factory=lambda: [1.0] * 5)
    offset_deg: list[float] = field(default_factory=lambda: [0.0] * 5)
    gripper_closed: float = 0.0      # LeRobot gripper value at the simulated closed end
    gripper_open: float = 100.0      # LeRobot gripper value at the simulated open end

    @classmethod
    def load(cls, path: str | Path | None) -> "JointMap":
        """Read a joint map from JSON. Missing keys keep their defaults."""
        if not path:
            return cls()
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    # LeRobot value = sign * degrees(sim) + offset
    def to_lerobot(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        out = np.empty(6, dtype=np.float32)
        out[:5] = np.asarray(self.sign) * np.degrees(q[:5]) + np.asarray(self.offset_deg)
        frac = (q[5] - GRIPPER_LO) / (GRIPPER_HI - GRIPPER_LO)
        out[5] = self.gripper_closed + frac * (self.gripper_open - self.gripper_closed)
        return out

    def to_sim(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        q = np.empty(6, dtype=np.float64)
        q[:5] = np.radians((v[:5] - np.asarray(self.offset_deg)) / np.asarray(self.sign))
        frac = (v[5] - self.gripper_closed) / (self.gripper_open - self.gripper_closed)
        q[5] = GRIPPER_LO + frac * (GRIPPER_HI - GRIPPER_LO)
        return q


DEFAULT_MAP = JointMap()


def deg(x: float) -> float:
    return math.degrees(x)
