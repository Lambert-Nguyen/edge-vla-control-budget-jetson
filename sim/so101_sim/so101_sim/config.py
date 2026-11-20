"""Settings shared by the simulator, the app, the expert and the recorder.

Defaults mirror the team's lab rig in hardware/so101_lab.sh: an SO-101 follower,
camera1 = front camera, camera2 = wrist camera, 640x480 images, a 30 Hz control
rate and the instruction "Pick up the cube and put it in the box."
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets" / "so101"
SCENE_XML = ASSETS_DIR / "scene_cube_box.xml"

# Joint order and names match LeRobot's so101_follower ("shoulder_pan.pos", ...).
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_JOINTS = JOINT_NAMES[:5]
FEATURE_NAMES = tuple(f"{j}.pos" for j in JOINT_NAMES)

DEFAULT_TASK = "Pick up the cube and put it in the box."
ROBOT_TYPE = "so101_follower"

# Randomization levels, from least to most.
RANDOMIZE_LEVELS = ("none", "light", "full")

# Start pose of every episode, in LeRobot units (degrees, gripper 0-100): the arm
# raised, jaws about 16 cm above the table and pointing down at it, clear of the
# cube and box spawn areas. The real arm's folded rest pose, roughly
# (0, -100, 95, 70, 0, 0), puts the jaws inside the cube spawn area.
HOME_POSE = (0.0, -40.0, 20.0, 70.0, 0.0, 20.0)

# How far each episode's start pose may stray from the home pose, LeRobot units,
# at start_spread = 1. On the real arm an episode starts wherever the arm is, so a
# policy should not learn one start pose. Draws that bring the jaws within
# START_MIN_HEIGHT of the table, or the arm into the cube or box, are redrawn.
START_SPREAD = (30.0, 15.0, 20.0, 20.0, 30.0, 20.0)
START_MIN_HEIGHT = 0.10      # m, jaws over the table

# Camera1 positions (position, target), metres. "top" is the default since
# 2026-11-20; "front" is what the first datasets used: 62 cm in front of the base
# and 42 cm up, looking at the table 20 cm in front of it.
CAMERA1_PRESETS = {
    "top": ((0.50, 0.0, 0.70), (0.20, 0.0, 0.0)),
    "front": ((0.62, 0.0, 0.42), (0.20, 0.0, 0.0)),
}

# Objects the task can use (see objects.py). "cube" alone is the original task.
OBJECT_KINDS = ("cube", "block", "cylinder", "ball", "marker")

# What is on the table. "single": one object. "colors": three objects of one kind
# in different colours, and the instruction names the colour ("the red cube").
# "kinds": three different kinds in one colour, and the instruction names the kind
# ("the ball"). "mixed": one of the three, drawn per episode.
SCENES = ("single", "colors", "kinds", "mixed")

# Named noise scales: 0 is a perfect robot and camera, 1 our estimate of a real
# SO-101 follower with USB cameras (see noise.py), 2 a stress test.
NOISE_LEVELS = {"none": 0.0, "real": 1.0, "heavy": 2.0}


def parse_noise(text: str | float) -> float:
    """A noise scale from a level name or a number."""
    if isinstance(text, (int, float)):
        return float(text)
    text = str(text).strip().lower()
    if text in NOISE_LEVELS:
        return NOISE_LEVELS[text]
    value = float(text)
    if value < 0:
        raise ValueError(f"noise scale must be >= 0, got {value}")
    return value


def noise_label(scale: float) -> str:
    for name, value in NOISE_LEVELS.items():
        if abs(scale - value) < 1e-9:
            return name
    return f"{scale:g}"


@dataclass(frozen=True)
class TaskConfig:
    instruction: str = DEFAULT_TASK
    # Cube spawn area, metres in the world frame (arm base at the origin, arm reaches along +x).
    cube_x: tuple[float, float] = (0.16, 0.26)
    cube_y: tuple[float, float] = (-0.10, 0.10)
    cube_yaw_deg: tuple[float, float] = (-45.0, 45.0)
    # Box centre spawn area. The side (left or right of the arm) is chosen at random,
    # then y is drawn from this range and given that side's sign.
    box_x: tuple[float, float] = (0.15, 0.24)
    box_abs_y: tuple[float, float] = (0.13, 0.19)
    # Minimum gap between the cube and the box wall, metres.
    clearance: float = 0.03


@dataclass(frozen=True)
class SimConfig:
    fps: int = 30
    substeps: int = 10               # physics steps per control step: 1/300 s
    width: int = 640
    height: int = 480
    cameras: tuple[str, ...] = ("camera1", "camera2")
    randomize: str = "light"
    episode_s: float = 20.0
    # Largest joint move per control step, degrees, like lerobot-record's
    # --robot.max_relative_target (MAX_STEP in so101_lab.sh). None turns it off.
    max_step_deg: float | None = None
    # Where camera1 sits and what it looks at. Change these to match your stand.
    # camera1 is a top view, as in lerobot/smolvla_base's training data: 70 cm up
    # and 30 cm in front of the workspace, tilted 67 degrees down and back toward
    # the arm, which is at the top of the image. Straight down, the raised arm
    # covers the objects. Datasets before 2026-11-20 used the "front" preset.
    camera1_pos: tuple[float, float, float] = (0.50, 0.0, 0.70)
    camera1_target: tuple[float, float, float] = (0.20, 0.0, 0.0)
    camera1_fovy: float = 52.0      # a Brio at 78 degrees diagonal, cropped to 4:3
    home_pose: tuple[float, ...] = HOME_POSE   # start pose, LeRobot units
    # Scale on START_SPREAD for the random start pose: 0 starts every episode at
    # home_pose. Randomize level "none" always starts at home_pose.
    start_spread: float = 1.0
    # Which objects an episode may use; one is drawn per episode. ("cube",) is the
    # original 3 cm cube at every randomize level.
    objects: tuple[str, ...] = ("cube",)
    scene: str = "single"            # see SCENES
    # Sensor, actuation and camera noise scale (noise.py): 0 off, 1 a real arm.
    noise: float = 0.0
    task: TaskConfig = field(default_factory=TaskConfig)

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def max_steps(self) -> int:
        return int(round(self.episode_s * self.fps))
