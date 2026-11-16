import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Headless Linux CI: render with EGL unless the caller chose a backend.
if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")
