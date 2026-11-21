import ctypes.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Headless Linux CI: render with EGL unless the caller chose a backend. Only when
# libEGL is installed: with MUJOCO_GL=egl and no libEGL, `import mujoco` itself
# fails, even for the tests that never render.
if (sys.platform.startswith("linux") and "DISPLAY" not in os.environ
        and ctypes.util.find_library("EGL")):
    os.environ.setdefault("MUJOCO_GL", "egl")
