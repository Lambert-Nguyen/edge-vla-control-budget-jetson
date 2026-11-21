#!/usr/bin/env bash
# SO-101 simulator launcher for macOS and Linux. Extra options pass through, e.g.
#   ./run_app.sh --randomize full --dataset my_demos
# The first run creates .venv here and installs everything (a few minutes, ~2 GB).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Setting up the Python environment (first run only)..."
  PY="${PYTHON:-}"
  if [ -z "$PY" ]; then
    for cand in python3.12 python3.13 python3; do
      if command -v "$cand" >/dev/null 2>&1 &&
         "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 12) else 1)' 2>/dev/null; then
        PY="$cand"; break
      fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "Python 3.12 or newer was not found."
    echo "  macOS: brew install python@3.12   (or the installer from https://www.python.org/downloads/)"
    echo "  Linux: your package manager, or uv python install 3.12"
    exit 1
  fi
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  if [ "$(uname -s)-$(uname -m)" = "Darwin-x86_64" ]; then
    echo "Intel Mac: PyTorch no longer ships for it, so LeRobot is skipped."
    echo "The simulator and teleop work; recording and policy evaluation do not."
    .venv/bin/python -m pip install -e .
  else
    .venv/bin/python -m pip install -e ".[record]"
  fi
fi

# SO101_SIM_COMMAND runs another command instead of the app, e.g. check or generate.
exec .venv/bin/python -m so101_sim "${SO101_SIM_COMMAND:-app}" "$@"
