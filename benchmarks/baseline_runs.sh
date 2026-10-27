#!/usr/bin/env bash
# Baseline records in the sweep setting, MAXN SUPER (mode 2) with jetson_clocks
# on, with the board put back afterwards.
#
#   first engine   the stand-in model in PyTorch FP16 and as a TensorRT FP16
#                  engine built on this board, at 224 px and 1344 px. Same
#                  iteration counts as the 2026-09-28 pinned checks.
#   smolvla        SmolVLA chunk timing with 2 or 1 synthetic 640x480 cameras
#                  and 10 or 5 denoising steps. The same four points as the
#                  2026-10-20 bring-up, which ran with the clocks scaling.
#
# Run it as your normal user, from a terminal:
#
#   bash benchmarks/baseline_runs.sh
#   bash benchmarks/baseline_runs.sh --dry-run   # no sudo, tiny runs, output deleted
#
# It asks for your sudo password once. Only nvpmodel and jetson_clocks run as
# root. On exit, including Ctrl-C or a failure, it restores the power mode and
# clock state it started from and checks that the GPU clock scales again.
#
# Takes about 10 minutes. Close what you can and tell the team first, since
# other load shows up in the worst-case latency. Apps that draw with the GPU
# matter most: the script lists any that are open and asks before it goes on.
# The fan may speed up while the clocks are pinned. Records land in results/
# and are not committed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/lerobot-py310-cuda"
PY="$VENV/bin/python"
GPU="$(ls -d /sys/class/devfreq/*.gpu | head -1)"
SWEEP_MODE=2
COOL_S=5
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

say() { printf '%s\n' "$*"; }
mode_id() { nvpmodel -q 2>/dev/null | awk '/^[0-9]+$/ {print $1}' | tail -1; }
mode_name() { nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: //p'; }
gpu_pinned() { [ "$(cat "$GPU/min_freq")" = "$(cat "$GPU/max_freq")" ]; }
gpu_range() {
  printf '%s-%s MHz' "$(( $(cat "$GPU/min_freq") / 1000000 ))" "$(( $(cat "$GPU/max_freq") / 1000000 ))"
}

# Apps with the GPU open share it with the timed runs. On 2026-10-26 one
# desktop app that renders with the GPU kept it about 60% busy while idle.
# With it open, PyTorch at 1344 px was about 50% slower at p50 and 224 px tripled
# at p99. The desktop itself (shell, X, portal) stays, so it is not listed.
gpu_apps() {
  local p comm
  for p in /proc/[0-9]*; do
    comm="$(cat "$p/comm" 2>/dev/null)" || continue
    case "$comm" in gnome-shell|Xorg|Xwayland|xdg-desktop-por*) continue ;; esac
    if ls -l "$p/fd" 2>/dev/null | grep -qE '/dev/nvgpu/|/dev/dri/renderD'; then
      printf '%s\n' "$comm"
    fi
  done | sort | uniq -c | awk '{print $2 " (" $1 " processes)"}'
}

gpu_busy() {  # percent of 2 s the GPU was busy, at whatever clock it ran
  local load="$GPU/device/load" sum=0 i
  [ -r "$load" ] || { printf '?'; return; }
  for i in 1 2 3 4 5 6 7 8; do sum=$((sum + $(cat "$load"))); sleep 0.25; done
  printf '%s' "$((sum / 80))"
}

# --- preflight ---------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
  say "Run this as your normal user, not with sudo. It asks for the password itself."
  exit 1
fi
if [ ! -x "$PY" ]; then
  say "No CUDA environment at $VENV. See docs/guides/03-python-environments.md."
  exit 1
fi
dirty="$(git -C "$REPO" status --porcelain -- . ':(exclude)results')"
if [ -n "$dirty" ] && [ "$DRY_RUN" -eq 0 ]; then
  say "The repo has uncommitted changes, and records from a dirty tree cannot be"
  say "reproduced. Commit or stash these first:"
  say "$dirty"
  exit 1
fi
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$VENV/cudss-lib:${LD_LIBRARY_PATH:-}"
# SmolVLA loads from the local cache. Never wait on the network mid-run.
export HF_HUB_OFFLINE=1
if ! "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  say "CUDA torch is not available in $VENV."
  exit 1
fi
if ! "$PY" -c 'import tensorrt' 2>/dev/null; then
  say "tensorrt is not importable. Install the system package: sudo apt install python3-libnvinfer"
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  DRY_OUT="$(mktemp -d)"
  say "== DRY RUN: no sudo, no mode changes, tiny runs in $DRY_OUT, deleted at exit"
else
  LOG="$HOME/baseline-runs-$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "$LOG") 2>&1
fi

say "== board before: $(mode_name) (mode $(mode_id)), GPU $(gpu_range), load $(cut -d' ' -f1-3 /proc/loadavg)"
say "== busiest processes right now, which add noise:"
ps -eo %cpu,comm --sort=-%cpu | sed -n '2,6p' | sed 's/^/     /'
say "== GPU busy $(gpu_busy)% of the last 2 s, with nothing of ours running"
apps="$(gpu_apps)"
if [ -n "$apps" ]; then
  say "[warn] these apps have the GPU open and will share it with the timed runs:"
  printf '%s\n' "$apps" | sed 's/^/       /'
  say "       Close them first. With one open, p99 latency tripled on 2026-10-26."
  if [ "$DRY_RUN" -eq 0 ] && [ -t 0 ]; then
    read -r -p "       Continue anyway? [y/N] " answer
    case "$answer" in y|Y|yes) ;; *) say "Stopped. Nothing was changed."; exit 1 ;; esac
  fi
fi

# --- save the board state and promise to restore it --------------------------
ORIG_MODE="$(mode_id)"
# Made only once sudo works, so stopping at the password prompt leaves nothing.
STATE_DIR=""
STATE=""
restored=0

restore() {
  [ "$restored" -eq 1 ] && return
  restored=1
  say ""
  say "== restoring the board"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "   (dry run) would switch back to mode $ORIG_MODE and restore the saved clocks"
    rm -rf "$DRY_OUT"
    return
  fi
  sudo -v || true
  echo no | sudo nvpmodel -m "$ORIG_MODE" >/dev/null || say "[warn] could not switch back to mode $ORIG_MODE"
  sudo jetson_clocks --restore "$STATE" >/dev/null || say "[warn] jetson_clocks --restore failed"
  sleep 2
  if gpu_pinned; then
    say "[warn] the GPU clock is still pinned ($(gpu_range)). Run"
    say "       sudo jetson_clocks --restore $STATE"
    say "       or reboot the board."
  else
    say "   $(mode_name) (mode $(mode_id), was $ORIG_MODE), GPU scaling again: $(gpu_range)"
    sudo rm -rf "$STATE_DIR"
  fi
}

if [ "$DRY_RUN" -eq 0 ]; then
  say ""
  say "sudo is needed for nvpmodel and jetson_clocks:"
  sudo -v
  STATE_DIR="$(mktemp -d)"
  STATE="$STATE_DIR/l4t_dfs.conf"
  sudo jetson_clocks --store "$STATE"
fi
trap restore EXIT
trap 'exit 130' INT TERM

# --- the sweep setting --------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
  say "== (dry run) would switch to mode $SWEEP_MODE and pin the clocks"
else
  # Answer no if nvpmodel ever asks to reboot. The mode check below catches it.
  echo no | sudo nvpmodel -m "$SWEEP_MODE" >/dev/null
  if [ "$(mode_id)" != "$SWEEP_MODE" ]; then
    say "[FAIL] nvpmodel did not switch to mode $SWEEP_MODE. It may need a reboot for that mode."
    exit 1
  fi
  sudo jetson_clocks >/dev/null
  sleep 3
  if ! gpu_pinned; then
    say "[FAIL] jetson_clocks did not pin the GPU clock ($(gpu_range))."
    exit 1
  fi
  say ""
  say "== $(mode_name) (mode $SWEEP_MODE), clocks pinned, GPU $(gpu_range)"
fi

written=()
keep() {  # harness output: remember the record it wrote
  local path
  path="$(printf '%s\n' "$1" | sed -n 's/^wrote //p' | tail -1)"
  case "$path" in /*) ;; *) path="$REPO/$path" ;; esac
  written+=("$path")
}

latency() {  # label, harness args...
  local label="$1" out
  shift
  if [ "$DRY_RUN" -eq 1 ]; then
    set -- "$@" --iters 30 --warmup 5 --out "$DRY_OUT/$label.json"
  fi
  if ! out="$("$PY" "$REPO/benchmarks/hello_latency.py" --label "$label" "$@" 2>&1)"; then
    printf '%s\n' "$out"
    say "[FAIL] harness run $label failed"
    exit 1
  fi
  printf '%s\n' "$out" | grep -E "^$label |^  engine|^  vs PyTorch|^  p50|^  board power|^  gpu clock|^\[warn\]" \
    | sed 's/^/   /' || true
  keep "$out"
  sleep "$COOL_S"
}

smolvla() {  # label, hello_smolvla args...
  local label="$1" out json
  shift
  if [ "$DRY_RUN" -eq 1 ]; then
    json="$DRY_OUT/$label.json"
    set -- "$@" --iters 2 --warmup 1
  else
    json="$REPO/results/$(date +%Y%m%d-%H%M%S)_$label.json"
    set -- "$@" --iters 100
  fi
  if ! out="$("$PY" "$REPO/hardware/hello_smolvla.py" "$@" --json "$json" 2>&1)"; then
    printf '%s\n' "$out"
    say "[FAIL] SmolVLA run $label failed"
    exit 1
  fi
  say "   $label"
  printf '%s\n' "$out" | grep -E "^  chunk inference|^  queued step|^  board power|^  gpu clock|^\[warn\]|^\[FAIL\]" \
    | sed 's/^/   /' || true
  keep "$out"
  sleep "$COOL_S"
}

# --- first TensorRT engine against PyTorch -------------------------------------
say ""
say "== stand-in model, PyTorch FP16 against a TensorRT FP16 engine"
for r in 224 1344; do
  iters=3000
  [ "$r" = 1344 ] && iters=1000
  latency "maxn-pinned-torch-$r" --runtime pytorch --resolution "$r" --iters "$iters"
  latency "maxn-pinned-trt-$r" --runtime tensorrt --resolution "$r" --iters "$iters"
done

# --- SmolVLA, the four bring-up points ------------------------------------------
say ""
say "== SmolVLA, synthetic 640x480 frames, 50-action chunks"
smolvla "maxn-pinned-smolvla-2cam-10step" --cameras 2
smolvla "maxn-pinned-smolvla-1cam-10step" --cameras 1
smolvla "maxn-pinned-smolvla-2cam-5step" --cameras 2 --steps 5
smolvla "maxn-pinned-smolvla-1cam-5step" --cameras 1 --steps 5

# --- summary ------------------------------------------------------------------
say ""
say "== summary"
"$PY" - "${written[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
print(f"   {'label':<34}{'p50 ms':>10}{'p99 ms':>10}{'VDD_IN W':>10}{'GPU MHz':>11}{'tj C':>7}")
for path in sys.argv[1:]:
    d = json.load(open(path))
    lat = d["latency"]["chunk"] if d.get("kind") == "smolvla_bringup" else d["latency"]
    # SmolVLA reports carry no label, so take it from the file name.
    label = d.get("label") or Path(path).stem.split("_", 1)[-1]
    power = d.get("power") or {}
    watts = ((power.get("rails") or {}).get("VDD_IN") or {}).get("mean_w")
    gpu = power.get("gpu_freq_mhz") or {}
    tj = (power.get("tj_temp_c") or {}).get("max")
    clock = f"{gpu['min']:.0f}-{gpu['max']:.0f}" if gpu else "-"
    print(f"   {label:<34}{lat['p50_ms']:>10.2f}{lat['p99_ms']:>10.2f}{watts or 0:>10.2f}"
          f"{clock:>11}{tj or 0:>7.1f}")
PYEOF
if [ "$DRY_RUN" -eq 0 ]; then
  say ""
  say "Records are in results/, not committed yet. Log: $LOG"
fi
