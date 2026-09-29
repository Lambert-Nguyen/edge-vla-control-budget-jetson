#!/usr/bin/env bash
# Harness sanity checks that need root, with the board put back afterwards.
#
#   power mode   the stand-in model at 224 px and 1344 px in modes 0 (15W),
#                1 (25W) and 2 (MAXN SUPER), with jetson_clocks on. The numbers
#                should differ by mode, and each record must name its own mode.
#   resolution   112 to 2048 px in mode 2 with jetson_clocks on. With the clocks
#                unable to scale, latency has to rise with resolution.
#
# Run it as your normal user, from a terminal:
#
#   bash benchmarks/sanity_checks.sh
#   bash benchmarks/sanity_checks.sh --dry-run   # no sudo, tiny runs, output deleted
#
# It asks for your sudo password once. Only nvpmodel and jetson_clocks run as
# root. On exit, including Ctrl-C or a failure, it restores the power mode and
# clock state it started from and checks that the GPU clock scales again.
#
# Takes about 5 to 7 minutes. Close what you can and tell the team first, since
# other load shows up in the worst-case latency. The fan may speed up while the
# clocks are pinned. Records land in results/ and are not committed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/lerobot-py310-cuda"
PY="$VENV/bin/python"
GPU="$(ls -d /sys/class/devfreq/*.gpu | head -1)"
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
if ! "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  say "CUDA torch is not available in $VENV."
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  DRY_OUT="$(mktemp -d)"
  say "== DRY RUN: no sudo, no mode changes, 30-iteration runs in $DRY_OUT, deleted at exit"
else
  LOG="$HOME/sanity-checks-$(date +%Y%m%d-%H%M%S).log"
  exec > >(tee -a "$LOG") 2>&1
fi

say "== board before: $(mode_name) (mode $(mode_id)), GPU $(gpu_range), load $(cut -d' ' -f1-3 /proc/loadavg)"
say "== busiest processes right now, which add noise:"
ps -eo %cpu,comm --sort=-%cpu | sed -n '2,6p' | sed 's/^/     /'

# --- save the board state and promise to restore it --------------------------
ORIG_MODE="$(mode_id)"
STATE_DIR="$(mktemp -d)"
STATE="$STATE_DIR/l4t_dfs.conf"
restored=0

restore() {
  [ "$restored" -eq 1 ] && return
  restored=1
  say ""
  say "== restoring the board"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "   (dry run) would switch back to mode $ORIG_MODE and restore the saved clocks"
    rm -rf "$STATE_DIR" "$DRY_OUT"
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
  sudo jetson_clocks --store "$STATE"
fi
trap restore EXIT
trap 'exit 130' INT TERM

set_mode() {  # mode id
  if [ "$DRY_RUN" -eq 1 ]; then
    say "== (dry run) would switch to mode $1 and pin the clocks"
    return
  fi
  sudo -v
  # Answer no if nvpmodel ever asks to reboot. The mode check below catches it.
  echo no | sudo nvpmodel -m "$1" >/dev/null
  if [ "$(mode_id)" != "$1" ]; then
    say "[FAIL] nvpmodel did not switch to mode $1. It may need a reboot for that mode."
    exit 1
  fi
  sudo jetson_clocks >/dev/null
  sleep 3
  if ! gpu_pinned; then
    say "[FAIL] jetson_clocks did not pin the GPU clock ($(gpu_range))."
    exit 1
  fi
  say ""
  say "== $(mode_name) (mode $1), clocks pinned, GPU $(gpu_range)"
}

written=()
run() {  # label, harness args...
  local label="$1" out path
  shift
  if [ "$DRY_RUN" -eq 1 ]; then
    set -- "$@" --iters 30 --warmup 5 --out "$DRY_OUT/$label.json"
  fi
  if ! out="$("$PY" "$REPO/benchmarks/hello_latency.py" --label "$label" "$@" 2>&1)"; then
    printf '%s\n' "$out"
    say "[FAIL] harness run $label failed"
    exit 1
  fi
  printf '%s\n' "$out" | grep -E "^$label |^  p50|^  board power|^  gpu clock|^\[warn\]" | sed 's/^/   /' || true
  path="$(printf '%s\n' "$out" | sed -n 's/^wrote //p')"
  case "$path" in /*) ;; *) path="$REPO/$path" ;; esac
  written+=("$path")
  sleep "$COOL_S"
}

# --- power-mode check, mode 2 last so the resolution check follows it ---------
for m in 0 1 2; do
  set_mode "$m"
  run "pm$m-pinned-224" --resolution 224 --iters 3000
  run "pm$m-pinned-1344" --resolution 1344 --iters 1000
done

# --- resolution check, same points and iteration counts as the unpinned run ---
say ""
say "== resolution check, mode 2, clocks pinned"
for r in 112 224 336 672; do
  run "pm2-pinned-res-$r" --resolution "$r" --iters 2000
done
for r in 1344 2048; do
  run "pm2-pinned-res-$r" --resolution "$r" --iters 1000
done

# --- summary ------------------------------------------------------------------
say ""
say "== summary"
"$PY" - "${written[@]}" <<'PYEOF'
import json, sys
print(f"   {'label':<20}{'mode':>11}{'p50 ms':>9}{'p99 ms':>9}{'VDD_IN W':>10}{'GPU MHz':>11}{'tj C':>7}")
for path in sys.argv[1:]:
    d = json.load(open(path))
    lat, power = d["latency"], d.get("power") or {}
    watts = ((power.get("rails") or {}).get("VDD_IN") or {}).get("mean_w")
    gpu = power.get("gpu_freq_mhz") or {}
    tj = (power.get("tj_temp_c") or {}).get("max")
    clock = f"{gpu['min']:.0f}-{gpu['max']:.0f}" if gpu else "-"
    print(f"   {d['label']:<20}{d['board']['nvpmodel_mode'] or '?':>11}{lat['p50_ms']:>9.2f}"
          f"{lat['p99_ms']:>9.2f}{watts or 0:>10.2f}{clock:>11}{tj or 0:>7.1f}")
PYEOF
if [ "$DRY_RUN" -eq 0 ]; then
  say ""
  say "Records are in results/, not committed yet. Log: $LOG"
fi
