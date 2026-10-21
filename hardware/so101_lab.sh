#!/usr/bin/env bash
# One SO-101 lab session on the Jetson: check the hardware, calibrate,
# teleoperate, record demos, and let a SmolVLA policy drive the follower.
#
#   bash hardware/so101_lab.sh check                    # env, model cache, serial ports, cameras
#   bash hardware/so101_lab.sh ports                    # which /dev/ttyACM* is which arm
#   bash hardware/so101_lab.sh motors follower          # or leader: do all 6 servos answer? read-only
#   bash hardware/so101_lab.sh calibrate follower       # or leader
#   bash hardware/so101_lab.sh teleop                   # leader drives follower, no cameras
#   bash hardware/so101_lab.sh preview [DEVICE]         # live view of one camera, to aim it (default CAM1)
#   bash hardware/so101_lab.sh record NAME [EPISODES]   # teleop demos with cameras, saved on the board
#   bash hardware/so101_lab.sh policy NAME [POLICY]     # SmolVLA drives the follower for one episode
#
# Settings are environment variables, so the script itself never needs editing:
#
#   FOLLOWER_PORT  follower serial port (default /dev/ttyACM0)
#   LEADER_PORT    leader serial port (default /dev/ttyACM1)
#   FOLLOWER_ID    calibration name of the follower (default lab_follower)
#   LEADER_ID      calibration name of the leader (default lab_leader)
#   CAM1, CAM2     camera devices, /dev/v4l/by-id/... paths. CAM2 is optional
#   CAM_W, CAM_H, CAM_FPS   camera mode (default 640x480 at 30)
#   TASK           the instruction, the same for every demo and for the policy
#   EPISODE_S      seconds per episode (default 20), RESET_S between them (default 10)
#   MAX_STEP       largest joint move per step when a policy drives, in degrees (default 3)
#   RESUME=1       add episodes to an existing dataset instead of refusing the NAME
#   STEPS          denoising steps for the policy (default: the checkpoint's, 10). On this board
#                  each step costs about 90 ms, and the arm pauses while a chunk is computed,
#                  so STEPS=5 roughly halves the pause (1.1 s to 0.7 s with two cameras)
#
# Example:
#   CAM1=/dev/v4l/by-id/usb-XYZ-video-index0 TASK="Put the red cube in the box." \
#     bash hardware/so101_lab.sh record cube_box 50
#
# Cameras are named camera1 and camera2 because that is what lerobot/smolvla_base
# expects, so demos, fine-tuning and the policy all use the same keys. Datasets
# stay on the board under ~/.cache/huggingface/lerobot/local/ and are never
# uploaded. The policy runs offline from the local model cache.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/lerobot-py310-cuda"
BIN="$VENV/bin"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$VENV/cudss-lib:${LD_LIBRARY_PATH:-}"
# Everything needed is cached, so never wait on a lab network. HF_HUB_OFFLINE=0 to override.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-lab_follower}"
LEADER_ID="${LEADER_ID:-lab_leader}"
CAM_W="${CAM_W:-640}"
CAM_H="${CAM_H:-480}"
CAM_FPS="${CAM_FPS:-30}"
TASK="${TASK:-Pick up the cube and put it in the box.}"
EPISODE_S="${EPISODE_S:-20}"
RESET_S="${RESET_S:-10}"
MAX_STEP="${MAX_STEP:-3}"
STEPS="${STEPS:-}"
DATA="$HOME/.cache/huggingface/lerobot/local"

say() { printf '%s\n' "$*"; }
die() { say "$*"; exit 1; }
quiet() { grep -v -i -E "scipy|warnings.warn" || true; }  # drop a harmless numpy/scipy warning

follower=(--robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID")
leader=(--teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID")

cameras() {  # the --robot.cameras argument, from CAM1 and CAM2
  [ -n "${CAM1:-}" ] || die "Set CAM1 to a camera, e.g. CAM1=/dev/v4l/by-id/...-video-index0. See: $0 check"
  local spec="camera1: {type: opencv, index_or_path: $CAM1, width: $CAM_W, height: $CAM_H, fps: $CAM_FPS}"
  if [ -n "${CAM2:-}" ]; then
    spec="$spec, camera2: {type: opencv, index_or_path: $CAM2, width: $CAM_W, height: $CAM_H, fps: $CAM_FPS}"
  fi
  printf '%s' "--robot.cameras={$spec}"
}

can_open() {  # serial port exists and this user may open it
  [ -e "$1" ] || { say "   $1: not present. Is the arm's USB plugged in? Try: $0 ports"; return 1; }
  [ -r "$1" ] && [ -w "$1" ] && return 0
  say "   $1: permission denied. Needs one of (sudo):"
  say "      sudo usermod -aG dialout $USER    # permanent, then log out and back in"
  say "      sudo chmod 666 $1             # until the arm is unplugged"
  return 1
}

fresh() {  # refuse to overwrite an earlier dataset
  [ ! -e "$DATA/$1" ] || die "$DATA/$1 already exists. Pick another NAME, or RESUME=1 to add to it."
}

limp_warning() {
  say "When this stops (time up or Ctrl-C), the follower's motors switch off and the arm drops."
  say "Hold the follower, or rest it on the table, before it ends."
}

case "${1:-}" in
  check)
    say "== python env"
    "$BIN/python" - <<'PYEOF' 2>&1 | quiet
import torch, lerobot, transformers
from huggingface_hub import snapshot_download
print(f"   torch {torch.__version__}, cuda {torch.cuda.is_available()}, lerobot {lerobot.__version__}, transformers {transformers.__version__}")
for repo in ["lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]:
    try:
        snapshot_download(repo, local_files_only=True, ignore_patterns=["onnx/*", "*.gif", "*.ipynb"])
        print(f"   cached: {repo}")
    except Exception as e:
        print(f"   NOT CACHED: {repo} ({type(e).__name__}). Run hardware/hello_smolvla.py once with a network.")
PYEOF
    say "== serial ports"
    ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sed 's/^/   /' || say "   none. Plug in the arm's USB cable."
    if [ -d /dev/serial/by-id ]; then
      for link in /dev/serial/by-id/*; do say "   $link -> $(readlink -f "$link")"; done
    fi
    id -nG | grep -qw dialout && say "   $USER is in dialout" || say "   $USER is NOT in dialout"
    for p in "$FOLLOWER_PORT" "$LEADER_PORT"; do can_open "$p" && say "   $p: OK" || true; done
    say "== cameras (use these paths for CAM1 and CAM2)"
    found_cam=0
    for cam in /dev/v4l/by-id/*-video-index0; do
      [ -e "$cam" ] || continue
      found_cam=1
      node="$(basename "$(readlink -f "$cam")")"
      speed="$(cat "/sys/class/video4linux/$node/device/../speed" 2>/dev/null || echo '?')"
      case "$speed" in
        5000|10000|20000) note="USB 3, good" ;;
        480) note="USB 2: use one of the board's own ports, or the camera or cable is USB 2" ;;
        *) note="unknown speed" ;;
      esac
      say "   $cam"
      say "      ${speed}M, $note"
    done
    [ "$found_cam" -eq 1 ] || say "   none"
    say "   Use project cameras only. A monitor with a built-in webcam shows up here too."
    say "== board"
    say "   $(nvpmodel -q 2>/dev/null | head -1), $(awk '/MemAvailable/ {printf "%.1f GiB RAM free", $2/1048576}' /proc/meminfo)"
    ;;
  ports)
    "$BIN/lerobot-find-port"
    ;;
  motors)
    case "${2:-}" in
      follower) port="$FOLLOWER_PORT" ;;
      leader) port="$LEADER_PORT" ;;
      *) die "usage: $0 motors follower|leader" ;;
    esac
    can_open "$port" || exit 1
    # Read-only: pings the bus and reads a few registers. Writes nothing, moves nothing,
    # and needs no calibration.
    "$BIN/python" - "$port" <<'PYEOF' 2>&1 | quiet
import sys
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

port = sys.argv[1]
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
bus = FeetechMotorsBus(port, {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
                              for i, n in enumerate(names, start=1)})
bus.connect(handshake=False)
found = bus.broadcast_ping(num_retry=2) or {}
print(f"   {port}")
if not found:
    print("   No servo answers. Check, in order:")
    print("   - the arm's power supply is plugged into the wall and into the driver board. USB alone does not power the servos")
    print("   - the 3-wire cable from the driver board to the first servo is seated")
    print("   - on a Waveshare driver board, both jumpers sit on B (USB)")
    bus.disconnect(disable_torque=False)
    sys.exit(1)
volts = []
for i, name in enumerate(names, start=1):
    if i not in found:
        print(f"   ID {i} {name:<14} MISSING")
        continue
    try:
        pos = bus.read("Present_Position", name, normalize=False, num_retry=2)
        volt = bus.read("Present_Voltage", name, normalize=False, num_retry=2) / 10
        temp = bus.read("Present_Temperature", name, normalize=False, num_retry=2)
        torque = bus.read("Torque_Enable", name, normalize=False, num_retry=2)
    except Exception as e:
        print(f"   ID {i} {name:<14} answers, but a read failed: {e}")
        continue
    volts.append(volt)
    kind = "STS3215" if found[i] == 777 else f"model {found[i]}, NOT an STS3215"
    print(f"   ID {i} {name:<14} {kind}   position {pos:4d} of 4095   {volt:4.1f} V   {temp:2d} C   "
          f"torque {'on' if torque else 'off'}")
extra = sorted(set(found) - set(range(1, 7)))
if extra:
    print(f"   unexpected IDs on the bus: {extra}")
n = sum(1 for i in range(1, 7) if i in found)
print(f"   {n} of 6 servos answer")
if n < 6:
    print("   The servos are chained one to the next. Check the cable going into the first MISSING one.")
if volts:
    v = max(volts)
    if v < 4.5:
        msg = "too low. The supply is off, too weak, or not plugged into the driver board"
    elif v < 9:
        msg = "a 5 V class supply. Right for 7.4 V servos, wrong for 12 V servos"
    else:
        msg = "a 12 V supply. Right only if the servo labels say 12 V"
    print(f"   supply about {v:.1f} V: {msg}")
bus.disconnect(disable_torque=False)
sys.exit(0 if n == 6 else 1)
PYEOF
    ;;
  calibrate)
    case "${2:-}" in
      follower) can_open "$FOLLOWER_PORT" || exit 1; "$BIN/lerobot-calibrate" "${follower[@]}" ;;
      leader) can_open "$LEADER_PORT" || exit 1
              "$BIN/lerobot-calibrate" --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" ;;
      *) die "usage: $0 calibrate follower|leader" ;;
    esac
    ;;
  teleop)
    can_open "$FOLLOWER_PORT" && can_open "$LEADER_PORT" || exit 1
    limp_warning
    "$BIN/lerobot-teleoperate" "${follower[@]}" "${leader[@]}"
    ;;
  preview)
    dev="${2:-${CAM1:-}}"
    [ -n "$dev" ] || die "usage: $0 preview DEVICE, or set CAM1. Devices: $0 check"
    say "Live view of $dev. Close the window or press Ctrl-C to stop."
    gst-launch-1.0 -q v4l2src device="$dev" ! decodebin ! videoconvert ! autovideosink sync=false
    ;;
  record)
    name="${2:?usage: $0 record NAME [EPISODES]}"
    resume=()
    if [ "${RESUME:-0}" = 1 ]; then resume=(--resume=true); else fresh "$name"; fi
    can_open "$FOLLOWER_PORT" && can_open "$LEADER_PORT" || exit 1
    say "Recording ${3:-10} episodes of ${EPISODE_S}s: \"$TASK\""
    say "Right arrow ends an episode early, left arrow redoes it, Esc stops. Saving takes about half a minute."
    limp_warning
    "$BIN/lerobot-record" "${follower[@]}" "$(cameras)" "${leader[@]}" \
      --dataset.repo_id="local/$name" --dataset.single_task="$TASK" \
      --dataset.num_episodes="${3:-10}" --dataset.episode_time_s="$EPISODE_S" \
      --dataset.reset_time_s="$RESET_S" --dataset.push_to_hub=false --play_sounds=false "${resume[@]}"
    ;;
  policy)
    name="eval_${2:?usage: $0 policy NAME [POLICY]}"
    policy="${3:-lerobot/smolvla_base}"
    fresh "$name"
    can_open "$FOLLOWER_PORT" || exit 1
    say "Policy $policy drives the follower for ${EPISODE_S}s: \"$TASK\""
    say "Joints are capped at ${MAX_STEP} degrees per step. Loading takes about 15 s."
    say "Clear the table around the arm. To stop it at once, unplug the follower's power supply."
    limp_warning
    read -r -p "Press Enter to start, Ctrl-C to cancel: "
    "$BIN/lerobot-record" "${follower[@]}" "$(cameras)" \
      --robot.max_relative_target="$MAX_STEP" \
      --policy.path="$policy" --policy.device=cuda ${STEPS:+--policy.num_steps="$STEPS"} \
      --dataset.repo_id="local/$name" --dataset.single_task="$TASK" \
      --dataset.num_episodes=1 --dataset.episode_time_s="$EPISODE_S" --dataset.reset_time_s=5 \
      --dataset.push_to_hub=false --play_sounds=false
    ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
