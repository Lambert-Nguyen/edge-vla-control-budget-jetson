#!/usr/bin/env bash
# Read-only audit of a pre-provisioned Jetson.
#
# Every command here only reads. Nothing is installed, started, stopped,
# configured, or written. Run it before following any setup guide so you know
# what is already on the board and do not overwrite someone else's work.
#
#   bash hardware/audit_board.sh
#   bash hardware/audit_board.sh > ~/board-audit-$(date +%F).txt
#
# `sudo -n` is used for the two power-state reads so the script never hangs
# waiting for a password. If they print "needs sudo", run them by hand.

set -u

hr()  { printf '\n%s\n' "------------------------------------------------------------"; }
sec() { hr; printf '## %s\n\n' "$1"; }
have(){ command -v "$1" >/dev/null 2>&1; }

# Print captured output indented, or an explicit "none" when it is empty.
# A bare `ls ... | sed ... || echo none` never fires the fallback, because the
# pipeline exits with sed's status, not ls's.
show() {
  local out="$1"
  if [ -n "$out" ]; then printf '%s\n' "$out" | sed 's/^/    /'
  else printf '    none\n'
  fi
}

# Print a command's output, or a clear marker when the tool is absent.
try() {
  if have "$1"; then "$@" 2>&1 | sed 's/^/  /'
  else printf '  [not installed] %s\n' "$1"
  fi
}

printf '============================================================\n'
printf ' Jetson board audit  %s\n' "$(date)"
printf ' host: %s   user: %s\n' "$(hostname)" "$(whoami)"
printf '============================================================\n'

sec "1. Identity and OS"
if [ -f /proc/device-tree/model ]; then
  printf '  model:   %s\n' "$(tr -d '\0' < /proc/device-tree/model)"
fi
[ -f /etc/nv_tegra_release ] && printf '  L4T:     %s\n' "$(head -1 /etc/nv_tegra_release)"
have apt-cache && printf '  JetPack: %s\n' "$(apt-cache policy nvidia-jetpack 2>/dev/null | awk '/Installed/{print $2}')"
[ -f /etc/os-release ] && printf '  ubuntu:  %s\n' "$(. /etc/os-release && echo "$PRETTY_NAME")"
printf '  kernel:  %s\n' "$(uname -r)"
printf '  arch:    %s\n' "$(uname -m)"
printf '  uptime:  %s\n' "$(uptime -p 2>/dev/null)"

sec "2. CUDA, cuDNN, TensorRT"
if [ -e /usr/local/cuda ]; then
  printf '  cuda symlink -> %s\n' "$(readlink -f /usr/local/cuda)"
fi
try nvcc --version
printf '  cudnn:    %s\n'    "$(dpkg -l 2>/dev/null | awk '/libcudnn[0-9]/{print $3; exit}')"
printf '  tensorrt: %s\n'    "$(dpkg -l 2>/dev/null | awk '/ libnvinfer[0-9]/{print $3; exit}')"

sec "3. Memory, swap, storage"
if have free; then free -h | sed 's/^/  /'; else printf '  [no free(1), not Linux]\n'; fi
printf '\n'
show "$(swapon --show 2>/dev/null)"
printf '\n'
df -h / /home 2>/dev/null | sed 's/^/  /'
printf '\n'
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null | sed 's/^/  /'

sec "4. Power mode and clocks (read only)"
if sudo -n true 2>/dev/null; then
  sudo -n nvpmodel -q 2>&1 | sed 's/^/  /'
  printf '\n'
  sudo -n jetson_clocks --show 2>&1 | grep -E 'GPU|EMC|Fan' | sed 's/^/  /'
else
  printf '  [needs sudo] run by hand:\n'
  printf '    sudo nvpmodel -q\n'
  printf '    sudo jetson_clocks --show\n'
fi

sec "5. sjsujetsontool"
if have sjsujetsontool; then
  printf '  path: %s\n' "$(command -v sjsujetsontool)"
  try sjsujetsontool version
else
  printf '  [not installed]\n'
fi
[ -d /Developer ] && printf '\n  /Developer exists:\n' && ls -la /Developer 2>/dev/null | head -20 | sed 's/^/    /'

sec "6. Docker"
try docker --version
if have docker; then
  printf '\n  running containers:\n'
  docker ps 2>&1 | sed 's/^/    /'
  printf '\n  images:\n'
  docker images 2>&1 | head -15 | sed 's/^/    /'
fi

sec "7. Python environments"
printf '  system python3: %s  (%s)\n' "$(python3 --version 2>&1)" "$(command -v python3)"
have uv && printf '  uv: %s\n' "$(uv --version 2>&1)"
[ -x "$HOME/.local/bin/uv" ] && printf '  uv: %s\n' "$("$HOME/.local/bin/uv" --version 2>&1)"

printf '\n  virtual environments in $HOME:\n'
found_venv=0
for d in "$HOME"/*/; do
  if [ -x "${d}bin/python" ]; then
    found_venv=1
    printf '    %s\n' "$d"
    printf '      python:  %s\n' "$("${d}bin/python" --version 2>&1)"
    for pkg in torch lerobot transformers numpy tensorrt; do
      v=$("${d}bin/python" -c "import $pkg,sys; sys.stdout.write(getattr($pkg,'__version__','?'))" 2>/dev/null)
      [ -n "$v" ] && printf '      %-12s %s\n' "$pkg" "$v"
    done
    cu=$("${d}bin/python" -c "import torch,sys; sys.stdout.write(str(torch.cuda.is_available()))" 2>/dev/null)
    [ -n "$cu" ] && printf '      %-12s %s\n' "cuda avail" "$cu"
  fi
done
[ "$found_venv" -eq 0 ] && printf '    none found\n'

sec "8. Project files already on the board"
for f in "$HOME/so101_unified_teleop.py" "$HOME/.cache/huggingface" "$HOME/.lerobot" \
         "$HOME/.cache/lerobot" "$HOME/lerobot" "$HOME/edgeAI"; do
  [ -e "$f" ] && printf '  present: %s\n' "$f"
done
printf '\n  calibration files:\n'
show "$(find "$HOME" -maxdepth 5 -path '*calibration*' -name '*.json' 2>/dev/null | head -10)"

sec "9. Network and hostname"
run_name=$(hostname)
file_name=$(cat /etc/hostname 2>/dev/null)
printf '  hostname now:      %s\n' "$run_name"
printf '  /etc/hostname:     %s\n' "${file_name:-<missing>}"

# Factory defaults mean nobody ever named this board.
case "$run_name" in
  ubuntu|nvidia-desktop|tegra-ubuntu|jetson|nvidia|localhost)
    printf '  verdict:           DEFAULT name, never configured\n' ;;
  *)
    printf '  verdict:           named, leave it alone\n' ;;
esac

if [ -n "$file_name" ] && [ "$file_name" != "$run_name" ]; then
  printf '  WARNING:           /etc/hostname differs, a rename is pending a reboot\n'
fi

if grep -qw "$run_name" /etc/hosts 2>/dev/null; then
  printf '  /etc/hosts entry:  present\n'
else
  printf '  /etc/hosts entry:  MISSING, expect slow sudo and flaky mDNS\n'
fi

# A cloned SSD leaves two boards sharing an id, which breaks DHCP and Tailscale.
printf '  machine-id:        %s\n' "$(cat /etc/machine-id 2>/dev/null || echo '<missing>')"

printf '\n  addresses:\n'
show "$(ip -4 addr show 2>/dev/null | awk '/inet /{print $NF ": " $2}')"

if have tailscale; then
  printf '\n  tailscale:\n'
  tailscale status 2>&1 | head -5 | sed 's/^/    /'
else
  printf '\n  tailscale: [not installed]\n'
fi

printf '\n  confirm resolution FROM YOUR LAPTOP, not here:\n'
printf '    ping -c1 %s.local\n' "${run_name%.local}"

sec "10. Attached hardware (arm and cameras)"
printf '  serial devices:\n'
show "$(ls -l /dev/ttyACM* /dev/ttyUSB* /dev/so101_* 2>/dev/null)"
printf '\n  video devices:\n'
show "$(ls /dev/video* 2>/dev/null)"
have v4l2-ctl && printf '\n' && v4l2-ctl --list-devices 2>&1 | sed 's/^/    /'
printf '\n  udev rules for the arm:\n'
show "$(ls /etc/udev/rules.d/ 2>/dev/null | grep -i -E 'so101|feetech|arm')"

sec "11. Monitoring tools"
for t in jtop tegrastats nvidia-smi; do
  if have "$t"; then printf '  %-12s %s\n' "$t" "$(command -v $t)"
  else printf '  %-12s [not installed]\n' "$t"; fi
done

hr
printf '\nAudit complete. Nothing was modified.\n'
printf 'Interpret the results with docs/guides/02a-verify-existing-setup.md\n\n'
