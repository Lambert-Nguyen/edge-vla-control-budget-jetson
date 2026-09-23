# Jetson setup from scratch

> **You almost certainly do not need this document.** Our board,
> `sjsujetson-36`, arrived provisioned by Dr. Liu. Go to
> [Verify an existing setup](02a-verify-existing-setup.md) instead.
>
> This guide is kept for the case where a board has to be rebuilt from nothing.
> Several of its steps (flashing, `set-hostname`, `sjsujetsontool update`)
> destroy work on a board that is already configured.

Goal: a board you can SSH into, that reports a healthy CUDA stack, and that
runs a GPU program you wrote.

Budget about half a day if the board is already flashed, a full day if you have
to flash it.

Reference: Dr. Liu's
[sjsujetsontool guide](https://lkk688.github.io/edgeAI/curriculum/00_sjsujetsontool_guide/)
and [cheatsheet](https://lkk688.github.io/edgeAI/curriculum/00b_sjsujetsontool_cheatsheet/).
Those are authoritative for the tooling. This document is the order to do
things in, with the reasoning a first-timer needs.

## 1. Physical setup

What you need on the desk:

- Jetson Orin Nano Developer Kit, 8 GB
- Its power supply. Either the official barrel-jack adapter or a **5 A**
  USB-C adapter. A 3 A phone charger will boot the board and then brown out
  mid-inference, which looks like a random kernel panic and wastes a day.
- Ethernet cable to the lab network (much less painful than Wi-Fi for SSH)
- HDMI or DisplayPort monitor, USB keyboard and mouse, for the first boot only
- NVMe SSD installed in the M.2 slot

Plug in everything except power. Then power on. The board has no power button
by default, so it boots when it gets power.

Why Ethernet: you are going to SSH into this board hundreds of times. mDNS
hostname resolution (`sjsujetson-36.local`) works far more reliably over wired
Ethernet than over a university Wi-Fi network with client isolation.

## 2. Find out what is already on the board

Do not reflash before you check. The lab image may already be set up.

Log in on the monitor with the username and password Dr. Liu gave you. Then:

```bash
cat /etc/nv_tegra_release        # L4T version
nvidia-smi                        # may not exist on Jetson, that is normal
cat /etc/os-release               # Ubuntu version
python3 --version
ls /usr/local/ | grep cuda        # CUDA version
dpkg -l | grep -i tensorrt | head
df -h                             # is the NVMe mounted and is it the root?
free -h                           # should show ~7.4 Gi total
```

Map what you see to a JetPack release:

| L4T | JetPack | Ubuntu | Python | CUDA |
| --- | --- | --- | --- | --- |
| R36.x | 6.x | 22.04 | 3.10 | 12.6 |
| R39.x | 7.x | 24.04 | 3.12 | 12.8 |

**Which one do you want?** JetPack 6.2 or newer. Two reasons. First, JetPack
6.2 added Super Mode for the Orin Nano, which raises the module power budget to
25 W and gives roughly a 1.7× generative-AI inference improvement over the old
15 W cap. On a project measuring control rate, leaving that on the table would
be silly. Second, the Jetson PyTorch wheel index at
`pypi.jetson-ai-lab.io/jp6/cu126` is mature and is what the professor's own
LeRobot environments are built against.

JetPack 7.2 is fine too and matches the Python 3.12 that the Hy-VLA repo
recommends. It is newer, so expect fewer prebuilt wheels.

**Decision rule:** if the board already runs JetPack 6.2+ or 7.x, leave it
alone and move on. Reflashing costs a day and buys nothing. Record the exact
versions in the team channel so everyone benchmarks against the same image.

## 3. Flashing, only if you have to

If the board is on JetPack 5.x or something unidentifiable, follow the
[JetPack 7 setup guide](https://lkk688.github.io/edgeAI/curriculum/01_jetpack7_setup_guide/).
The short version:

1. Back up everything. The NVMe gets completely erased.
2. Confirm the board is already on JetPack 6.x, because the R36.x UEFI
   firmware is a prerequisite for the JetPack 7 installer. Older firmware needs
   a separate update path first.
3. Download the JetPack 7.2 ISO (~4 GB), flash it to a 16 GB+ USB stick with
   Balena Etcher.
4. Boot the Jetson from USB and install onto the NVMe.

Ask Dr. Liu before reflashing a lab board. Someone else's work may be on it.

## 4. Install sjsujetsontool

This is the course's wrapper around the container, model servers, health
checks, and networking. Install it on the board:

```bash
curl -fsSL https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/install_sjsujetsontool.sh | bash
source ~/.bashrc
```

No `sudo`. Then:

```bash
sjsujetsontool update          # updates both the script and the container image
sjsujetsontool healthcheck     # the one command to run when anything is weird
sjsujetsontool setup-check     # verifies the /Developer folder exists
sjsujetsontool list            # every available subcommand
```

`healthcheck` prints hardware model, kernel, JetPack/L4T version, CUDA, cuDNN,
TensorRT, memory, disk, thermal zones, power rails, Docker status, and which
service ports are live. **Copy its output into the team channel now.** That
block is the board state you will cite in every result for the rest of the
project.

## 5. Give the board a name and get on the network

> **Our board is already named `sjsujetson-36`.** Skip the `set-hostname` step
> below. Renaming it breaks its Tailscale registration and any hostname a
> teammate has already saved. Verify instead:
> ```bash
> hostnamectl
> ```
> See [Verify an existing setup](02a-verify-existing-setup.md) section 9.

On a fresh board, each team gets a hostname. Set it, then reboot:

```bash
sjsujetsontool set-hostname sjsujetson-36
sudo reboot
```

From your laptop:

```bash
ssh sjsujetson@sjsujetson-36.local
ssh -X sjsujetson@sjsujetson-36.local    # -X forwards GUI windows, useful for camera preview
```

Two fallbacks when mDNS fails:

```bash
ssh sjsujetson@192.168.55.1               # direct USB-C ethernet, always works
ssh sjsujetson@192.168.100.X              # the course Headscale mesh VPN
```

The USB-C route is the one that saves you. Connect a USB-C cable from your
laptop to the Jetson and the board presents itself at `192.168.55.1` regardless
of what the lab network is doing.

Get the board's actual mesh address from `tailscale status` rather than
guessing it. The course guide's example maps `sjsujetson-04` to
`192.168.100.14`, so the pattern appears to be the board number plus ten, but
confirm rather than assume.

To join the course mesh VPN so you can reach the board from off campus:

```bash
sjsujetsontool tailscale install
sjsujetsontool tailscale up
sjsujetsontool tailscale status
```

If you changed the hostname after joining, re-register:

```bash
sjsujetsontool tailscale up --force
```

Set up SSH keys now so you are not typing a password a hundred times a day.
From your laptop:

```bash
ssh-copy-id sjsujetson@sjsujetson-36.local
```

## 6. Storage and memory configuration

The board has 8 GB of LPDDR5 shared by CPU, GPU, and display. Our model wants
8.7 GB in FP16. Managing that boundary is the project, so set up the board
honestly and document what you did.

Check the NVMe is the root filesystem:

```bash
df -h /
lsblk
```

If the SSD reports less space than it physically has (a cloned image artifact):

```bash
sudo apt install cloud-guest-utils
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h
```

Check swap. JetPack enables zram by default, which compresses swap in RAM:

```bash
free -h
swapon --show
zramctl
```

A note on swap and this project specifically. Adding NVMe swap will keep an
oversized model from being killed by the OOM killer, and it will also make your
latency numbers meaningless, because page faults to disk are thousands of times
slower than memory. Policy for this repo:

- Keep zram as shipped.
- Do not add NVMe swap for benchmark runs.
- If you add swap to get something to load during exploration, say so in the
  result record. A latency measurement taken while swapping is not a result.

Free memory before a run by stopping the desktop:

```bash
sudo systemctl isolate multi-user.target   # drops to console, frees ~500 MB
# to get the desktop back:
sudo systemctl start graphical.target
```

## 7. Power modes and clocks

This is the single most common reason two teammates get different numbers from
the same code.

```bash
sudo nvpmodel -q                  # what mode am I in?
sudo nvpmodel -q --verbose        # list all available modes
```

On the Orin Nano 8 GB you will typically have a 15 W mode, a 25 W mode, and
MAXN SUPER (uncapped, throttles on TDP). On `sjsujetson-36` they are:

| Mode | Name | What it is |
| --- | --- | --- |
| 0 | 15W | The original pre-Super Orin Nano envelope. Our tightest power budget |
| 1 | 25W | The module TDP, capped. Deterministic |
| 2 | MAXN SUPER | Uncapped. Highest clocks, throttles back when it exceeds TDP |

Set the mode explicitly:

```bash
sudo nvpmodel -m 2                # confirm the numbers with -q --verbose
sudo jetson_clocks                # pin clocks to max instead of letting them scale
sudo jetson_clocks --show
```

`jetson_clocks` disables dynamic frequency scaling. You want it on for
benchmark runs because it removes governor behavior from your variance. You
want it off for realistic deployment numbers, because a real robot does not pin
its clocks forever. Measure both and label them.

**Rule for this repo: every result record includes the nvpmodel mode and the
jetson_clocks state.** `benchmarks/hello_latency.py` captures these
automatically. Do not hand-roll a harness that skips them.

Watch power and thermals live while a run is going:

```bash
sudo pip3 install -U jetson-stats   # once
sudo systemctl restart jtop.service
jtop                                 # interactive dashboard
tegrastats --interval 100            # raw text, easier to parse in a script
```

`jtop` reads the board's INA3221 rails and gives you per-rail current and
power. That is where the power column of our results table comes from.

## 8. Hello world: prove the GPU stack works

Copy the repo onto the board and run the check script:

```bash
# on the Jetson
git clone https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson.git ~/edge-vla
cd ~/edge-vla
python3 hardware/hello_jetson.py
```

It prints the board identity, the CUDA/TensorRT versions, memory, power mode,
and then runs a real GPU matrix multiply and a small convolution, timing both.

What you should see:

```text
[ok]   torch 2.x.x, CUDA available: True
[ok]   device: Orin, compute capability 8.7
[ok]   matmul 4096x4096 fp16: ~x.x ms  (~xx TFLOP/s)
```

If `CUDA available: False`, you are in the wrong Python environment. That is
normal and expected on a fresh board. Go to
[Python environments](03-python-environments.md) and come back.

You can also run it inside the course container, which already has a working
CUDA Python:

```bash
sjsujetsontool shell
python3 -c "import torch; print(torch.cuda.is_available())"
exit                               # leaves the shell, container keeps running
sjsujetsontool stop                # actually stops it
```

## 9. Optional but useful: confirm the course LLM stack runs

Not required for our project, and a good five-minute sanity check that the GPU
is really doing work:

```bash
sjsujetsontool llama qwen4b bg     # serves a model in the background on :8080
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi in five words"}],
       "max_tokens":32,
       "chat_template_kwargs":{"enable_thinking":false}}'
sjsujetsontool llama stop
```

If tokens come out at a sane rate, the GPU, the memory, and the power supply
are all fine. If the board reboots halfway through, your power adapter is not
5 A.

## 10. Before you walk away

```bash
sjsujetsontool stop      # stop containers before cutting power
```

Pulling power from a running container is how filesystems get corrupted. Also
never unplug while an `apt` or `update` is mid-flight.

Use `sjsujetsontool sysupgrade` rather than a bare `apt upgrade`. The wrapper
holds back the NVIDIA and CUDA packages. A plain `apt upgrade` can pull a
mismatched CUDA and leave you with a board that no longer runs your engines.

## Done when

- [ ] `sjsujetsontool healthcheck` is clean and its output is posted in the team channel
- [ ] You can SSH in from your laptop without a password
- [ ] `nvpmodel -q` and `jetson_clocks --show` output is recorded
- [ ] `hardware/hello_jetson.py` reports CUDA available and a plausible matmul time
- [ ] `jtop` shows power rails
