# Project update: Jetson bring-up and arm decision

**Project:** Real-Time Control Budgets for VLA Manipulation Policies on the Jetson Orin Nano (CMPE 295A)
**Team:** Lam Nguyen, Vi Thi Tuong Nguyen, James Pham
**Advisor:** Dr. Kaikai Liu
**Meeting:** 2026-09-29. Work below done on 2026-09-28 on `sjsujetson-36`.

## Summary

- **The Jetson software stack is up and verified.** Both LeRobot environments
  from the course recipe are built: a CUDA env (torch 2.8.0, LeRobot 0.4.4) for
  inference and benchmarks, and a CPU env (LeRobot 0.5.1) for the arm. The GPU
  check reports 14 checks, 0 failed, and 2 warnings, both understood and
  documented.
- **The measurement harness now records board power**, alongside latency,
  memory, power mode, clock state and temperature, in every result. Two of the
  three sanity checks on the harness are done. The third needs sudo.
- **Our first measurement produced a methodology finding.** With clocks
  unpinned, the GPU governor hides compute cost as power. Latency stayed flat
  across a 36× increase in pixels while board power rose from 6.5 W to 14.9 W.
- **The camera path is characterized** with a USB webcam. Lighting and
  preprocessing both eat into a 30 Hz budget.
- **The arm software is staged and pre-tested.** This caught a bug that would
  have stopped the arm's first test script on day one.
- **The arm is not ordered.** The Hiwonder kit we planned to buy turns out to use
  Hiwonder's own servos, not the Feetech servos LeRobot is built around. **We
  need your call on this before ordering** (section 6).

## 1. Status

| Workstream | Status | Notes |
| --- | --- | --- |
| Arm purchase | **Open** | Not ordered. Hiwonder question in section 6 |
| Board audit | Done | Re-audited 2026-09-28, no system-level changes |
| Python environments | **Done** | Both envs built and verified, versions pinned |
| Arm software | Staged | Tools installed and tested without hardware. Hardware steps wait for the arm |
| Benchmark harness | Mostly done | Power sampling added. Warmup check passed, resolution check confounded by clock scaling, power-mode check needs sudo |
| Model and simulation | Not started | Workstation work, needs a ≥ 24 GB GPU |

## 2. Board state (`sjsujetson-36`)

| Component | Version / state |
| --- | --- |
| Module | Orin Nano 8 GB, Engineering Reference Developer Kit Super |
| JetPack / L4T | 6.2 / R36.4.3, Ubuntu 22.04.5, kernel 5.15.148-tegra |
| CUDA / cuDNN / TensorRT | 12.6.68 / 9.3.0.75 / 10.3.0.30 (runtime only, see below) |
| Memory / storage | 7.4 GiB unified, zram swap / 465.8 GB NVMe, 322 GB free |
| Power mode | MAXN_SUPER (mode 2), `jetson_clocks` off, as shipped |
| Monitoring | jtop 4.3.2 installed. INA3221 rails readable without root |

Two things differed from what we expected on a provisioned board:

1. **The LeRobot environments were not preinstalled.** We built both following
   your recipe for `cmpe-jetson`.
2. **TensorRT's Python bindings are missing.** The runtime is installed, but
   `python3-libnvinfer` and `trtexec` (`libnvinfer-bin`) are not, so
   `import tensorrt` fails. Both are in the apt cache at the matching 10.3.0.30.
   A dry run shows 2 new packages, 0 upgrades and 0 removals, with no
   `apt update` needed. We have not run it, because it needs sudo (section 7).

## 3. Python environments

| | `~/lerobot-py310-cuda` (alias `vla`) | `~/lerobot-py312` (alias `arm`) |
| --- | --- | --- |
| Python | 3.10.12 (system), `--system-site-packages` | 3.12.14 (uv-managed) |
| torch | 2.8.0, CUDA 12.6, Jetson wheel from `pypi.jetson-ai-lab.io/jp6/cu126` | 2.10.0+cpu (by design) |
| LeRobot | 0.4.4 `[feetech]` | 0.5.1 `[feetech]` |
| Other pins | numpy 1.26.4, opencv-python-headless 4.11.0.86, pygame 2.6.1 | numpy 2.2.6 |
| Verified by | `hardware/hello_jetson.py`: 14 checks, 0 failed, 2 warnings. FP16 matmul 4096³ in 27.5 ms (5.0 TFLOP/s, clocks unpinned) | `lerobot-info`, `pip check` clean, Feetech SDK imports |

Where this board differed from your recipe:

- **`~/.local` leaks into the CUDA env.** Someone's `pip install --user`
  (numpy 2.2.6, tensorflow, protobuf 7) sits on `sys.path` ahead of the system
  packages. The `vla` alias now sets `PYTHONNOUSERSITE=1`.
- **CUDA `torch.linalg` fails** (`undefined symbol:
  cusolverDnXsyevBatched_bufferSize`). The torch 2.8.0 wheel expects a newer
  cuSOLVER than JetPack 6.2's 11.6.4. Matmul, convolution and attention are
  unaffected, so decompositions go on the CPU. `hello_jetson.py` now warns
  about it.
- **The current torch 2.8.0 wheel no longer links cuDSS.** We kept the cuDSS
  step to match your recipe. It is isolated and harmless.
- **Guard against the classic CPU-torch accident.** A `pip.conf` inside the
  CUDA venv applies a constraints file pinning torch, torchvision and
  torchaudio, and numpy<2. A `pip install` that would silently replace the
  Jetson wheel now fails loudly instead.

Exact package sets are committed in `hardware/envs/`, and `requirements.txt`
is pinned, including the torch wheel's sha256.

## 4. Measurement harness and first measurements

`benchmarks/hello_latency.py` now samples the INA3221 rails, GPU clock and
junction temperature in a thread covering exactly the timed loop. It reads
sysfs directly, with no root and no jtop, and still runs on the workstation.
Every record carries p50/p90/p99 latency, peak memory, mean and peak power per
rail, energy per inference, power mode, clock state, GPU clock range, peak
temperature, and the git commit. The model is still the stand-in 4-layer
convnet, so these numbers validate the harness and are **not a VLA baseline**.
All 13 records are committed in `results/` against a clean commit (`3729b4f`).

Conditions: MAXN_SUPER, clocks unpinned (no sudo), and the desktop session
running at a load average of about 2.1 on 6 cores.

| Check | Result | Verdict |
| --- | --- | --- |
| Warmup matters | p99 397.5 ms with no warmup, against 5.7 ms with 20 warmup iterations | Passes |
| Sampler overhead | p99 4.90 ms with power sampling, 4.98 ms without, over 3000 iterations | No measurable perturbation |
| Latency rises with resolution | Flat from 112 to 672 px (table below) | Confounded by DVFS |
| Power mode changes the answer | Not run | Needs sudo |

**The finding from the resolution check.** With `jetson_clocks` off, the GPU governor spends
the extra work as clock and power, so latency does not move:

| Resolution | p50 (ms) | p99 (ms) | Board power, VDD_IN (W) | GPU clock (MHz) | Energy / inference (mJ) |
| --- | --- | --- | --- | --- | --- |
| 112 | 2.57 | 6.45 | 6.49 | 306 – 408 | 23 |
| 224 | 2.52 | 5.62 | 7.13 | 408 – 714 | 23 |
| 336 | 2.06 | 5.35 | 8.87 | 408 – 918 | 24 |
| 672 | 2.53 | 5.94 | 14.87 | 1020 | 54 |
| 1344 | 13.82 | 16.29 | 16.23 | 1020 | 212 |
| 2048 | 30.74 | 32.34 | 16.36 | 1020 | 491 |

Only after the clock saturates at 1020 MHz does latency follow the pixel
count: 1344 → 2048 px gives a 0.45 latency ratio against 0.43 of the pixels,
which also confirms the harness times execution rather than queueing. Up to
672 px the stand-in model is bound by a fixed ~2 ms of launch and sync
overhead, and its p50-to-p99 spread (≈ 2.5 vs 6 ms) persists even with the GPU
at 1020 MHz. That points to the CPU side, meaning the CPU governor and the
desktop load. Pinned clocks and a quiet board should show how much of it is
real.

What this means for the project:

- A latency number without its power and clock columns can be misleading on
  this board. The harness now always records them.
- The main sweep should run with clocks pinned. The choice between mode 1
  (25 W, capped and more deterministic) and mode 2 (MAXN SUPER, fastest but
  able to throttle) is still open.
- Energy per inference, not just latency, belongs in the trade-off curves.

## 5. Camera path (Logitech BRIO on the board)

`hardware/hello_camera.py` (new) measures the achieved frame rate, capture CPU
and preprocessing time to a 224×224 GPU tensor. No frames are saved. The room
was dark, and mean brightness was 3–4 out of 255.

| Mode | Achieved fps (asked 30) | CPU, % of one core | Preprocess p50 (ms) |
| --- | --- | --- | --- |
| MJPG 640×480 | 16.2 | 26 | 8.4 (area) / 5.7 (linear) |
| YUYV 640×480 | 16.3 | 29 | 7.3 |
| MJPG 1280×720 | 15.1 | 42 | 15.7 |
| MJPG 1920×1080 | 15.3 | 64 | 24.2 |

- **Lighting is part of the control budget.** In low light, auto-exposure
  stretched each exposure to 31–62 ms and capped the camera at about 15 fps,
  half a 30 Hz loop. A 15.6 ms manual exposure brought it back to 27.7 fps.
  The camera's auto settings were restored afterwards.
- **Preprocessing is not free.** An `INTER_AREA` resize from 1080p alone takes
  about 12 ms in a tight loop. In the live loop, total preprocessing reaches
  24 ms, probably because the CPU clock drops while it waits for frames.
  Capture near the model's resolution. The resize filter has to match the
  policy's training pipeline, so it is an option, not a default change.
- The BRIO sits on the USB 2.0 bus. MJPEG works up to 1080p30, but YUYV stops
  at 640×480@30, which matters once two cameras and two arm adapters share the
  hubs.
- Cameras need no custom udev rules. The serial-keyed
  `/dev/v4l/by-id/...-video-index0` path is stable, and LeRobot accepts it
  together with `fourcc="MJPG"`.

## 6. Decision needed: is the Hiwonder SO-ARM101 kit right for this project?

We had planned to buy the Hiwonder SO-ARM101 advanced kit, $459.99 on
hiwonder.com and also sold on Amazon. Checking it against Hiwonder's own manual
and product page on 2026-09-28 showed that it differs from the reference SO-101
in ways that matter for this project.

| | Hiwonder SO-ARM101 (Advanced, assembled) | Reference SO-101 (WowRobo, Seeed, PartaBot) |
| --- | --- | --- |
| Follower servos | 6 × Hiwonder **HX-30HM**, 30 kg·cm @ 11.1 V, 1:345 | 6 × Feetech STS3215, 1/345 |
| Leader servos | 6 × Hiwonder **HX-10HM**, 10 kg·cm, **1:147 on every joint** | STS3215 with mixed 1/191, 1/345, 1/147, "to make sure it can both sustain its own weight and it can be moved without requiring much force" (LeRobot docs) |
| Supplies | Two 12 V 5 A, both arms at 12 V | Typically 12 V follower, 5 V leader |
| Controller | BusLinker V3.0 | Feetech / Waveshare bus adapter |
| Cameras | 0.3 MP (480p) wrist and 2 MP (1080p) wide-angle external, with a 1.2 kg stand | Usually one camera, or none |
| Software | Hiwonder's own `lerobot.zip` snapshot, older API (`python -m lerobot.record`), conda Python 3.10 | Stock LeRobot from PyPI, which our envs and your teleop helper use |

**Compatibility evidence.**

- *For:* Hiwonder says its HM-series servos use the STS3215's protocol and
  register map, and its driver gives the HX-30HM the STS3215's model number,
  777. LeRobot's Feetech driver checks exactly that number. Hiwonder's own
  manual drives the arms with the standard `so101_follower` / `so101_leader`
  types and the `[feetech]` extra.
- *Against:* Hiwonder's two pull requests adding HX-30HM support to upstream
  LeRobot (#3871 and #3872) were closed without being merged. No LeRobot
  release lists these servos. No model number is published for the leader's
  HX-10HM. Nothing we found shows the kit running on stock LeRobot 0.4.4 or
  0.5.x, or with `so101_unified_teleop.py`.
- *Design difference:* the uniform 1:147 leader should be easier to move, but
  it may not hold its own pose the way the reference leader's 1/345 shoulder
  does.

**Questions for you:**

1. Has the lab, or an earlier cohort, run a Hiwonder SO-ARM101 on stock
   LeRobot or with your teleop helper? Would you accept a non-Feetech servo
   kit for a project whose output is repeatable measurements?
2. If not, should we buy a Feetech STS3215 kit instead? Checked on
   2026-09-28: WowRobo Package 3 is $299 plus international shipping,
   assembled, with one leader, one follower and one camera. Seeed's
   SO-ARM101 Pro kit is $277.99 for the servos, adapter board and cables only,
   so we would print the parts and assemble it ourselves. PartaBot's kits are
   sold out. Either would need a second camera, which the Hiwonder advanced
   kit includes along with a camera stand and a USB hub.
3. Does the lab already have an SO-101 pair we could borrow? Is there a
   department purchase path, such as Amazon Business?

## 7. What we need

**From you:**

- The arm decision in section 6.
- OK for these admin changes on your board:
  - `sudo apt install --no-install-recommends python3-libnvinfer libnvinfer-bin`,
    which adds TensorRT Python and `trtexec` at 10.3.0.30, with no
    `apt update`.
  - Adding the `sjsujetson` user to `dialout` so LeRobot can open the arm's
    serial port.
  - Changing `nvpmodel` and `jetson_clocks` during benchmark runs, restored
    afterwards.
- Workstation access: a GPU with at least 24 GB for RoboTwin 2.0, the FP16
  baseline and the fine-tune. That could be a lab machine, the SJSU cluster,
  or renting. RunPod lists an RTX 4090 at $0.34 per hour on its Community
  Cloud (checked 2026-09-28).

**Team decisions:** the target control rate (30 Hz assumed), and the primary
power mode for the sweep.

**After your OK**, we will apply the admin changes above, about 15 minutes of
work, then run the power-mode check and re-run the resolution check with
clocks pinned.

## 8. Next steps (proposed, next two weeks)

1. Order the arm once section 6 is settled.
2. Finish the harness checks with pinned clocks, and pick the sweep power
   mode.
3. **A small VLA end to end on the Jetson**, SmolVLA through the
   harness with camera frames and power, to shake out the plumbing before
   Hy-VLA. This needs the `smolvla` extra and about 1 GB of weights.
4. Write the safety layer in `hardware/` before any closed-loop run: joint
   limits, a velocity cap using LeRobot's `max_relative_target`, and a watchdog.
5. Secure the workstation. Install RoboTwin 2.0 there and take the FP16
   Hy-VLA reference baseline.
6. Arm day, once it arrives: servo check, calibration, udev rules, teleop, and
   10 throwaway episodes to exercise recording.

## Appendix A: fixes landed in the repo

Branch `feat/jetson-bringup`, awaiting teammate review:

- `hello_latency.py` crashed after writing its record whenever `--out` was
  given.
- The benchmark's dirty-tree check counted the previous run's JSON, so every
  run after the first in a sweep was marked non-reproducible.
- `hello_arm.py` could not import the SO-101 driver on LeRobot 0.4.4 or 0.5.1,
  because the module moved. It would also have dropped into the interactive
  calibration prompt on an uncalibrated arm.
- The power state was recorded as unknown on this board, since there is no
  passwordless sudo. It is now read without sudo.

## Appendix B: changes to the shared board, outside the repo

- `uv` 0.12.20 in `~/.local/bin`, plus a uv-managed CPython 3.12.14.
- `~/lerobot-py310-cuda` (2.4 GB) and `~/lerobot-py312` (1.7 GB). The CUDA
  venv also has `pip.conf` and `constraints.txt`.
- The `vla` and `arm` aliases appended to `~/.bashrc`, 7 lines.
- `~/so101_unified_teleop.py`, identical to the course repo (sha256 `fad46f13…`).
- `~/board-audit-2026-09-28.txt`.

Nothing else changed. No sudo, no apt, no power-mode change, no container
touched. The BRIO's exposure was changed briefly for one test and restored to
auto.
