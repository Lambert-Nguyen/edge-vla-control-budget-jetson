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
- **The benchmark harness is done.** It records board power alongside
  latency, memory, power mode, clock state and temperature in every result,
  and all three sanity checks pass.
- **Our first measurements changed how we will measure.** With the default
  clock scaling, latency stayed flat across a 36× increase in pixels while
  board power rose from 6.5 W to 14.9 W. With the clocks locked, latency
  follows the work, the worst case at 224 px drops from 4.9 to 1.4 ms, and
  large images take about a quarter less time. We will lock the clocks for
  every experiment.
- **The arm software is staged and pre-tested.** This caught a bug that would
  have stopped the arm's first test script on day one.
- **The arm is not ordered.** The Hiwonder kit we planned to buy turns out to use
  Hiwonder's own servos, not the Feetech servos LeRobot is built around. **We
  need your call on this before ordering** (section 5).

## 1. Status

| Workstream | Status | Notes |
| --- | --- | --- |
| Arm purchase | **Open** | Not ordered. Hiwonder question in section 5 |
| Board audit | Done | Re-audited 2026-09-28, no system-level changes |
| Python environments | **Done** | Both envs built and verified, versions pinned |
| Arm software | Staged | Tools installed and tested without hardware. Hardware steps wait for the arm |
| Benchmark harness | **Done** | Power sampling added. All three sanity checks pass with the clocks locked |
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
   `apt update` needed. We have not run it, because it needs sudo (section 6).

## 3. Python environments

| | `~/lerobot-py310-cuda` (alias `vla`) | `~/lerobot-py312` (alias `arm`) |
| --- | --- | --- |
| Python | 3.10.12 (system), `--system-site-packages` | 3.12.14 (uv-managed) |
| torch | 2.8.0, CUDA 12.6, Jetson wheel from `pypi.jetson-ai-lab.io/jp6/cu126` | 2.10.0+cpu (by design) |
| LeRobot | 0.4.4 `[feetech]` | 0.5.1 `[feetech]` |
| Other pins | numpy 1.26.4, opencv-python-headless 4.11.0.86, pygame 2.6.1 | numpy 2.2.6 |
| Verified by | `hardware/hello_jetson.py`: 14 checks, 0 failed, 2 warnings. FP16 matmul 4096³ in 27.5 ms (5.0 TFLOP/s, default clocks) | `lerobot-info`, `pip check` clean, Feetech SDK imports |

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

All 25 records are committed in `results/` against clean commits. The first 13
ran with the board's default clock scaling. The other 12 ran later on
2026-09-28 with the clocks locked (`jetson_clocks`), through
`benchmarks/sanity_checks.sh`, which switches the power mode for each check
and then restores the board. The desktop session was running throughout, at a
load average of about 1.3 to 2.1 on 6 cores.

| Check | Result | Verdict |
| --- | --- | --- |
| Warmup matters | p99 397.5 ms with no warmup, against 5.7 ms with 20 warmup iterations | Passes |
| Sampler overhead | p99 4.90 ms with power sampling, 4.98 ms without, over 3000 iterations | No measurable perturbation |
| Latency rises with resolution | Clocks locked: flat near 1.15 ms up to 336 px, then 2.8, 9.9 and 22.7 ms at 672, 1344 and 2048 px. Default scaling stayed flat to 672 px | Passes with clocks locked |
| Power mode changes the answer | At 1344 px: 12.6 ms in 15 W, 10.3 ms in 25 W, 9.9 ms in MAXN SUPER. Each record names its own mode | Passes |

**The resolution check, default clock scaling against locked clocks** (MAXN
SUPER, 1000 to 2000 iterations per point; locked, the GPU ran at 1020 MHz at
every size):

| Resolution | p50 default (ms) | p50 locked (ms) | p99 default (ms) | p99 locked (ms) | VDD_IN default (W) | VDD_IN locked (W) | GPU clock default (MHz) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 112 | 2.57 | 1.14 | 6.45 | 1.33 | 6.49 | 7.44 | 306 – 408 |
| 224 | 2.52 | 1.17 | 5.62 | 1.46 | 7.13 | 8.50 | 408 – 714 |
| 336 | 2.06 | 1.16 | 5.35 | 1.36 | 8.87 | 10.75 | 408 – 918 |
| 672 | 2.53 | 2.77 | 5.94 | 3.16 | 14.87 | 14.25 | 1020 |
| 1344 | 13.82 | 9.91 | 16.29 | 10.12 | 16.23 | 15.47 | 1020 |
| 2048 | 30.74 | 22.73 | 32.34 | 27.71 | 16.36 | 15.60 | 1020 |

Left to itself, the governor answers extra work with higher clocks and more
power, so latency stayed flat from 112 to 672 px while board power rose from
6.5 W to 14.9 W. With the clocks locked, latency follows the work. Up to 336 px
it holds at the stand-in model's fixed overhead of about 1.15 ms, and above that
it climbs with the pixel count: 2.29 times longer from 1344 to 2048 px for 2.32
times the pixels.

Locking also removed most of the jitter. At 224 px, over 3000 steps each, the
worst 1% went from 4.90 ms to 1.38 ms. Large images took about a quarter less
time: 13.8 to 9.9 ms at 1344 px, and 30.7 to 22.7 ms at 2048 px. That happened
although the GPU ran at 1020 MHz in both cases, so another clock was scaling,
most likely the memory clock. `jetson_clocks` locks it too, but we can't read
it without root. The one near-tie is 672 px, where the medians are 2.5 and
2.8 ms but the worst case halves from 5.9 to 3.2 ms. Locking costs 1 to 2 W
more at small sizes.

**Power modes, clocks locked:**

| Mode | CPU / GPU clock (MHz) | 224 px p50 / p99 (ms) | 1344 px p50 / p99 (ms) | Power at 1344 px (W) | Energy per step at 1344 px (mJ) |
| --- | --- | --- | --- | --- | --- |
| 15 W (mode 0) | 1497 / 612 | 1.34 / 1.57 | 12.60 / 16.76 | 12.2 | 155 |
| 25 W (mode 1) | 1344 / 918 | 1.44 / 1.80 | 10.30 / 14.01 | 14.2 | 148 |
| MAXN SUPER (mode 2) | 1728 / 1020 | 1.15 / 1.38 | 9.91 / 13.05 | 15.4 | 154 |

- Lower modes trade speed for power at about the same energy per step,
  roughly 150 mJ at 1344 px.
- From MAXN SUPER to 15 W the GPU clock drops 40% but latency rises only 27%,
  so this workload isn't limited by the GPU clock alone.
- The 25 W mode has the slowest CPU (1344 MHz, against 1497 in 15 W), and it
  was the slowest mode at small sizes, where per-step CPU overhead dominates.
  That matters for a pipeline with CPU-side preprocessing.
- No run throttled. The chip peaked at 61.5 °C.

What this means for the project:

- Every experiment runs with the clocks locked. Default scaling adds jitter
  and understates the board.
- We still need to choose the power mode for the main sweep. MAXN SUPER with
  locked clocks was fastest, with no throttling in these short runs, though
  longer runs of the real model need watching; the harness flags a locked
  clock that drops. 15 W gives about the same energy per step, which fits a
  battery-powered story.
- Energy per inference, not just latency, belongs in the trade-off curves.

## 5. Decision needed: is the Hiwonder SO-ARM101 kit right for this project?

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

## 6. What we need

**From you:**

- The arm decision in section 5.
- OK for these admin changes on your board:
  - `sudo apt install --no-install-recommends python3-libnvinfer libnvinfer-bin`,
    which adds TensorRT Python and `trtexec` at 10.3.0.30, with no
    `apt update`.
  - Adding the `sjsujetson` user to `dialout` so LeRobot can open the arm's
    serial port.
  - Switching power modes and locking the clocks during benchmark runs. We
    did this once already, on the evening of 2026-09-28, for the harness
    checks, and put the board back to MAXN SUPER with clocks unlocked
    afterwards. Is it OK to keep doing it for experiments?
- Workstation access: a GPU with at least 24 GB for RoboTwin 2.0, the FP16
  baseline and the fine-tune. That could be a lab machine, the SJSU cluster,
  or renting. RunPod lists an RTX 4090 at $0.34 per hour on its Community
  Cloud (checked 2026-09-28).

**Team decisions:** the target control rate (30 Hz assumed), and the primary
power mode for the sweep. The data is in section 4.

**After your OK**, we will apply the two installs above, about 10 minutes of
work.

## 7. Next steps (proposed, next two weeks)

1. Order the arm once section 5 is settled.
2. Pick the power mode for the main experiments, using the data in
   section 4.
3. **A small VLA end to end on the Jetson**: SmolVLA through the harness with
   power, on recorded frames first and on the kit's cameras once they arrive.
   This shakes out the plumbing before Hy-VLA, and needs the `smolvla` extra
   and about 1 GB of weights.
4. Write the safety layer in `hardware/` before any closed-loop run: joint
   limits, a velocity cap using LeRobot's `max_relative_target`, and a watchdog.
5. Secure the workstation. Install RoboTwin 2.0 there and take the FP16
   Hy-VLA reference baseline.
6. Arm day, once it arrives: servo check, calibration, udev rules, teleop, and
   10 throwaway episodes to exercise recording. Measure the kit's cameras with
   our camera test tool: frame rate, CPU cost and preprocessing time.

## Appendix A: fixes landed in the repo

Merged into `main` on 2026-09-28 as pull request #3, with every commit kept.
The locked-clock records and the script that ran them are on branch
`exp/pinned-clock-checks`, not yet reviewed.

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
- `~/board-audit-2026-09-28.txt`, and the log of the harness checks,
  `~/sanity-checks-20260928-214925.log`.
- GitHub's CLI, `gh` 2.101.0, in `~/.local/bin`, used to open our pull
  requests.

Nothing else changed. Sudo was used once, on the evening of 2026-09-28, for
the harness checks, and only for `nvpmodel` and `jetson_clocks`. The board went
back to MAXN SUPER with clocks unlocked, and we checked it afterwards. No apt,
and no containers touched.
