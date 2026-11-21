# Real-Time Control Budgets for Vision-Language-Action Manipulation Policies on the NVIDIA Jetson Orin Nano

**SJSU CMPE 295A/B** — Master's Project

**Team:** Lam Nguyen, Vi Thi Tuong Nguyen, James Pham
**Advisor:** Dr. Kaikai Liu
**Department of Computer Engineering, San José State University**

---

## New to the project? Start here

[docs/guides/](docs/guides/README.md) walks from an empty desk to a policy
running on the arm, in the order to do things. It assumes no prior experience
with embedded boards, servos, or LeRobot.

- [Hardware purchase guide](docs/guides/01-hardware-purchase-guide.md). What to buy, before anything else.
- [Verify an existing setup](docs/guides/02a-verify-existing-setup.md). Auditing `sjsujetson-36`, which arrived already provisioned. Start here.
- [Jetson setup from scratch](docs/guides/02-jetson-setup.md). Flashing and provisioning, kept as reference.
- [Python environments](docs/guides/03-python-environments.md). The Jetson torch wheel problem and how to avoid it.
- [Arm bring-up](docs/guides/04-arm-bringup.md). Servo IDs, calibration, teleoperation, safety.
- [First benchmark](docs/guides/05-first-benchmark.md). The measurement harness.
- [Model and simulation](docs/guides/06-model-and-simulation.md). Hy-Embodied-0.5-VLA, RoboTwin 2.0, and the SO-101 simulator.
- [Troubleshooting](docs/guides/07-troubleshooting.md)

---

## Abstract

> _TODO: Write the abstract once the first round of latency and success-rate
> measurements are in. Target ~200 words. Should state: the problem (VLA
> policies are trained and benchmarked on datacenter GPUs, but manipulation
> requires closed-loop control at the edge under a hard memory and power
> budget), what we measure (end-to-end action latency, sustained control rate,
> peak unified-memory footprint, and board power under a quantization ×
> resolution × action-chunk sweep), where we measure it (RoboTwin 2.0 in
> simulation and a physical SO-101 arm), and the headline result (the
> accuracy-vs-control-rate frontier achievable within 8 GB)._

## Project Overview

Vision-language-action models can generate robot actions directly from camera
frames and a natural-language instruction, but published results are dominated
by throughput on unconstrained hardware. Deploying the same policies on an
8 GB Jetson Orin Nano turns the problem into a **budgeting** one: model
precision, input resolution, and action-chunk length all trade against each
other for a fixed memory and power envelope, and the binding constraint is not
throughput but whether the loop closes fast enough to control the arm.

This project characterizes that trade-off space and then optimizes against it:

1. **Benchmark** a baseline VLA policy on the Orin Nano — latency
   distribution, sustained control rate, peak unified-memory use, and power.
2. **Optimize** via post-training quantization and TensorRT compilation,
   sweeping precision, resolution, and action-chunk length.
3. **Evaluate** task success in RoboTwin 2.0, so that any speedup is reported
   alongside its accuracy cost rather than in isolation. For our own
   single-arm task, the SO-101 simulator in `sim/so101_sim/` measures success
   with the Jetson's measured latency injected.
4. **Validate** the selected operating points on a physical SO-101 arm, where
   real sensing and actuation latency are in the loop.

## Hardware

| Component | Details |
| --- | --- |
| Edge compute | NVIDIA Jetson Orin Nano Developer Kit, 8 GB unified LPDDR5 |
| Storage | NVMe SSD (models, sim assets, episode recordings — kept out of git) |
| Manipulator | SO-101 5-DoF arm with gripper (follower) |
| Teleoperation | SO-101 leader arm for demonstration collection |
| Cameras | USB RGB cameras — one wrist-mounted, one static third-person |
| Power instrumentation | On-board INA3221 rails via `tegrastats` / `jetson-stats` |
| Host workstation | Linux, NVIDIA GPU with >= 24 GB VRAM (RTX 4090 / 3090 / A100), 64 GB RAM, 1 TB NVMe. Runs RoboTwin 2.0, the unquantized reference baseline, and the SO-101 fine-tune. See [the workstation spec](docs/guides/06-model-and-simulation.md#what-the-workstation-actually-needs), which also covers renting one |

Jetson software target: JetPack 6.x (L4T r36.x), CUDA 12.x, TensorRT 10.x.
Exact versions are pinned in [requirements.txt](requirements.txt) once the
board image is verified.

## Setup

Step-by-step instructions live in [docs/guides/](docs/guides/README.md). The
sections below record what was actually installed on our board, which is what
makes a measurement reproducible.

> _TODO: Fill in after the Jetson is flashed and the environment is verified
> end to end. Each section should be reproducible from a fresh board._

### 1. Jetson Orin Nano

Board: `sjsujetson-36`, provisioned by the department. Reach it with
`ssh sjsujetson@sjsujetson-36.local`.

It arrived already set up, so do not flash it, rename it, or update its
container. Audit it instead with `bash hardware/audit_board.sh` and see
[Verify an existing setup](docs/guides/02a-verify-existing-setup.md).

Power modes available, as shipped in mode 2:

| Mode | Name |
| --- | --- |
| 0 | 15W |
| 1 | 25W |
| 2 | MAXN SUPER (default) |

Audited 2026-09-28:

| Component | Version |
| --- | --- |
| Module | Jetson Orin Nano 8 GB, Engineering Reference Developer Kit Super |
| JetPack / L4T | 6.2 / R36.4.3 (`nvidia-jetpack` meta-package not installed) |
| OS / kernel | Ubuntu 22.04.5 LTS / 5.15.148-tegra |
| CUDA | 12.6 (nvcc V12.6.68) |
| cuDNN | 9.3.0.75 |
| TensorRT | 10.3.0.30. Python bindings (`python3-libnvinfer`) and `trtexec` (`libnvinfer-bin`, at `/usr/src/tensorrt/bin/trtexec`) installed 2026-10-26 |
| Storage | 465.8 GB NVMe, root filesystem, 322 GB free |
| Memory | 7.4 GiB unified, zram swap 6 × 635 MB, no NVMe swap |
| `sjsujetsontool` | v1.0.0, container image `jetson-llm:v1` |

`nvpmodel -q` works without sudo on this image, and GPU clock limits are
readable from `/sys/class/devfreq/17000000.gpu`, which is how the harness
records clock state. Changing the mode or running `jetson_clocks` still needs
sudo.

#### Sweep setting

Decided 2026-10-26. Every sweep record is taken in **mode 2, MAXN SUPER, with
`jetson_clocks` on**, at a **30 Hz** control rate. Modes 0 and 1 are used
only for the power comparison of the best configurations.

- **Not mode 1.** [First benchmark](docs/guides/05-first-benchmark.md)
  suggested 25 W, but `/etc/nvpmodel.conf` caps the CPU at 1.344 GHz in that
  mode, below the 1.498 GHz of the 15 W mode and the 1.728 GHz of MAXN SUPER.
  SmolVLA's chunk time is bound by the CPU thread that launches GPU work, so a
  lower CPU cap makes every chunk slower.
- **MAXN SUPER costs no extra energy per inference.** In the 2026-09-28
  pinned checks at 1344 px, p50 was 12.60, 10.30 and 9.91 ms in modes 0, 1
  and 2, at 155, 148 and 154 mJ per inference. The GPU held its pinned clock
  in every run, with no throttling.
- **Pinned clocks** because with the clocks scaling, latency stayed flat from
  112 px to 672 px while board power rose from 6.5 W to 14.9 W.
- **30 Hz** is LeRobot's recording default, and `hardware/so101_lab.sh`
  records demos and runs policies at it. A policy has to run at the rate of
  the demos it learned from.

`benchmarks/hello_latency.py` warns when a run is not in this setting.
`bash benchmarks/baseline_runs.sh` pins it, takes the baseline records, and
puts the board back afterwards.

### 2. Python environment

Two environments, built 2026-09-28 with `uv` 0.12.20 following
[Python environments](docs/guides/03-python-environments.md) path A. Their exact
package sets are in [hardware/envs/](hardware/envs/).

| Env | Python | Key packages | Use |
| --- | --- | --- | --- |
| `~/lerobot-py310-cuda` | 3.10.12 (system), `--system-site-packages` | torch 2.8.0 + CUDA 12.6, torchvision 0.23.0, lerobot 0.4.4 `[feetech]`, numpy 1.26.4, opencv-python-headless 4.11.0.86 | Inference, benchmarks. Alias `vla` |
| `~/lerobot-py312` | 3.12.14 (uv-managed) | lerobot 0.5.1 `[feetech]`, torch 2.10.0+cpu, numpy 2.2.6 | Teleop, calibration, recording. Alias `arm` |

Install order in the CUDA env: torch, torchvision and torchaudio **first**, from
`https://pypi.jetson-ai-lab.io/jp6/cu126` (wheel names and sha256 in
[requirements.txt](requirements.txt)), then `nvidia-cudss-cu12`, then LeRobot,
then the numpy/OpenCV pins. A `pip.conf` inside the venv applies
`constraints.txt`, so a later `pip install` that would replace the Jetson torch
fails instead of silently going CPU-only.

`vla` also sets `PYTHONNOUSERSITE=1`. Without it, `~/.local` (numpy 2.2.6,
tensorflow, protobuf 7) sits on `sys.path` ahead of the system packages.

### 3. Simulation

**SO-101 simulator.** MuJoCo, on Windows, macOS and Linux. Run the launcher in
`sim/so101_sim/` (`run_app.bat`, `run_app.command` or `run_app.sh`); the first
run builds its own environment with LeRobot 0.5 (tested on 0.5.1) on Python 3.12. It records
LeRobot datasets with the same features as `hardware/so101_lab.sh`, and `eval`
writes success records to `results/` with chunk latency taken from a harness
record. See [sim/so101_sim/README.md](sim/so101_sim/README.md).

Two expert datasets made with it are public on Hugging Face, so
`lerobot-train --dataset.repo_id=<id>` trains on them without generating them
again:

- [`duylam1407/so101_sim_multi_object_noisy`](https://huggingface.co/datasets/duylam1407/so101_sim_multi_object_noisy):
  300 episodes, five kinds of object, scenes where the instruction picks one
  object out of three by colour or by kind, real-arm noise, overhead camera.
  Use this one for new policies.
- [`duylam1407/so101_sim_cube_box`](https://huggingface.co/datasets/duylam1407/so101_sim_cube_box):
  200 episodes, one cube, no noise, front camera. The first runs used it.

Which commit made each, and the training runs on them, are under
[Shared datasets](sim/so101_sim/README.md#shared-datasets).

**RoboTwin 2.0.**

> _TODO: RoboTwin 2.0 install, asset download, and the task subset used for
> evaluation._

### 4. SO-101 arm

> _TODO: Servo ID assignment, calibration procedure, leader/follower pairing,
> camera intrinsics, and the workspace safety limits enforced before any
> closed-loop run._

### 5. Reproducing an experiment

> _TODO: `configs/` → run → `results/` walkthrough, including how a config
> hash is recorded alongside its measurements._

## Repository Structure

```text
.
├── src/            Inference, quantization, and TensorRT compilation code
├── benchmarks/     Latency, control-rate, memory, and power harnesses
├── sim/            RoboTwin 2.0 evaluation scripts, and so101_sim/: the SO-101
│                   MuJoCo simulator (Windows, macOS, Linux)
├── hardware/       Jetson setup, arm calibration, teleoperation
├── configs/        YAML experiment configs for the sweep
├── results/        Measurement outputs (CSV/JSON only)
├── docs/           Reports, paper drafts
│   └── guides/     Step-by-step setup and bring-up guides
├── meetings/       Advisor and team meeting notes, one file per meeting
├── requirements.txt
├── CONTRIBUTING.md
└── .github/        Issue and pull request templates, the simulator's CI workflow
```

### What goes where

- **`src/`** — the policy wrapper, preprocessing, quantization passes, ONNX
  export, and TensorRT engine builds. Everything here must run without a
  specific checkpoint on disk: model paths are arguments, never constants.
- **`benchmarks/`** — measurement harnesses only. Each writes a machine-
  readable record to `results/` and prints a human summary. Keep the
  measurement logic separate from the inference code it measures.
- **`sim/`** — RoboTwin 2.0 task setup, rollout drivers, and success-rate
  scoring.
  [`sim/so101_sim/`](sim/so101_sim/README.md) simulates our single-arm cube-to-box
  task: demo recording, and closed-loop success under the Jetson's measured latency.
- **`hardware/`** — board provisioning scripts, arm calibration, teleoperation
  and demonstration capture, and the safety limits used on the physical arm.
- **`configs/`** — one YAML per sweep point (precision × resolution × action
  chunk × control rate). Configs are the unit of reproducibility; results
  reference the config that produced them.
- **`results/`** — CSV/JSON only. Plots are regenerated from these files, not
  committed as source of truth.
- **`docs/`** — 295A/295B reports, figures, and paper
  drafts.
- **`meetings/`** — one note per advisor or team meeting
  (`YYYY-MM-DD-advisor.md`, `YYYY-MM-DD-team.md`), indexed in
  [meetings/README.md](meetings/README.md).

### What this repo must never contain

Model checkpoints, TensorRT engines, datasets, episode recordings, videos, and
large logs are all excluded by [.gitignore](.gitignore). Store them on the
NVMe or in shared team storage and reference them by path in a config. If you
find yourself running `git add -f` on a weight file, stop. Datasets and
checkpoints the team shares go on Hugging Face, and the README of the code that
made them records the repo id and the commit, as
[Shared datasets](sim/so101_sim/README.md#shared-datasets) does.

## Status

295A builds the baseline that 295B improves on: a VLA policy running on the
Jetson, the measurement harness, and evaluation in simulation (agreed with the
advisor on 2026-09-29). As of 2026-11-21 most pieces are merged:

| Baseline piece | State |
| --- | --- |
| Measurement harness (latency, memory, power, clocks) | Done, on `main` |
| SmolVLA on the Jetson | Timed in the sweep setting on `main`. With live cameras and pinned clocks ([#13](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/13)): 10 denoising steps take 916 ms per chunk and async still runs dry; 5 steps take 546 ms and hold 30 Hz |
| Hy-VLA in BF16 on the workstation | RoboTwin, 5 tasks × 100 episodes: 473/500, 94.6% ([#20](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/20)); timing ([#18](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/18)). The released checkpoint runs in BF16, so 295B compares against BF16 |
| Hy-VLA on the Jetson | **Runs, first numbers.** FP16 does not fit: TensorRT 10.3 asks for 5.3 GB of scratch to build the backbone as one engine, with about 5.8 GB free, and INT4 needs the same scratch and is the slowest. Package `hyvla-int8-v1`: the backbone as four W8A8 INT8 engines (language layers 0–1 and the vision fc2 and merger in FP16) plus an FP16 action expert. One chunk (backbone and 10 expert steps) takes 614 ms p50, actions match FP32 at cosine 0.99, and the lowest free board RAM is 439 MiB ([#19](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/19), [record](results/20261121-015818_orin-nano-hyvla-int8-v1-package.json); export in draft [#22](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/22)). Clocks not pinned, calibrated on the 2 test cases, no power yet. INT4 with pinned clocks is no faster than FP16 but 3.8× smaller ([#15](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/15)) |
| Simulation with the Jetson in the loop | Websocket link from RoboTwin to the Jetson, frames scaled to 224 px, Wi-Fi round trip p50 29 ms, p99 103 ms ([#17](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/17)); its messages refuse object arrays, which would unpickle. No policy runs yet |
| SO-101 simulator | Five kinds of object, scenes where the instruction picks one by colour or kind, real-arm noise, two public datasets (see [Setup](#setup)) ([#11](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/11)). An ACT policy, as a pipeline check, succeeds in about a third of episodes. Across chunk latency (50 episodes a point), async holds up to 750 ms and falls beyond 833 ms, its budget with 25 of 50 actions left; sync declines slowly from 0 ms. SmolVLA is not trained yet |
| Real arm | Our own kit is calibrated; teleop works and the first demos are recorded ([#21](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/21)). The safety layer passes all nine E-8 bench cases on the team follower, with a table-height check calibrated from 13 jaw-tip touches ([#14](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/14)). No autonomous run yet |

The sim and the board agree on the async budget: ACT in the simulator falls
off beyond 833 ms, and SmolVLA at 10 steps (916 ms) runs dry on the Jetson
while 5 steps (546 ms) do not.

### What the baseline must fix for 295B

Quantization, TensorRT, resolution and chunk-length results in 295B count only
against a frozen reference. By the end of 295A:

1. **Reference setup, written down:** model checkpoints, the RoboTwin task and
   seed list, the SO-101 task and seeds, and the sweep setting.
2. **One baseline table**, a row per model and machine: chunk latency p50 and
   p99, sustained control rate, peak board memory, board power, success in
   RoboTwin and the SO-101 simulator, and success on the real arm.
3. **Success versus latency:** SmolVLA in the SO-101 simulator at the latencies
   the Jetson measured, sync and async. This is the project's title claim.

### Next steps

In order to the end of the term; each says what it needs and what it produces.

**This week**

1. **Use the team arm's joint map in the simulator.** The table calibration
   gives it: `shoulder_lift` is 10.3° and `wrist_flex` 6.4° off the simulator's
   degrees ([sim/so101_sim/joint_maps/team_follower.json](sim/so101_sim/joint_maps/team_follower.json),
   see [Matching the real arm](sim/so101_sim/README.md#matching-the-real-arm)).
   Pass `--joint-map` for new datasets and the *Real follower* mirror. Training
   SmolVLA on the existing dataset (step 3) is still fine for simulation.
2. **Freeze the reference setup** (point 1 above) in `configs/`.
3. **Train SmolVLA on the sim dataset** (RTX 4090). Run the command under
   [Shared datasets](sim/so101_sim/README.md#shared-datasets), on
   `so101_sim_multi_object_noisy`; it downloads the data by id. Upload
   `checkpoints/last/pretrained_model` to Hugging Face as a model repo and add
   it to that section with the commit it was trained at.
4. **First Hy-VLA sweep points on the Jetson** (critical path). Time the
   `hyvla-int8-v1` chunk with pinned clocks and add board power. Calibrate INT8
   on RoboTwin demos (seeds from 0, kept apart from the evaluation seeds), make
   a new package, and check accuracy on held-out cases. Free RAM is tight
   (439 MiB at the lowest), so record the peak with the policy loop running.

**Next week**

5. **SmolVLA success versus latency in the SO-101 simulator.**
   `python -m so101_sim eval` with `--objects all --scene mixed --noise none,real
   --latency-ms 0,250,500,750,1000,1250`, sync and async, 50 episodes each,
   plus `--latency-from` the pinned live-loop records for 5 and 10 steps. The
   ACT sweep at commit `77255d7` is the cross-check.
6. **First autonomous run on our arm**, now that E-8 passes: SmolVLA fine-tuned
   on the recorded demos, 5 denoising steps, async, through
   `so101_safe_follower`, with success counted separately from teleop. Before
   it, the arm-only checks in
   [Matching the real arm](sim/so101_sim/README.md#matching-the-real-arm):
   check `shoulder_pan`, `wrist_roll` and the joint signs against the
   simulator (the table calibration covers the other joints), measure the overhead camera for
   `--camera1-pos` and `--camera1-target`, check the backlash and camera delay
   estimates, and record a few trajectories to fit the servo gains.
7. **RoboTwin with the Jetson in the loop** over #17, with the INT8 engines
   from step 4, on the frozen tasks and seeds, compared episode by episode with
   the BF16 workstation run in #20 (the L3 check).

**Last stretch**

8. **295A report** around the baseline table, the success-versus-latency
   curve, and the finding that Hy-VLA does not fit the board in FP16 or as
   INT4, and does as INT8 in four engines.

**If Hy-VLA's success on the board is not measured this term:** the 295A
baseline is SmolVLA end to end on the Jetson (timing, simulated and real
success), Hy-VLA in BF16 on the workstation (94.6%), and Hy-VLA's INT8 chunk
timing and memory on the board. Its success there becomes the first 295B
result.

**Later, in 295B:** more language grounding in the SO-101 simulator. Picking by
colour and by kind is done; next is target places (several boxes, or "left
of"), so the instruction also changes where the object goes.

See the [2026-09-29 advisor update](docs/reports/2026-09-29-advisor-update.md)
for the earlier state.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
