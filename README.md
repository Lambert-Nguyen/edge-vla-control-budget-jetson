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
- [Model and simulation](docs/guides/06-model-and-simulation.md). Hy-Embodied-0.5-VLA and RoboTwin 2.0.
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
   alongside its accuracy cost rather than in isolation.
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
| TensorRT | 10.3.0.30 runtime. Python bindings (`python3-libnvinfer`) and `trtexec` (`libnvinfer-bin`) are **not installed** yet |
| Storage | 465.8 GB NVMe, root filesystem, 322 GB free |
| Memory | 7.4 GiB unified, zram swap 6 × 635 MB, no NVMe swap |
| `sjsujetsontool` | v1.0.0, container image `jetson-llm:v1` |

`nvpmodel -q` works without sudo on this image, and GPU clock limits are
readable from `/sys/class/devfreq/17000000.gpu`, which is how the harness
records clock state. Changing the mode or running `jetson_clocks` still needs
sudo.

> _TODO: the power mode the main sweep is taken in, once the team agrees.
> [First benchmark](docs/guides/05-first-benchmark.md) suggests mode 1 (25 W)._

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

### 3. RoboTwin 2.0 simulation

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
├── sim/            RoboTwin 2.0 evaluation scripts
├── hardware/       Jetson setup, arm calibration, teleoperation
├── configs/        YAML experiment configs for the sweep
├── results/        Measurement outputs (CSV/JSON only)
├── docs/           Reports, meeting notes, paper drafts
│   └── guides/     Step-by-step setup and bring-up guides
├── requirements.txt
├── CONTRIBUTING.md
└── .github/        Issue and pull request templates
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
- **`hardware/`** — board provisioning scripts, arm calibration, teleoperation
  and demonstration capture, and the safety limits used on the physical arm.
- **`configs/`** — one YAML per sweep point (precision × resolution × action
  chunk × control rate). Configs are the unit of reproducibility; results
  reference the config that produced them.
- **`results/`** — CSV/JSON only. Plots are regenerated from these files, not
  committed as source of truth.
- **`docs/`** — 295A/295B reports, advisor meeting notes, figures, and paper
  drafts.

### What this repo must never contain

Model checkpoints, TensorRT engines, datasets, episode recordings, videos, and
large logs are all excluded by [.gitignore](.gitignore). Store them on the
NVMe or in shared team storage and reference them by path in a config. If you
find yourself running `git add -f` on a weight file, stop.

## Status

295A: scaffolding and baseline characterization in progress. As of
2026-09-28 the Jetson environments and the measurement harness (latency,
memory, board power) are verified on `sjsujetson-36`. The arm is not ordered
yet. See the [2026-09-29 advisor update](docs/reports/2026-09-29-advisor-update.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
