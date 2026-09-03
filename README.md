# Real-Time Control Budgets for Vision-Language-Action Manipulation Policies on the NVIDIA Jetson Orin Nano

**SJSU CMPE 295A/B** — Master's Project

**Team:** Lam Nguyen, Vi Thi Tuong Nguyen, James Pham
**Advisor:** Dr. Kaikai Liu
**Department of Computer Engineering, San José State University**

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
| Host workstation | x86-64 + discrete NVIDIA GPU, for RoboTwin 2.0 sim and reference (unquantized) baselines |

Jetson software target: JetPack 6.x (L4T r36.x), CUDA 12.x, TensorRT 10.x.
Exact versions are pinned in [requirements.txt](requirements.txt) once the
board image is verified.

## Setup

> _TODO: Fill in after the Jetson is flashed and the environment is verified
> end to end. Each section should be reproducible from a fresh board._

### 1. Jetson Orin Nano

> _TODO: JetPack flashing, power mode (`nvpmodel`) and clock settings, swap
> and zram configuration, NVMe mount, and the CUDA/cuDNN/TensorRT versions the
> measurements were taken against._

### 2. Python environment

> _TODO: Virtual environment creation and the install order for the
> Jetson-specific PyTorch wheel, which must come from the NVIDIA index rather
> than PyPI. See [requirements.txt](requirements.txt)._

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

295A: scaffolding and baseline characterization in progress.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
