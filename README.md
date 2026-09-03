# Vision-Language-Action Control Budget on Jetson Orin Nano

Graduate capstone research repository for benchmarking and optimizing a vision-language-action (VLA) manipulation policy for real-time closed-loop control on NVIDIA Jetson Orin Nano (8 GB unified memory), evaluated in RoboTwin 2.0 simulation and on a physical SO-101 arm.

## Team
- Lam Nguyen
- Vi Thi Tuong Nguyen
- James Pham

## Advisor
- Dr. Kaikai Liu

## Course
- San José State University (SJSU) CMPE 295A/B

## Abstract
_TODO: Add project abstract._

## Hardware
- NVIDIA Jetson Orin Nano (8 GB unified memory)
- SO-101 robotic arm
- RGB/RGB-D camera (to be finalized)
- Power and telemetry instrumentation (to be finalized)

## Setup Instructions
_TODO: Add environment, dependency, and deployment setup steps._

## Repository Structure
```text
.
├── benchmarks/   # Latency, control-rate, memory, and power harnesses
├── configs/      # YAML experiment configs for sweeps
├── docs/         # Reports, meeting notes, and paper drafts
├── hardware/     # Jetson setup, arm calibration, and teleoperation
├── results/      # Measurement outputs (CSV/JSON only)
├── sim/          # RoboTwin 2.0 evaluation scripts
└── src/          # Inference, quantization, and TensorRT compilation code
```
