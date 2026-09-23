# Project guides

Start here. These documents take the project from an empty desk to a Jetson
running a vision-language-action policy on a real arm, in the order you should
actually do things.

The audience is someone who has never flashed an embedded board, never wired a
servo, and has never touched LeRobot. Every step says what to run, what you
should see, and why the step exists.

## Our board is already provisioned

Dr. Liu set up `sjsujetson-36` before handing it over. JetPack, CUDA,
TensorRT, `sjsujetsontool` and likely the LeRobot Python environments are all
present.

That makes the first job an audit, not an install. Start at
[Verify an existing setup](02a-verify-existing-setup.md) and run
`bash hardware/audit_board.sh`. Guide 2r exists for the case where a board has
to be rebuilt, and you should not need it.

## Read in this order

| # | Guide | What it covers | When |
| --- | --- | --- | --- |
| 1 | [Hardware purchase guide](01-hardware-purchase-guide.md) | What to buy, what not to buy, and why | Week 1, before anything else |
| 2 | [Verify an existing setup](02a-verify-existing-setup.md) | Audit `sjsujetson-36`, record its state, change nothing | First day on the board |
| 2r | [Jetson setup from scratch](02-jetson-setup.md) | Flashing, hostname, power modes. Reference only | Only if reprovisioning a board |
| 3 | [Python environments](03-python-environments.md) | Virtual envs, Jetson torch wheels, LeRobot, the model stack | After the board boots |
| 4 | [Arm bring-up](04-arm-bringup.md) | Assembly, servo IDs, calibration, teleop, first safe motion | When the arm arrives |
| 5 | [First benchmark](05-first-benchmark.md) | The measurement harness and your first recorded result | After 3 |
| 6 | [Model and simulation](06-model-and-simulation.md) | Hy-Embodied-0.5-VLA, RoboTwin 2.0, the workstation spec and how to rent one | Parallel with 4 |
| 7 | [Troubleshooting](07-troubleshooting.md) | The failures you will actually hit | As needed |

## What this project is doing, in plain terms

A vision-language-action (VLA) model takes camera frames plus a sentence like
"pick up the red block" and outputs joint commands directly. No separate
detector, no motion planner. The catch is that these models are trained and
benchmarked on datacenter GPUs with 80 GB of memory.

Our board has 8 GB, shared between CPU, GPU, the operating system, and the
camera pipeline. The model we are targeting, Hy-Embodied-0.5-VLA, needs about
8.7 GB in FP16. It does not fit. That is the project.

Two numbers define success:

- **Memory.** Get the model under roughly 6 GB so there is headroom for the OS
  and cameras.
- **Control rate.** A tabletop manipulation loop needs new actions fast enough
  that the arm never stalls waiting for the policy. If inference takes 400 ms
  and the arm needs a command every 33 ms, the policy must emit a chunk of
  future actions and stay ahead of the arm.

Everything we do (quantization, TensorRT compilation, fewer flow-matching
steps, longer action chunks) buys speed or memory and costs accuracy. The
deliverable is the curve showing that trade-off, plus the best point on it.

## The five-minute mental model of the hardware

```text
  Host workstation (x86 + NVIDIA GPU)
    |  - trains and fine-tunes policies
    |  - runs RoboTwin 2.0 simulation
    |  - builds reference FP16 baselines
    |
    | ssh / scp over the lab network
    v
  Jetson Orin Nano 8 GB          <- everything we measure happens here
    |
    +-- USB -> SO-101 follower arm (the arm that moves)
    +-- USB -> SO-101 leader arm   (the arm you hold, for demos)
    +-- USB -> wrist camera
    +-- USB -> third-person camera
```

The Jetson is the data-collection and inference node. It is not a training
machine. Any instruction that says "train" belongs on the workstation.

## Ground rules before you touch anything

1. Read [CONTRIBUTING.md](../../CONTRIBUTING.md). Branch per change, no weights
   in git, every result carries its config and commit.
2. Never leave the arm powered and unattended.
3. Record the board's power mode and clock state with every measurement. A
   latency number without `nvpmodel` state is not reproducible and cannot go
   in the report.
4. The Jetson, the arm, and the workstation are shared by three people. Post in
   the team channel before a timing run.

## Where things live

Guides go in `docs/guides/`. The runnable checks referenced by these guides
live in the repo proper:

- `hardware/audit_board.sh`. Read-only inventory of a board someone else set
  up. Modifies nothing.
- `hardware/hello_jetson.py`. Proves the board's GPU stack works.
- `hardware/hello_arm.py`. Proves the arm moves safely.
- `benchmarks/hello_latency.py`. The first real measurement, and the template
  every later harness copies.
