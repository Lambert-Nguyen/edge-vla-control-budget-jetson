# Model and simulation

Goal: understand what Hy-Embodied-0.5-VLA actually is, why it does not fit on
the board, and how the simulation half of the project connects to the physical
half.

This one is more reading than typing. Do it in parallel with
[arm bring-up](04-arm-bringup.md) so the team is not blocked on shipping.

It also contains the concrete workstation specification, including what to do
if the team does not have one. Settle that early, because the reference
baseline and the fine-tune both depend on it.

## The model

[Hy-Embodied-0.5-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)
(Tencent Hunyuan, released June 2026, Apache-2.0) is a vision-language-action
model with two parts:

- A 4B-parameter mixture-of-transformers vision-language backbone that reads
  camera frames and the task instruction.
- A 370M-parameter flow-matching action expert that turns the backbone's
  representation into continuous joint commands.

Details that matter for our work:

| Property | Value | Why we care |
| --- | --- | --- |
| FP16 weight size | ~8.7 GB | Larger than the whole board's 8 GB |
| Stated VRAM requirement | >= 16 GB | Two boards' worth |
| Flow-matching steps | 10 Euler steps at inference | A direct latency lever. Each step is a forward pass of the action expert |
| History frames | K = 6, compressed by a parameter-free memory encoder | Six frames of camera buffer on a memory-constrained board |
| Action format | delta-chunk, decoupled from embodiment kinematics | This is what makes SO-101 transfer plausible |
| Input resolution | 224 × 224 per camera | Another sweep axis |
| RoboTwin 2.0 result | 90.9% clean, 90.1% randomized | The accuracy we are trying to stay within 5 points of |

Two checkpoints on Hugging Face:

- `tencent/Hy-Embodied-0.5-VLA-UMI`, generalist, pre-trained on 10,000+ hours
  of UMI demonstrations. The base for fine-tuning.
- `tencent/Hy-Embodied-0.5-VLA-RoboTwin`, fine-tuned on RoboTwin 2.0. This is
  the one our simulation numbers compare against.

## Why it does not fit, spelled out

8.7 GB of FP16 weights against 8 GB of LPDDR5 that is already holding the
kernel, the desktop or console, the camera pipeline, the CUDA context, and
activations. Realistically about 6 GB is available to a model.

So the paths to fitting are:

- **INT8 weights** roughly halves it to ~4.4 GB. Comfortable, and the accuracy
  cost is usually small for the backbone.
- **INT4 / W4A16 weights** takes it to ~2.2 GB, which leaves real headroom for
  activations and the camera pipeline, at a larger accuracy cost.
- **Mixed** is the likely answer. Quantize the 4B backbone hard, keep the 370M
  action expert at higher precision, because the action expert runs 10 times
  per inference and it is the part that determines whether the gripper lands on
  the block.

That last sentence is a hypothesis, not a result. Testing it is the project.

## What the workstation actually needs

"A workstation with an NVIDIA GPU" is too vague to shop against. Here is the
real requirement, job by job.

| Job | VRAM | Why that number |
| --- | --- | --- |
| RoboTwin 2.0 simulation | 8 GB, 16 GB comfortable | SAPIEN rendering plus the policy under test. More VRAM means more parallel rollouts, which is the difference between an hour and a morning per sweep point |
| FP16 reference baseline | 16 GB minimum, 24 GB comfortable | The model's own stated floor. 8.7 GB of weights plus activations and CUDA context |
| SO-101 fine-tune, frozen backbone | 24 GB | See the memory math below |
| SO-101 fine-tune, full | ~80 GB | Not happening on one consumer card. Do not plan for this |

So the binding requirement is **24 GB of VRAM**, set by the fine-tune.

### Concrete specification

| Component | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| GPU | 16 GB VRAM, Ampere or newer (sm_80+) | 24 GB VRAM (RTX 4090, RTX 3090, A100 40 GB) | VRAM is the only spec that gates anything. A slow 24 GB card beats a fast 16 GB one |
| System RAM | 32 GB | 64 GB | Dataset loading and SAPIEN both want headroom |
| Disk | 500 GB free | 1 TB NVMe | Checkpoints, RoboTwin assets, your own episodes |
| OS | Ubuntu 22.04 or 24.04 | Ubuntu 22.04 | See the warning below |
| CUDA | 12.x | 12.4+ | Matches the model's requirement and keeps ONNX export consistent with the board |
| Python | 3.12 | 3.12 | What the Hy-VLA repo recommends |

Cards that work: RTX 4090 (24 GB), RTX 3090 or 3090 Ti (24 GB), RTX A5000
(24 GB), A100 (40 or 80 GB), L40S (48 GB), H100 (80 GB).

Cards that do not: anything with 12 GB or less, and anything pre-Ampere. An
RTX 4070 Ti at 12 GB cannot hold the baseline. A 16 GB card (4060 Ti 16 GB,
4080, V100) runs the sim and the baseline but not the fine-tune.

**The OS warning is not optional.** If the machine runs Windows, that is a
bigger obstacle than the GPU. SAPIEN and RoboTwin want native Linux, and Vulkan
passthrough under WSL2 is fragile enough that you will spend your time
debugging a graphics stack instead of doing the project. Dual-boot Ubuntu.

### One thing the workstation cannot do

Build TensorRT engines for the Jetson. Engines are tied to both the TensorRT
version and the GPU architecture. Your desktop card is sm_89 or sm_80, the Orin
is sm_87. ONNX export and quantization calibration happen on the workstation,
the engine build happens on the board. Guide 05 and
[requirements.txt](../../requirements.txt) say the same thing from the other
direction.

## Why the fine-tune needs 24 GB

Worth showing the arithmetic, because it drives the whole hardware decision.

A full fine-tune of all 4.37B parameters with AdamW needs roughly:

```text
bf16 weights            4.37e9 x 2 bytes  =  8.7 GB
bf16 gradients          4.37e9 x 2 bytes  =  8.7 GB
fp32 optimizer states   4.37e9 x 8 bytes  = 35.0 GB   (exp_avg + exp_avg_sq)
fp32 master weights     4.37e9 x 4 bytes  = 17.5 GB
                                            -------
                                             69.9 GB, before activations
```

That is a multi-GPU job. Not our project.

**Freeze the backbone and train only the 370M action expert instead:**

```text
frozen bf16 backbone (forward only)        =  8.0 GB
bf16 action-expert weights + grads         =  1.5 GB
fp32 optimizer states + master weights     =  4.4 GB
activations, with gradient checkpointing   =  2-4 GB
                                              -------
                                              ~16-18 GB
```

That fits in 24 GB with room to spare.

This is also the better choice scientifically, not a compromise. The
delta-chunk action representation is the embodiment-specific part of the
architecture, and adapting the action expert is what it was designed for. If
freezing underfits on your demonstrations, the next step is LoRA on the
backbone plus a full action-expert train, which still fits in 24 GB.

Expect batch size 1 or 2 with gradient accumulation. The published SFT used
batch 32 to 128, which is cluster scale. You match the effective batch, not the
per-step one, so wall-clock time goes up.

## If you do not have a workstation

Ask in this order before spending anything.

1. **Dr. Liu.** The curriculum's own LeRobot page says heavy training belongs on
   "a workstation GPU, Jetson Thor, or a CUDA container," which suggests the lab
   has something. Ask in the same message as the arm and the teleop script.
2. **SJSU research computing.** A faculty advisor can usually sponsor student
   cluster accounts. Free, with job queues instead of instant access.
3. **Teammates.** There are three of you. Someone may have a gaming desktop with
   a 3090 or 4090 in it.
4. **Rent by the hour.** Covered below. Cheap enough that it should not be the
   reason to narrow the project.

### Renting: what moves and what cannot

| Stays local, always | Moves to the cloud |
| --- | --- |
| Every Jetson measurement | RoboTwin 2.0 and its assets |
| The arm, cameras, teleoperation | FP16 reference baseline |
| TensorRT engine builds (sm_87) | ONNX export, quantization calibration |
| Closed-loop physical runs | The SO-101 fine-tune |

The split is clean because the Jetson is the thing being measured. Nothing that
produces a number for the report leaves the lab.

### Vendor recommendations

Prices below are ballparks from September 2026 and they move constantly. Check
current rates before committing.

| Vendor | Card and rough rate | Persistent storage | Use it when |
| --- | --- | --- | --- |
| [RunPod](https://www.runpod.io/) Community Cloud | RTX 4090 ~$0.34/hr, A100 80 GB ~$1.19/hr | Network volumes, billed per GB-month | Default choice for this project |
| [RunPod](https://www.runpod.io/) Secure Cloud | RTX 4090 ~$0.69/hr | Same | Long fine-tunes where an eviction would hurt |
| [Vast.ai](https://vast.ai/) | RTX 4090 from ~$0.34/hr, spot sometimes near $0.14/hr | Varies by host | Cheapest, least predictable. Good for restartable sim rollouts |
| [Lambda](https://lambda.ai/) | A100 40 GB ~$1.99/hr | Persistent filesystems | You want managed reliability and can justify the premium |

**Recommendation: RunPod Community Cloud with a network volume, on an RTX 4090.**
It has the persistent-storage story this project needs, the 4090's 24 GB covers
every job including the fine-tune, and the pricing is close enough to Vast.ai's
that the predictability is worth it. Use Vast.ai spot instances for bulk sim
rollouts if the bill gets uncomfortable.

Skip Colab Pro+. You sometimes get an A100, but session limits and disconnects
make multi-hour fine-tunes miserable, and there is no good persistent-storage
answer.

### Budget

Rough semester estimate, assuming you are not careless:

```text
compute   150-250 GPU-hours x $0.35-1.50/hr   =  $55 - 375
storage   ~150 GB x ~$0.07/GB-month x 5 months =  $55
                                                 ---------
                                                 $110 - 430
```

The 150-250 hour range accounts for debugging and re-runs, which always cost
more than the plan says. Split three ways this is comparable to one person's
textbook budget, and it is a defensible line item to bring to Dr. Liu or the
department rather than paying out of pocket.

### Cloud gotchas that cost money

**Storage is the sneaky cost, not compute.** Most instances wipe on
termination. Re-downloading assets every session burns hours and bandwidth.
Rent a persistent volume, and remember it bills even while the instance is off.

**Be selective about what you keep.** You do not need RoboTwin's
100,000-trajectory dataset. That is for training from scratch, and you are
fine-tuning from a released checkpoint. Store the two checkpoints, the sim
assets, and your own demonstrations. That is roughly 100-150 GB rather than
500 GB or more, which changes the storage bill by a factor of four.

**Spot for rollouts, on-demand for fine-tunes.** Spot instances are much
cheaper and get killed without warning. Sim rollouts are restartable. A 20-hour
fine-tune without checkpointing is not.

**Stop your instances.** Hourly billing runs while you sleep. This is the
single most common way students burn a budget.

**Record the rented box's specs in every result.** GPU model, driver version,
CUDA version. A baseline from an unspecified machine has exactly the same
reproducibility problem as a latency number from an unspecified power mode, and
[CONTRIBUTING.md](../../CONTRIBUTING.md) does not care that it came from the
cloud.

### Remote sim against the real board

RoboTwin's `eval_policy.sh` supports a remote policy server with a local
simulator. Combined with the course mesh VPN, that is genuinely useful:

```bash
# on the Jetson, once
sjsujetsontool tailscale up
sjsujetsontool tailscale status    # note the mesh IP
```

The rented box can then reach the Jetson directly, so the simulator runs in the
cloud while the quantized policy runs on the actual board. That closes a hole
that otherwise sits in the report, where sim accuracy and board latency come
from two different artifacts.

Keep the two measurements separate. Latency and control rate are measured
locally on the Jetson with no network in the loop. Success rate comes from the
remote sim. Network latency between the cloud and the lab would corrupt the
former and does not affect the latter.

## Getting a reference baseline, and why it goes on the workstation

You cannot measure an accuracy regression without something to regress from.
The FP16 baseline runs on the workstation (local or rented, see above), not on
the Jetson. It needs 16 GB of VRAM and the board has 8 GB shared, so there is
no version of this that runs on the Orin Nano.

```bash
# on the workstation
git clone https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA
cd Hy-Embodied-0.5-VLA
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Minimal inference call, from the model's README:

```python
config = HyVLAConfig.from_pretrained(ckpt)
policy = HyVLA.from_pretrained(ckpt, config=config).to("cuda").eval()

batch = {
    "observation.images.top_head":   img,   # (B, K, 3, 224, 224)
    "observation.images.hand_left":  img,
    "observation.images.hand_right": img,
    "observation.state":             state, # normalized dual-arm EEF
    "task": ["pick up the red block"],
}
with torch.no_grad():
    actions = policy.forward_evaluate(batch)["pred"]
```

Look closely at that batch. Three cameras. Dual-arm end-effector state. That is
the shape of the problem described below.

## The embodiment gap

The published checkpoint is dual-arm. RoboTwin 2.0's default embodiment is
`aloha-agilex`, a two-armed robot. Our physical validation platform is a
single 6-DoF SO-101 controlled in joint space with two cameras.

This is not a blocker, and it is also not something to discover in April. Plan
for it now:

1. **Simulation evaluation uses the native embodiment.** Run RoboTwin 2.0 with
   `aloha-agilex` and the RoboTwin checkpoint, unchanged. This gives clean
   accuracy-versus-configuration numbers with no embodiment confound. Every
   quantization and compilation decision gets validated here first.

2. **Physical validation needs a fine-tune onto the SO-101.** Start from the
   UMI checkpoint, collect demonstrations with the leader arm, fine-tune on the
   workstation. The delta-chunk action representation exists precisely to make
   this transfer work.

3. **Someone writes the observation adapter.** Two cameras into three expected
   slots, six joint positions into whatever the fine-tuned model consumes. This
   is a small piece of code with a large capacity to silently ruin results.
   Write it early, test it against recorded data, and put it in `src/`.

Keep the two tracks honest in the report. A number from the aloha-agilex sim
and a number from a fine-tuned SO-101 policy are different measurements and
should never be averaged together.

## RoboTwin 2.0

[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) is the simulation
benchmark. Host workstation only. It needs a real GPU and it is not something
to put on the Jetson.

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git
```

Note `--recurse-submodules`. Forgetting it produces import errors that look
like a broken install. Follow the
[official install doc](https://robotwin-platform.github.io/doc/usage/robotwin-install.html),
which the maintainers estimate at about 20 minutes.

Assets and the 100,000+ pre-collected trajectories come from the
[Hugging Face dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).
Download to the workstation's own storage. These do not belong anywhere near
this repo.

Evaluation runs through `scripts/eval_policy.sh`, which supports multi-task and
multi-GPU scheduling with a remote policy server and a local simulator.

That remote-policy-server split is worth noticing. It means the simulator can
run on the workstation while the policy runs elsewhere. Running the simulator
on the workstation and the **quantized policy on the Jetson** gives task success
rate for the exact binary and the exact board we are measuring latency on.
Setting that up is more valuable than it sounds, because otherwise sim accuracy
and board latency come from two different artifacts and the report has a hole
in it.

### Picking the task subset

Do not run the full benchmark on every sweep point. It costs hours and most of
it does not discriminate between configurations.

Choose a small fixed subset early, ideally 4 to 6 tasks, and keep it fixed for
the whole project. Pick for spread: one easy task, one requiring fine
positioning, one requiring language grounding, one with clutter. Write the
choice and the reasoning into `sim/README.md` so the report can justify it.

Changing the task set mid-project invalidates every earlier accuracy number.

## Suggested order of work

This ordering exists so that nobody is blocked and no one builds on an
unverified layer.

| Phase | Jetson track | Workstation track |
| --- | --- | --- |
| 1 | Board setup, environments, `hello_jetson.py` | RoboTwin 2.0 install, asset download |
| 2 | `hello_latency.py`, power sampling | FP16 reference baseline, task subset chosen |
| 3 | Small policy end to end (SmolVLA or similar) to exercise the pipeline | ONNX export path for Hy-VLA |
| 4 | INT8 quantization, TensorRT engine build, first real sweep | Sim evaluation of each sweep point |
| 5 | Arm bring-up, demo collection with the leader | Fine-tune UMI checkpoint onto SO-101 |
| 6 | Closed-loop runs on the physical arm at the chosen operating points | Analysis, figures, report |

Phase 3 matters more than it looks. Getting a small model running end to end on
the board (camera in, actions out, arm moving) surfaces every plumbing problem
while they are still cheap to fix. Doing it with an 8.7 GB model that does not
fit means debugging plumbing and memory at the same time.

## A note on TensorRT engines

Engines are not portable across TensorRT versions, and they are not portable
across devices. An engine built on the workstation will not load on the Jetson.
Build engines on the board they run on, and record the TensorRT version in the
result record. This is why
[requirements.txt](../../requirements.txt) tells you not to pip install
tensorrt and to pin the system version instead.

## Done when

- [ ] A workstation is secured (lab machine, SJSU cluster, teammate's desktop,
      or a rented instance) with at least 24 GB VRAM and running Linux
- [ ] Its GPU model, driver version, and CUDA version are recorded in the team
      channel, the same way the Jetson's board state is
- [ ] RoboTwin 2.0 installed there and a task runs
- [ ] Both checkpoints downloaded, paths recorded in the team channel
- [ ] FP16 reference baseline measured
- [ ] Evaluation task subset chosen and written into `sim/README.md`
- [ ] The embodiment-gap plan is a tracked issue with an owner
- [ ] If renting: a persistent volume exists and someone owns stopping idle
      instances
