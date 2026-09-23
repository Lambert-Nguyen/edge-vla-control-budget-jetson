# Python environments

> **Run the audit before you build anything.** `~/lerobot-py312` and
> `~/lerobot-py310-cuda` are the paths on Dr. Liu's own board, and
> `sjsujetson-36` was provisioned by him, so they are likely already installed
> and working.
>
> ```bash
> bash hardware/audit_board.sh   # section 7 lists every venv and its packages
> ```
>
> If section 7 shows a venv with `cuda avail True`, skip this guide, use what
> is there, and pin [requirements.txt](../../requirements.txt) from it. See
> [Verify an existing setup](02a-verify-existing-setup.md) section 7.
>
> The rest of this document is for building the environments from nothing.

Goal: a Python on the Jetson where `torch.cuda.is_available()` is `True`,
LeRobot imports, and the arm driver works.

This is the step that eats the most time and produces the most confusing
errors, so read the whole page before running anything.

## Why this is hard

On a normal machine you `pip install torch` and move on. On a Jetson you
cannot. PyPI's torch wheels are built for x86 or for generic ARM CPU. The
Jetson needs a build compiled against the exact CUDA and cuDNN that shipped
with your JetPack, using NVIDIA's unified-memory allocator.

If you `pip install torch` from PyPI on a Jetson, it installs successfully,
imports successfully, and reports `torch.cuda.is_available() == False`. Nothing
warns you. This wastes an afternoon roughly once per team.

**Rule: install the Jetson torch wheel first, from NVIDIA's index, then
everything else.** If a later `pip install` pulls a different torch over it,
you are back to CPU-only and you will not be told.

## Which environment layout to use

The professor's board runs two environments, and there is a good reason for the
split. Follow the same pattern.

| Env | Python | Purpose | CUDA? |
| --- | --- | --- | --- |
| `lerobot-py312` | 3.12 | Teleop, calibration, dataset recording | No |
| `lerobot-py310-cuda` | 3.10 | Policy inference, TensorRT, benchmarks | Yes |

Why two. The newest LeRobot releases want a recent Python. The CUDA torch
wheels and the camera SDKs are built against the system Python that shipped
with JetPack. Trying to satisfy both in one environment is where people end up
compiling glibc, which you must not do on a lab board.

Practically: you drive the arm and record data in the CPU env, and you run the
model and take measurements in the CUDA env. Those are different activities
anyway.

If your board is on JetPack 7 (Ubuntu 24.04, system Python 3.12), you can use a
single environment. Check your L4T version from
[Jetson setup](02-jetson-setup.md) step 2 before picking a path below.

## Install uv

`uv` is a fast Python installer that can create environments against a specific
Python version without touching the system one. The curriculum uses it.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
~/.local/bin/uv --version
```

## Path A: JetPack 6.x (L4T R36.x, Ubuntu 22.04)

### A1. The CPU environment, for the arm

```bash
~/.local/bin/uv python install 3.12
~/.local/bin/uv venv --python 3.12 ~/lerobot-py312
~/.local/bin/uv pip install --python ~/lerobot-py312/bin/python pip
~/.local/bin/uv pip install --python ~/lerobot-py312/bin/python "lerobot[feetech]==0.5.1"

source ~/lerobot-py312/bin/activate
lerobot-info          # should print version and detected hardware
deactivate
```

`[feetech]` pulls the serial-bus servo driver for the SO-101's ST3215 motors.
Without that extra, LeRobot imports but cannot talk to the arm.

`torch.cuda.is_available()` is `False` in this environment. That is correct.
Do not try to fix it.

### A2. The CUDA environment, for inference and benchmarks

```bash
~/.local/bin/uv venv --system-site-packages --python /usr/bin/python3 ~/lerobot-py310-cuda

# torch FIRST, from the Jetson wheel index
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

`--system-site-packages` matters. TensorRT is installed as a system Debian
package by JetPack, not as a pip wheel. Without this flag your venv cannot see
`import tensorrt` and you would be tempted to `pip install tensorrt`, which
downloads a different version that will not match the runtime your engines were
built for. TensorRT engines are not portable across versions.

torch 2.8 needs cuDSS, which is not in the JetPack image. Install it and
isolate the library so it does not shadow system libraries:

```bash
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python nvidia-cudss-cu12==0.7.1.6
mkdir -p ~/lerobot-py310-cuda/cudss-lib
cp -a ~/lerobot-py310-cuda/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss*.so* \
  ~/lerobot-py310-cuda/cudss-lib/
```

Then the rest:

```bash
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python "lerobot[feetech]==0.4.4"
~/.local/bin/uv pip install --python ~/lerobot-py310-cuda/bin/python \
  numpy==1.26.4 opencv-python-headless==4.11.0.86 pygame==2.6.1
```

Pin `numpy<2`. A lot of the Jetson-built wheels are compiled against the NumPy
1.x ABI and fail at import with NumPy 2.

Use `opencv-python-headless`, not `opencv-python`. The headless build skips the
GUI dependencies, which you do not have over SSH anyway, and it avoids a Qt
conflict with the system OpenCV that JetPack installs.

### A3. Activating the CUDA environment

Every time. Put it in a shell alias or you will forget and spend an hour on a
missing `libcudss.so.0`:

```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=$HOME/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
```

Add an alias to `~/.bashrc`:

```bash
echo "alias vla='source ~/lerobot-py310-cuda/bin/activate && export LD_LIBRARY_PATH=\$HOME/lerobot-py310-cuda/cudss-lib:\$LD_LIBRARY_PATH'" >> ~/.bashrc
echo "alias arm='source ~/lerobot-py312/bin/activate'" >> ~/.bashrc
source ~/.bashrc
```

## Path B: JetPack 7.x (L4T R39.x, Ubuntu 24.04)

System Python is already 3.12 and glibc is 2.39, so one environment works:

```bash
sudo apt install -y libusb-1.0-0
python3 -m venv ~/lerobot-py312
source ~/lerobot-py312/bin/activate
pip install -U pip
pip install "lerobot[feetech]" "numpy<2"

# torch from the JetPack 7 wheel index
pip install torch torchvision \
  --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v72/pytorch/
```

For GPU work under JetPack 7 the curriculum recommends the NGC PyTorch `-igpu`
container via the provided `Dockerfile.jp7`. If pip wheels give you trouble,
take the container route rather than fighting it.

## Verify, properly

Do not trust "it installed". Run the check script from the repo:

```bash
# in the CUDA env
python3 hardware/hello_jetson.py
```

Expected output includes:

```text
[ok]   torch X.Y.Z  cuda=12.x  available=True
[ok]   device 0: Orin  (sm_87)
[ok]   tensorrt X.Y.Z
[ok]   matmul fp16 4096^3: ... ms
```

If any line reads `[FAIL]`, go to [Troubleshooting](07-troubleshooting.md)
before continuing. A half-working environment will produce plausible-looking
but wrong numbers, which is worse than a crash.

Quick manual checks if you prefer:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import tensorrt; print(tensorrt.__version__)"
python3 -c "import lerobot; print(lerobot.__version__)"
python3 -c "import numpy; print(numpy.__version__)"   # must start with 1.
```

## The model stack

Hy-Embodied-0.5-VLA wants Python 3.12, torch >= 2.4, CUDA 12.x, and >= 16 GB
of VRAM. We have 8 GB shared. The full-precision model will not run on the
Jetson, and that is the whole premise of the project.

So the order of work is:

1. Install the model repo and its checkpoints on the **host workstation**
   first. Get a reference FP16 baseline there. That is the number everything
   gets compared against.
2. On the Jetson, bring up the pipeline with something that actually fits
   (SmolVLA, or a heavily quantized Hy-VLA) so the plumbing is exercised.
3. Then work the quantization and TensorRT path until Hy-VLA fits.

Clone the model repo on the workstation:

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA
cd Hy-Embodied-0.5-VLA
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Checkpoints on Hugging Face:

- `tencent/Hy-Embodied-0.5-VLA-UMI`, the generalist pre-trained model
- `tencent/Hy-Embodied-0.5-VLA-RoboTwin`, fine-tuned for the RoboTwin 2.0
  benchmark, which is what our sim evaluation compares against

Download them to the NVMe or to shared team storage, never into the repo.
See [.gitignore](../../.gitignore).

## Pinning, and the requirements.txt situation

[requirements.txt](../../requirements.txt) is currently all placeholders with
`TODO(pin)` markers. That is deliberate. Once your environment verifies clean,
one person pins it and opens a PR:

```bash
source ~/lerobot-py310-cuda/bin/activate
pip freeze > /tmp/frozen.txt
```

Then edit `requirements.txt` to replace each `>=` with `==` from
`/tmp/frozen.txt`, and record in [README.md](../../README.md) under Setup:

- JetPack version and L4T release
- CUDA, cuDNN, TensorRT versions
- The exact torch wheel filename and the index it came from
- The LeRobot commit SHA

Do not paste a raw `pip freeze` into `requirements.txt`. It captures transitive
pins that make the file unusable on the workstation.

## Done when

- [ ] `hardware/hello_jetson.py` passes every check in the CUDA env
- [ ] `lerobot-info` runs in the CPU env
- [ ] Both `vla` and `arm` shell aliases work
- [ ] The exact versions are posted in the team channel
- [ ] Model checkpoints downloaded to the NVMe, path recorded in the channel
