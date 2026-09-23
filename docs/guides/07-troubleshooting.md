# Troubleshooting

Grouped by where you are when it breaks. Most of these come from the course
guide's own troubleshooting tables and from the professor's board notes, which
means other students have already lost a day to each one.

## Always try this first

```bash
sjsujetsontool healthcheck
```

It reports JetPack/L4T, CUDA, cuDNN, TensorRT, memory, disk, thermals, power
rails, Docker status, and which ports are in use. Most "it stopped working"
questions are answered in that output.

## Board and boot

**The board reboots randomly, usually under load.**
Power supply. You need a 5 A USB-C adapter or the official barrel jack. A 3 A
phone charger boots the board fine and then browns out during inference. This
is the single most common hardware issue on these kits.

**The board will not boot after a power cut during an update.**
The filesystem is likely damaged. Run `sjsujetsontool stop` before cutting
power, always. Recovery means reflashing. Ask Dr. Liu before doing that to a
lab board.

**Docker daemon will not start.**
```bash
sjsujetsontool dockerfix
```

**The SSD reports less space than it should.**
Cloned-image artifact. See [Jetson setup](02-jetson-setup.md) step 6 for
`growpart` and `resize2fs`.

**`apt upgrade` broke CUDA.**
Use `sjsujetsontool sysupgrade` instead, which holds back the NVIDIA packages.
Recovery is reinstalling the matching CUDA meta-package for your JetPack, and
it is easier to ask before than after.

## Network and SSH

**`ssh sjsujetson-36.local` cannot resolve the host.**
mDNS is unreliable on university networks. Fall back to the USB-C connection:

```bash
ssh sjsujetson@192.168.55.1
```

Plug a USB-C cable from your laptop to the Jetson. This works regardless of
the network.

**Tailscale says the hostname is already taken.**
Someone cloned the image. Set a new hostname and force re-registration:

```bash
sjsujetsontool set-hostname sjsujetson-36
sudo reboot
sjsujetsontool tailscale up --force
```

**A port is already in use.**
```bash
sudo lsof -i :8080
sjsujetsontool status      # what the tool thinks is running
```

## Python and CUDA

**`torch.cuda.is_available()` is `False`.**
Ranked by likelihood:

1. You are in the wrong environment. The `lerobot-py312` CPU env reports
   `False` by design. Activate `lerobot-py310-cuda`.
2. Something installed a PyPI torch over the Jetson wheel. Check:
   ```bash
   python3 -c "import torch; print(torch.__file__, torch.version.cuda)"
   ```
   If `torch.version.cuda` is `None`, it is a CPU build. Reinstall from the
   Jetson index and then install everything else.
3. You are inside a container started without the NVIDIA runtime.

**`ImportError: libcudss.so.0: cannot open shared object file`.**
torch 2.8 needs cuDSS, which is not in the JetPack image. Install it, copy the
libraries into an isolated directory, and prepend that directory to
`LD_LIBRARY_PATH`. See [Python environments](03-python-environments.md) step
A2. Nearly every occurrence of this error is a forgotten `LD_LIBRARY_PATH`
export in a new shell. Use the `vla` alias.

**`import tensorrt` fails inside the venv.**
TensorRT is a system Debian package from JetPack, not a pip wheel. Recreate the
venv with `--system-site-packages`. Do not `pip install tensorrt`, which pulls
a different version and silently breaks engine compatibility.

**Import errors mentioning `numpy.dtype size changed` or `_ARRAY_API`.**
NumPy 2 against wheels built for the NumPy 1.x ABI.

```bash
pip install "numpy<2"
```

**`pyrealsense2` fails with `GLIBC_2.38 not found`.**
JetPack 6 ships glibc 2.35. Use the Python 3.10 environment with
`pyrealsense2==2.58.1.10581`. Do not upgrade glibc on a lab board. The
professor's notes say this explicitly and they are right.

**Something got killed with no error message.**
The OOM killer. Confirm:

```bash
dmesg | tail -40 | grep -i "killed process"
```

Free memory by dropping to console (`sudo systemctl isolate
multi-user.target`), closing other sessions, and reducing batch or resolution.
If you add swap to work around it, say so in the result record. A measurement
taken while swapping is not a measurement.

## Arm and serial

**`Permission denied: /dev/ttyACM0`.**
```bash
sudo usermod -aG dialout $USER
newgrp dialout      # or log out and back in
```

**The ports swapped and the leader is now the follower.**
Enumeration order is not stable. Write the udev rules in
[Arm bring-up](04-arm-bringup.md) step 2 and use `/dev/so101_follower` and
`/dev/so101_leader` everywhere.

**One joint does not respond to jogging.**
Almost always a duplicate servo ID. Two servos with the same ID on a shared
bus means neither is addressable. Disconnect the arm, isolate that servo, and
re-run the ID assignment.

**The follower jerks or oscillates under leader control.**
Calibration mismatch between the two arms. Recalibrate both. On the Pro kit the
leader and follower have different gear ratios, so their calibrations are not
interchangeable.

**The policy consistently misses by the same offset.**
Stale calibration. It happens after a servo is replaced, the arm is
disassembled, or a horn slips. Recalibrate and keep the file in
`hardware/calibration/`.

**The device is busy and will not connect.**
A previous run did not disconnect. Find and kill the holding process:

```bash
sudo fuser -v /dev/so101_follower
```

`hardware/hello_arm.py` disconnects in a `finally` block for this reason. Write
your own scripts the same way.

## Cameras

**LeRobot cannot open the camera.**
It needs a readable color V4L2 device. Check what you actually have:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

**The CSI camera shows black in OpenCV.**
CSI sensors (Arducam IMX219) emit raw Bayer, which OpenCV cannot decode. You
need an `nvarguscamerasrc` pipeline bridged through `v4l2loopback`. See the
[curriculum page](https://lkk688.github.io/edgeAI/curriculum/05d_lerobot_so101/).
Simpler answer: use USB webcams for this project.

**Argus `FrameConsumer` error over SSH.**
Prefix the pipeline with `env -u DISPLAY`.

**Recording is much slower than expected and the CPU is pinned.**
The Orin Nano has no hardware H.264 encoder, so LeRobot falls back to software
`x264enc`. Do not record and benchmark simultaneously. Consider recording at a
lower resolution and framerate for demonstration collection.

## Benchmarks and results

**Two teammates get different latencies from the same commit.**
Compare the `board` section of the two JSON records. It is almost always a
different `nvpmodel` mode or `jetson_clocks` state. That is exactly why the
harness captures them.

**p99 is enormous but p50 looks fine.**
Either warmup was not discarded (CUDA context creation and autotuning land in
the first few iterations) or something else was running on the board. Check
with `jtop` and re-run on an idle board. Post in the team channel before timing
runs, as CONTRIBUTING requires.

**Latency does not change when resolution changes.**
You are timing kernel launches rather than execution. CUDA calls are
asynchronous. `torch.cuda.synchronize()` has to be inside the timed region,
after the forward pass.

**A TensorRT engine will not load.**
Engines are not portable across TensorRT versions or across devices. Rebuild on
the board it runs on, and record the version in the result.

**A result in `results/` has `"dirty": true`.**
It was taken with uncommitted changes and cannot be reproduced. Commit, re-run,
replace it. Do not cite it in the report.

## When you are properly stuck

1. Run `sjsujetsontool healthcheck` and read all of it.
2. Search the
   [curriculum troubleshooting tables](https://lkk688.github.io/edgeAI/curriculum/00_sjsujetsontool_guide/).
3. Post in the team channel with the healthcheck output, the exact command, and
   the full error. Not a screenshot of the last line.
4. Then email Dr. Liu (kaikai.liu@sjsu.edu) with the same information.

Write down what fixed it. Add it to this file in your next PR.
