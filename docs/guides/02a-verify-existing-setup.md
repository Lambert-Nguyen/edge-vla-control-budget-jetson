# Verify an existing setup

**Start here.** This is the Jetson guide for our project.

Dr. Liu provisioned `sjsujetson-36` before handing it over, and said everything
is already set up. Confirmed so far:

- JetPack, CUDA, cuDNN and TensorRT installed
- Hostname set to `sjsujetson-36`
- `sjsujetsontool` installed, and updated once on first login

So the job is not installing things. It is finding out what is here, writing it
down, and leaving it alone. A provisioned lab board may also carry another
student's Python environments, a Tailscale registration, container images, and
calibration files. Reinstalling over any of that destroys work and costs days.

[Jetson setup from scratch](02-jetson-setup.md) is kept for the case where a
board has to be rebuilt. You should not need it.

**Nothing in this document modifies the board.** Every command reads.

## The three commands that would break your board

Guide 02 is written for a fresh board. These are safe there and wrong here.

| Command | What it would do |
| --- | --- |
| Flashing the NVMe | Erases everything. The board is gone, along with whatever was on it |
| `sjsujetsontool set-hostname` | Renames the board, breaking its Tailscale registration and any hostname teammates use |
| `sjsujetsontool update` | Pulls a new container image over a working one, and can change library versions your measurements depend on |

Two more to hold off on until you know why you need them:

- `sjsujetsontool sysupgrade` changes packages. Fine eventually, pointless on
  day one, and it can move a version you are about to benchmark against.
- `growpart` and `resize2fs` are only for a disk that reports less space than
  it physically has. Do not run them speculatively.

Also relevant, from [Python environments](03-python-environments.md): that
guide creates `~/lerobot-py312` and `~/lerobot-py310-cuda`. Those are the exact
paths on Dr. Liu's own board. If they already exist here, creating them again
throws away a working install. The audit checks for them.

## Run the audit

First get this repo onto the board, so you can run the scripts there:

```bash
git clone https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson.git ~/edge-vla
cd ~/edge-vla
bash hardware/audit_board.sh | tee ~/board-audit-$(date +%F).txt
```

Use the HTTPS URL, not the `git@github.com:` SSH one. SSH only works from a
machine whose key GitHub recognises, and the Jetson is a different machine from
your laptop.

If the repo is private, HTTPS will prompt for credentials and GitHub will not
accept your account password. Either authenticate with the GitHub CLI:

```bash
sudo apt install gh
gh auth login
gh repo clone Lambert-Nguyen/edge-vla-control-budget-jetson ~/edge-vla
```

or give the board its own SSH key, which is worth doing anyway since you will
push results from it later:

```bash
ssh-keygen -t ed25519 -C "sjsujetson-36"
cat ~/.ssh/id_ed25519.pub     # paste into github.com/settings/keys
```

It prints eleven sections covering identity, CUDA, storage, power mode,
sjsujetsontool, Docker, Python environments, project files already present,
network, attached hardware, and monitoring tools.

Post the output in the team channel. It is the board's starting state, and you
will want it later when a measurement looks wrong.

## Interpreting each section

### 1. Identity and OS

Looking for L4T R36.x (JetPack 6) or R39.x (JetPack 7).

| What you see | Do |
| --- | --- |
| L4T R36.4 or newer, or R39.x | Nothing. This is what you want |
| L4T R35.x or older (JetPack 5) | Ask Dr. Liu before anything. Do not flash a lab board on your own judgment |
| No `/etc/nv_tegra_release` | You are not on the Jetson. Check which machine you are logged into |

Write the L4T and JetPack versions into [README.md](../../README.md) under
Setup. Every measurement is quoted against them.

### 2. CUDA, cuDNN, TensorRT

Record all three. TensorRT engines are not portable across versions, so the
version here is part of every result.

If `nvcc` is missing but `/usr/local/cuda` exists, CUDA is installed and just
not on your `PATH`. That is a shell setting, not a missing package:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
```

### 3. Memory, swap, storage

Expect roughly 7.4 GiB total on an 8 GB module, the rest being carve-outs.

Check that `/` sits on the NVMe rather than a microSD. If `lsblk` shows root on
`mmcblk0`, tell the team, because disk speed will show up in your numbers.

If swap is on an NVMe file, note it. It does not need changing, but a benchmark
taken while swapping is not a valid measurement, and you want to know the
option exists.

### 4. Power mode and clocks

`sjsujetson-36` offers three modes and arrived in mode 2:

| Mode | Name | What it is |
| --- | --- | --- |
| 0 | 15W | The original pre-Super Orin Nano envelope. Our tightest power budget |
| 1 | 25W | The module TDP, capped. Deterministic |
| 2 | MAXN SUPER | Uncapped. Highest clocks, throttles back when it exceeds TDP |

**Leave it in mode 2 for now.** Changing the power mode is a per-run decision
you record in each result, not a permanent setting to fix on day one. See
[First benchmark](05-first-benchmark.md) for which mode to take the main sweep
in and why.

If the audit reports a different mode than 2, someone changed it. Ask before
changing it back, because their reason may still apply.

### 5. sjsujetsontool

Already installed on our board. Run the read-only checks and keep the output:

```bash
sjsujetsontool version
sjsujetsontool healthcheck
sjsujetsontool status
```

`healthcheck` prints hardware model, kernel, JetPack and L4T, CUDA, cuDNN,
TensorRT, memory, disk, thermal zones, power rails, Docker status, and which
service ports are live. That block is the board state you cite in every result
for the rest of the project, so post it in the team channel.

On `update`: it refreshes the script and the container image. Running it once
on a new board is reasonable and ours has been updated. Do not run it again
casually, because a new container image can move library versions your
measurements were taken against. Whoever ran it should post the resulting
`sjsujetsontool version` output so the whole team benchmarks against the same
state.

### 6. Docker

Existing images and containers are someone's work. Leave them. `docker ps` and
`docker images` are read-only.

### 7. Python environments

The section that matters most.

| What you see | Do |
| --- | --- |
| `~/lerobot-py310-cuda` exists with torch and `cuda avail True` | Use it. Skip guide 03 path A2 entirely |
| `~/lerobot-py312` exists with lerobot | Use it. Skip guide 03 path A1 |
| The venvs exist but torch reports `cuda avail False` in the CUDA one | Do not rebuild yet. Check `LD_LIBRARY_PATH` first, see [Troubleshooting](07-troubleshooting.md) |
| No venvs in `$HOME` | Follow [Python environments](03-python-environments.md) normally |

If the environments exist, record their exact package versions now and pin
[requirements.txt](../../requirements.txt) from them. Someone already solved
the hard part of this project's setup, and the versions they landed on are the
ones that work on this board.

Never `pip install` into an inherited environment without telling the team. A
stray upgrade of torch or numpy can break it for everyone.

### 8. Project files already on the board

`so101_unified_teleop.py`, a `~/.lerobot` directory, or calibration JSON all
mean the board has driven an arm before. Calibration files especially are worth
having, since they save you a recalibration. Copy them into
`hardware/calibration/` in this repo.

A populated `~/.cache/huggingface` may already hold checkpoints. Check before
re-downloading tens of gigabytes.

### 9. Network and hostname

The audit prints a verdict line. Read it this way:

Our board is `sjsujetson-36`, already named. The audit should print
`named, leave it alone`. If it prints anything else, stop and ask the team
before changing it.

| Verdict | Meaning |
| --- | --- |
| `named, leave it alone` | Someone set it. Do not run `set-hostname` |
| `DEFAULT name, never configured` | Still `ubuntu`, `nvidia-desktop`, `tegra-ubuntu` or similar. Naming it is reasonable, but ask the team first |
| `/etc/hostname differs` | A rename is pending a reboot. Find out who did it before rebooting |
| `/etc/hosts entry: MISSING` | Causes slow `sudo` and flaky mDNS. Worth fixing, and it is a one-line change |

The real test is whether the name resolves, and it has to be run from your
laptop rather than from the board:

```bash
ping -c1 sjsujetson-36.local
ssh sjsujetson@sjsujetson-36.local
```

If that works, the hostname is set up in the only sense that matters.

The audit also prints `/etc/machine-id`. If two lab boards were cloned from one
SSD they share an id, which causes DHCP lease and Tailscale conflicts that
present as random network failures. Dr. Liu's guide flags this under its safety
reminders. Raise it with him rather than regenerating the id yourself, since
that drops the board off the mesh.

If Tailscale is already up and registered, leave it. Renaming the host is what
breaks it. Note the hostname and mesh IP, since that is how you reach the board
later and how a rented cloud box reaches it for remote sim evaluation.

### 10. Attached hardware

Empty right now, since the arm has not arrived. Run the audit again when it
does, and compare. This section is how you will confirm the arm and cameras
enumerated.

Existing udev rules for `so101` mean someone already made the port names
stable. Read them before writing your own.

### 11. Monitoring tools

`tegrastats` ships with JetPack. `jtop` comes from `jetson-stats` and may not
be installed. Installing it is additive and low risk, but it wants sudo and a
service restart, so mention it in the team channel first.

## What to actually do today

1. Run the audit and post the output.
   ```bash
   bash hardware/audit_board.sh | tee ~/board-audit-$(date +%F).txt
   ```
2. Run `sjsujetsontool healthcheck` and `sjsujetsontool version`, and post both.
3. Set up SSH keys from your laptop. Additive, breaks nothing:
   ```bash
   ssh-copy-id sjsujetson@sjsujetson-36.local
   ```
4. Run `python3 hardware/hello_jetson.py` inside whichever environment the
   audit found in section 7. It reads and benchmarks, and changes nothing.
5. Fill in the version table in [README.md](../../README.md) from the audit.
6. Grab the course teleop helper so it is ready when the arm lands:
   ```bash
   wget https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/robotics/so101_unified_teleop.py \
     -O ~/so101_unified_teleop.py
   ```

Then go to [First benchmark](05-first-benchmark.md). If the environments are
already there, you can take a real measurement today without installing
anything.

## The only parts of guide 2r you may still need

Use the audit to decide. On a fully provisioned board the list is short:

- SSH keys from your laptop, if not set up yet
- Power mode selection, which is a per-run decision you record in each result
  anyway, not a one-time setup step
- `jetson-stats` for power measurement, if the audit says `jtop` is absent

Skip flashing, hostname changes, and further container updates unless something
is actually broken and the team agrees first.
