# Arm bring-up

Goal: the follower arm moves under your command, the leader arm drives it, and
you have not broken anything.

Do not skip the safety section. A 30 kg·cm servo at the end of a 40 cm arm will
break a finger, and a badly configured joint limit will drive the arm into the
table until something snaps.

## Safety first, and this is not boilerplate

Before the first power-on:

1. **Clear the workspace.** A sphere of radius equal to the full arm reach,
   plus margin. Nothing fragile, no laptops, no coffee.
2. **Clamp the base to the table.** Both arms. The table clamps come in the
   kit. An unclamped arm walks itself off the desk during a fast motion.
3. **Know how to stop it.** The inline power switch on the servo rail is the
   emergency stop. Learn where it is with your eyes closed. Killing USB does
   not stop the arm, because the servos hold their last commanded position.
4. **Match each power supply to its arm.** Read the labels on your own kit.
   The reference SO-101 runs the follower at 12 V and the leader at 5 V, while
   some kits run both arms at 12 V. Where the two differ, putting the 12 V
   brick on a 5 V rail destroys those servos. Tape a label on each brick as
   well as the board, because the bricks look identical in a cable bundle.
5. **Hand on the switch for every first run** of any new script.
6. **Never leave it powered and unattended.** Repo rule, see
   [CONTRIBUTING.md](../../CONTRIBUTING.md).

Software safety, which you build once and then rely on:

- Per-joint position limits, enforced in our wrapper, not in the policy.
- A velocity cap, so a bad action cannot command a full-speed swing.
- A watchdog that commands hold-position if no new action arrives in time.

Those live in `hardware/`. Write them before the first closed-loop policy run,
not after the first crash.

## 1. Assembly

Follow the official build. There is no point duplicating it here, it changes,
and it has pictures:

- [SO-ARM100/101 repository](https://github.com/TheRobotStudio/SO-ARM100)
- [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)

Also keep your vendor's own documentation open. If you bought the Hiwonder kit,
that is the
[SO-ARM101 user manual](https://docs.hiwonder.com/projects/LeRobot/en/latest/docs/SO-ARM101%20Open-Source%206-Axis%20Robotic%20Arm%20User%20Manual.html).
Vendors differ in control board and servo firmware, and the vendor page is the
faster answer for anything board-specific.

Two things that trip up everyone:

**Servo IDs must be set one at a time, before assembly.** All Feetech servos
ship as ID 1. They sit on a shared serial bus, so two servos with the same ID
means neither is addressable. Connect exactly one servo, set its ID, label it
with tape, set it aside. Then the next. Doing this after assembly means taking
the arm apart.

**Joint zero positions matter.** Each servo has a limited travel range. If you
bolt a horn on at the wrong angle, that joint runs out of travel halfway
through its useful range and you will not find out until calibration produces
nonsense.

LeRobot's setup-motors flow walks the ID assignment:

```bash
source ~/lerobot-py312/bin/activate
lerobot-setup-motors --help
```

Budget a full day per arm if this is your first time. Skip this section
entirely if you ordered the assembled Full Kit, but still run step 3 onward.
Assembled means built, not calibrated.

### If your servos have magnetic encoders

Some kits ship STS3215 servos with 360° magnetic encoders instead of the stock
potentiometer. This is an upgrade. Potentiometer servos have a dead zone near
the travel limits, and magnetic encoders do not, so position feedback is
cleaner and repeatability improves. For a project whose output is success rate
on precise manipulation, less position noise is worth having.

The catch is that some units ship needing a firmware update, and the symptom
looks like a wiring fault. If `lerobot-find-port` sees the board but
`lerobot-setup-motors` reports motors not found on every baud rate, go to your
vendor's firmware page before opening a LeRobot issue. For Hiwonder that is the
[magnetic encoder servo firmware flashing tutorial](https://docs.hiwonder.com/projects/LeRobot/en/latest/docs/Magnetic%20Encoder%20Servo%20Firmware%20Flashing%20Tutorial.html).

## 2. Find the serial ports

Plug in the follower first. Then:

```bash
source ~/lerobot-py312/bin/activate
lerobot-find-port
```

It tells you to unplug a device and watch which port disappears. Typically you
get `/dev/ttyACM0` for the follower and `/dev/ttyACM1` for the leader.

If permission is denied:

```bash
sudo usermod -aG dialout $USER
# log out and back in, or:
newgrp dialout
```

### Make the port names stable

`/dev/ttyACM0` and `/dev/ttyACM1` swap on reboot depending on enumeration
order. If they swap mid-project, the leader arm receives follower commands and
the arm you are holding tries to move on its own. Pin them with a udev rule.

Find the serial numbers:

```bash
udevadm info -a -n /dev/ttyACM0 | grep -m1 'ATTRS{serial}'
udevadm info -a -n /dev/ttyACM1 | grep -m1 'ATTRS{serial}'
```

Then:

```bash
sudo tee /etc/udev/rules.d/99-so101.rules >/dev/null <<'RULES'
SUBSYSTEM=="tty", ATTRS{serial}=="PASTE_FOLLOWER_SERIAL", SYMLINK+="so101_follower"
SUBSYSTEM=="tty", ATTRS{serial}=="PASTE_LEADER_SERIAL",   SYMLINK+="so101_leader"
RULES
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/so101_*
```

Now use `/dev/so101_follower` everywhere instead of `/dev/ttyACM0`. Commit the
rules file into `hardware/` so teammates get the same names.

## 3. Calibrate

Calibration records each joint's mechanical range so LeRobot can convert raw
servo ticks into normalized positions. It writes a file you reuse forever.

```bash
source ~/lerobot-py312/bin/activate
lerobot-calibrate --help | grep -n "so101" -A 20
```

The flow asks you to move each joint to its extremes by hand while it records.
Do it slowly. Do not force a joint past its hard stop.

Calibrate the follower and the leader separately. They have different gear
ratios on the Pro kit, so their calibrations are not interchangeable.

Back up the calibration files into `hardware/calibration/` in this repo. They
are small JSON, they belong in git, and losing them means recalibrating.

Recalibrate whenever the arm is disassembled, a servo is replaced, or a horn
slips. A stale calibration shows up as a policy that consistently misses by a
fixed offset, which is very hard to diagnose from the policy side.

## 4. Hello world: make one joint move, safely

This is the arm equivalent of printing "hello". It connects, reads the current
pose, moves one joint a few degrees, moves it back, and disconnects.

Hand on the power switch.

```bash
source ~/lerobot-py312/bin/activate
python3 hardware/hello_arm.py --port /dev/so101_follower --dry-run
```

`--dry-run` connects and reads without commanding motion. Confirm the joint
names and positions look sane. Then, for real:

```bash
python3 hardware/hello_arm.py --port /dev/so101_follower --joint wrist_roll --degrees 8
```

Pick `wrist_roll` for the first test. It is the joint least able to hurt
anything if the sign convention is backwards.

What you should see: a small, slow motion, then a return to the starting pose,
then a printed summary of commanded versus measured positions. The gap between
those two is your servo tracking error, and it is worth noting, because it sets
a floor on how precisely any policy can be executed.

If the arm moves the wrong direction, stop, and check the calibration rather
than negating a sign in your code.

## 5. Keyboard jogging

### Get the course teleop helper

Dr. Liu's `so101_unified_teleop.py` wraps leader-follower teleop, keyboard
jogging, gamepad control, and a remote-command server in one script. The
curriculum invokes it from the home directory, so put it there:

```bash
wget https://raw.githubusercontent.com/lkk688/edgeAI/main/jetson/robotics/so101_unified_teleop.py \
  -O ~/so101_unified_teleop.py
python ~/so101_unified_teleop.py --help
```

Six modes: `leader`, `gamepad-local`, `keyboard`, `remote-server`,
`mac-ps5-client`, and `api-post`. Its docstring notes that it targets both
LeRobot 0.4.4 and 0.5.x by sticking to stable APIs, so it runs from either of
the environments in [Python environments](03-python-environments.md). The
[curriculum page](https://lkk688.github.io/edgeAI/curriculum/05d_lerobot_so101/)
documents each mode under "Unified teleop helper".

### Jog the joints

Useful for positioning the arm and for checking every joint responds. Works
over SSH:

```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=$HOME/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
python ~/so101_unified_teleop.py keyboard \
  --follower-port /dev/so101_follower --robot-id so101_follower
```

Controls:

```text
q/a  shoulder_pan     w/s  shoulder_lift    e/d  elbow_flex
r/f  wrist_flex       t/g  wrist_roll       y/h  gripper
x or ESC  exit
```

Jog each joint through a small range. All six should respond. A joint that does
nothing usually means a duplicate servo ID.

## 6. Leader-follower teleoperation

This is what the leader arm is for. You move the leader by hand, the follower
mirrors it, and LeRobot records both as a demonstration.

```bash
python ~/so101_unified_teleop.py leader \
  --follower-port /dev/so101_follower --leader-port /dev/so101_leader \
  --robot-id so101_follower --leader-id so101_leader
```

First time: hold the leader loosely, move slowly, watch the follower. If the
follower jerks or oscillates, stop. That usually means a calibration mismatch
between the two arms.

## 7. Cameras

### Which arm gets a camera

The follower gets the wrist camera. The static camera watches the workspace.
The leader gets nothing.

That is worth stating plainly because it catches people. The leader is an input
device you hold, and at inference time it is not in the loop at all. The policy
replaces it. So every camera has to sit in the same place during demonstration
collection as during policy execution, and a camera watching the leader would
produce observations the policy can never receive.

The same logic is why the static camera's pose matters so much. Tape its
position, and photograph the setup.

Two cameras is what this project needs, matching the hardware table in
[README.md](../../README.md). Note that Hy-VLA expects three streams
(`top_head`, `hand_left`, `hand_right`), so the observation adapter maps your
static camera to `top_head` and your wrist camera to one of the hand slots. See
[Model and simulation](06-model-and-simulation.md).

### Confirm the board sees them

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

You want a line showing `MJPG` in the format list. If you only see `YUYV`, the
camera will still work, at a higher CPU and USB cost.

Give the cameras stable names too, since `/dev/videoN` renumbers. Same udev
approach as the serial ports, keyed on `ATTRS{serial}` or the USB port path.

Physical setup that matters more than you would expect:

- The static camera goes on a weighted stand or a clamp, not a tripod that gets
  bumped. Mark its position on the table with tape. If the camera pose changes
  between training data and evaluation, success rate drops and it looks like a
  model problem.
- The wrist camera cable needs strain relief and enough slack for full joint
  travel. Jog every joint through its full range with the camera mounted and
  watch the cable before you ever run a policy.
- Photograph the whole setup once it is final. You will need it for the report
  and for rebuilding after someone moves the table.

## 8. Record a first dataset

Once teleoperation is smooth:

```bash
source ~/lerobot-py312/bin/activate
lerobot-record --help | grep -n "so101" -A 20
```

Start with 10 episodes of something trivial, like pushing a block ten
centimeters. The point is exercising the pipeline, not collecting good data.

Two warnings:

- Episodes go to the NVMe, never into this repo. See the `.gitignore` rules.
- The Orin Nano has no hardware H.264 encoder, so LeRobot falls back to
  software encoding. That is heavy CPU load. Do not record and benchmark at the
  same time, and expect recording to be slower than the docs suggest.

## Done when

- [ ] Both arms calibrated, calibration files committed to `hardware/calibration/`
- [ ] udev rules give stable `/dev/so101_follower` and `/dev/so101_leader`
- [ ] `hardware/hello_arm.py` moves a joint and returns it
- [ ] All six joints respond to keyboard jogging
- [ ] Leader-follower teleoperation is smooth
- [ ] Both cameras enumerate with MJPEG and have stable device names
- [ ] 10 throwaway episodes recorded end to end
- [ ] Workspace and velocity limits implemented in `hardware/`
