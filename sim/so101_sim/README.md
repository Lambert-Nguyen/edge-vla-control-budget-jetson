# SO-101 simulator

A MuJoCo simulation of our lab task, *"Pick up the cube and put it in the box."*,
on a single SO-101 follower. It runs on **Windows, macOS and Linux** and does three
jobs:

1. **Collect training data.** A scripted expert, keyboard teleop, or the real
   SO-101 leader arm on USB records demonstrations as LeRobot datasets with the
   same features as `hardware/so101_lab.sh`: `observation.state` and `action` with
   the six `<joint>.pos` names, `observation.images.camera1` (top) and
   `camera2` (wrist) at 640×480, 30 fps, robot type `so101_follower`.
   `lerobot-train` reads them like a real dataset. Five kinds of object, scenes
   where the instruction picks one of three, and the noise of a real arm and real
   cameras are options; see [Objects and scenes](#objects-and-scenes) and
   [Real-world noise](#real-world-noise).
2. **Measure task success without the arm.** `eval` runs any LeRobot checkpoint
   (SmolVLA, π0/π0.5, ACT) in closed loop over fixed, seeded layouts and writes a
   JSON record with a Wilson 95% interval.
3. **Put the Jetson's latency into that measurement.** Chunk latency is injected
   in simulated time, taken from our harness records in `results/`. So the success
   rate is the one the policy would get at the Orin Nano's latency, measured on
   any laptop, and the observation staleness the abstract talks about is reported
   per action.

It also closes an open item from the 2026-09-28 meeting: RoboTwin 2.0 is
dual-arm and Linux-only, Isaac Sim does not run on macOS, and this is a single
SO-101 that runs everywhere.

## Start it

You need Python 3.12 the first time; the launcher builds a private environment
in `.venv/` next to it (about 2 GB, a few minutes) and reuses it afterwards.

| System | How |
| --- | --- |
| Windows 10/11 (64-bit) | Double-click `run_app.bat`. Python from [python.org](https://www.python.org/downloads/), with "Add python.exe to PATH" ticked |
| macOS 13+ on Apple Silicon | Double-click `run_app.command`. Python via `brew install python@3.12` or python.org |
| Linux | `./run_app.sh` |
| Intel Mac | Works without LeRobot: simulator and teleop yes, recording and policy eval no. PyTorch no longer ships for Intel Macs |

A browser tab opens at `http://127.0.0.1:8765/`. The simulator itself runs in the
terminal window; closing that window stops it. Saved episodes are safe whenever
it stops, because the dataset is finalized after every episode.

If macOS says `run_app.command` cannot be opened (common after downloading a
zip instead of cloning), right-click it and choose Open once, or run
`chmod +x run_app.command run_app.sh` in Terminal.

To check a computer before a session:

```bash
python -m so101_sim check          # versions, a rendered test image, 3 expert episodes
```

Every launcher option passes through, e.g. `run_app.bat --randomize full --dataset my_demos`.
With the environment active (`.venv\Scripts\activate` or `source .venv/bin/activate`),
`python -m so101_sim <command> -h` lists all options.

## The app

![app](docs/app.png)

The big view orbits with the mouse. Below it are the two images a policy sees and
the joint readout in LeRobot units.

**Control modes**

| Mode | What drives the arm |
| --- | --- |
| Keyboard | You steer the jaws: <kbd>W</kbd>/<kbd>S</kbd> forward/back, <kbd>A</kbd>/<kbd>D</kbd> left/right, <kbd>R</kbd>/<kbd>F</kbd> up/down, <kbd>T</kbd>/<kbd>G</kbd> tilt, <kbd>Q</kbd>/<kbd>E</kbd> wrist roll, <kbd>Space</kbd> opens or closes the gripper (the first press opens it fully), hold <kbd>Shift</kbd> for fine moves. Inverse kinematics turns that into joint targets. Far out or high up the wrist cannot point the jaws straight down; they then tilt as far as the joint allows, and the page shows the tilt they have |
| Sliders | One slider per joint |
| Leader arm | Your real SO-101 leader on USB, through LeRobot's own driver. Calibrate it on this computer once with `lerobot-calibrate --teleop.type=so101_leader --teleop.port=PORT --teleop.id=ID`. Ports look like `COM3` on Windows and `/dev/tty.usbmodem…` on macOS; `lerobot-find-port` tells you which |
| Expert | The scripted expert does the task on the current layout |
| Policy | A checkpoint drives the arm with a chosen latency, sync or async |

The header shows the simulator's rate. Rendering is nearly all of the cost
(about 14 ms per camera image on an M-series Mac, mostly the arm's meshes, not
the resolution), so `camera1` and `camera2` refresh in the page at 10 Hz unless
you are recording or running a policy, and the 3D view drops to 10 Hz while
recording. Recordings are in simulated time, so a slow computer only makes
keyboard, expert and policy episodes take longer. With the leader arm it changes
the data: you move in real time, so below 30 Hz your motion is stored sped up, and
the log says so.

**Real follower.** The *Real follower* panel connects a real SO-101 follower on
USB and mirrors the simulated arm on it: whatever drives the simulator (keyboard,
sliders, the leader arm, the expert or a policy) drives the real arm too. With the
leader arm as the control mode this is teleop through the simulator. The follower
first moves to the simulated pose at 1° per step (about 30°/s), and again after a
new scene; then it follows at most 3° per step, as `MAX_STEP` in the lab script
(`--follower-max-step` changes it). It waits while *as fast as possible* is on, and
a serial error stops the mirror without stopping the app. Calibrate the follower
on this computer once with `lerobot-calibrate --robot.type=so101_follower
--robot.port=PORT --robot.id=ID`. Stopping the mirror turns its torque off, as
LeRobot does, so the arm goes limp: support it or bring it to rest first. The real
arm does not feed back into the simulator, so the cube it would push stays in the
simulator only. Not yet tried on the real arm.

**Recording** works like `lerobot-record`: *Record episode* starts from a fresh
layout, <kbd>→</kbd> saves, <kbd>←</kbd> discards, and an episode saves itself at
the time limit (20 s). *Record expert demos* records N successful expert demos in
a row; tick *as fast as possible* to skip real-time pacing. Datasets go to
`~/.cache/huggingface/lerobot/local/<name>` (on Windows
`%USERPROFILE%\.cache\huggingface\lerobot\local\<name>`), as on the Jetson.

## From demos to a success rate

**1. Generate demos.** Headless and much faster than real time on a laptop GPU:

```bash
python -m so101_sim generate --name sim_cube_box --episodes 200 --randomize full

# every object, all three scenes, with the noise of the real rig
python -m so101_sim generate --name sim_multi_object_noisy --episodes 300 --randomize full \
  --objects all --scene mixed --noise real
```

Layout seeds start at 0 and each episode's seed and start pose are logged in
`meta/so101_sim_episodes.jsonl`. Only successful episodes are kept. The expert
succeeded on all 3,100 layouts checked from the home pose (seeds 0–1399 at
`light`, 0–1399 and 100000–100299 at `full`), and on all 800 checked from random
start poses (seeds 0–299 at `light`, 0–299 and 100000–100199 at `full`). Over a
far box, or above a far cube, it tilts the jaws a little off vertical, because
pointing them straight down there needs more `wrist_flex` than the joint has.

**2. Train on the workstation.** Copy the dataset folder to the RTX 4090 machine,
then fine-tune as with real data, for example SmolVLA:

```bash
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/sim_cube_box --dataset.root=/path/to/sim_cube_box \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=64 --steps=20000 --output_dir=outputs/train/smolvla_sim
```

The camera keys are already `camera1` and `camera2`, the names `smolvla_base`
expects, so no renaming is needed.

LeRobot decodes the videos with torchcodec, which needs FFmpeg's shared libraries
on the machine. If `lerobot-train` stops with `Could not load libtorchcodec` (seen
on a Mac without FFmpeg), add `--dataset.video_backend=pyav`.

**3. Evaluate with the Jetson's latency.**

```bash
# p50 and spread of chunk latency from a harness record on the board
python -m so101_sim eval --policy outputs/train/smolvla_sim/checkpoints/last/pretrained_model \
  --episodes 50 --num-steps 5 \
  --latency-from ../../results/20261026-223659_maxn-pinned-smolvla-1cam-5step.json

# or sweep latency to get the success-versus-latency curve
python -m so101_sim eval --policy … --episodes 50 --latency-ms 0,250,500,750,1000,1250
```

Records go to the repo's `results/` as `<time>_sim-eval_<name>_<mode>_<latency>ms.json`,
where `<name>` is the training run and checkpoint (`smolvla_sim-last`), with the success count and Wilson interval, per-episode results, the mean and
p95 observation staleness, the fraction of stalled steps and the git commit.
Evaluation layouts start at seed 100000, so they never repeat a training layout.

`--policy expert` runs the scripted expert through the same latency machinery.
It is a plumbing check, not a result: the expert plays an open-loop plan, so
staleness cannot hurt it the way it hurts a policy that reacts to the camera.

### Sync and async, and what latency costs

| | sync (what `lerobot-record` does now) | async (LeRobot async inference) |
| --- | --- | --- |
| When a chunk is planned | when the queue is empty | when `--threshold` × chunk is left |
| While it is computed | the arm holds its last target | the old chunk keeps playing |
| When it arrives | played in full | actions for steps already past are dropped |
| Cost of latency | pauses, so episodes take longer | stale observations; stalls if the queue runs dry |

As a first look with the expert stand-in over 12 layouts: at the pinned
SmolVLA one-camera, five-step latency (488 ms) async succeeds 12/12 with 7%
stalled steps. At the unpinned two-camera, ten-step latency (1,129 ms) and
threshold 0.5, the queue runs dry 58% of the time and async drops to 10/12,
while sync stays at 12/12 but takes 30% longer. Raising `--threshold` above
latency ÷ chunk duration (here 1.13 s ÷ 1.67 s ≈ 0.68) is the obvious knob.

## Objects and scenes

`--objects` chooses what to pick up, one drawn per episode: `cube` (the default),
`block`, `cylinder`, `ball`, `marker`, a comma list, or `all`. At `light` and
`full` each has a random size. The instruction names it: *"Pick up the ball and
put it in the box."*

| Object | Shape | Size | Grasped |
| --- | --- | --- | --- |
| cube | box | 2.4–3.6 cm sides | across two faces |
| block | box lying flat | 4.4–6 cm long, 2.4–3 cm wide and high | across the short side |
| cylinder | standing up | 2.4–3.4 cm across, 3.6–6 cm tall | any direction |
| ball | sphere | 2.8–3.6 cm across | any direction |
| marker | capsule lying flat | 2–2.6 cm across, 7.2–9.4 cm long | across the body |

Mass comes from the volume at 1100 kg/m³, so the 3 cm cube weighs 30 g as before.
Long objects lie within 45° of across the arm's reach: pointing straight at the
arm near its base, the grasp needs the wrist rolled 90°, and the wrist camera
mount then hits the shoulder.

`--scene` puts more than one object on the table, so the policy has to read the
instruction:

| Scene | On the table | Instruction |
| --- | --- | --- |
| `single` (default) | one object | *"Pick up the cylinder…"* |
| `colors` | three of one kind, each a different colour | *"Pick up the red cube…"* |
| `kinds` | three different kinds, all one colour | *"Pick up the ball…"* |
| `mixed` | one of the three above, drawn per episode | either |

Colours are red, green, blue, yellow, purple and white. The other objects stand
at least 3.5 cm apart (beyond their size) so the open jaws come down clear of
them; when three long markers do not fit, one is left out (10 in 60 `colors`
episodes). `eval` records count episodes where a wrong object ended up in the
box.

The scripted expert succeeded on every episode checked: 40 per object at `full`
without noise and at `real` noise, and 60 per scene at `full`, with and without
noise. At `heavy` noise the ball drops to 36 of 40.

## Real-world noise

`--noise` adds what a real SO-101 and its USB cameras do to a perfect simulator:
`none` (the default), `real` (our estimate of the lab rig), `heavy` (twice
that), or any scale. Each episode draws its own errors from its seed, so a seed
gives the same layout at any noise and the same noise at the same scale.

| | At `real` |
| --- | --- |
| Joint readings | STS3215 encoder ticks (0.088°) plus 0.15° noise; gripper 0.3 units |
| Calibration | each joint's zero off by up to 2° (gripper 2 units): the arm moves to command + offset and reports position − offset |
| Backlash | 0.8° of play in each arm joint's gears |
| Servo gains | each joint's position gain off by up to 15% |
| Command jitter | 0.1° per step |
| Exposure, white balance | up to 15% and 6% per channel per episode, plus 2% flicker per frame |
| Image noise and compression | 3 levels of 255 per pixel, then JPEG at quality 70–95, as an MJPEG webcam sends |
| Camera delay | frames arrive 1–2 steps (33–67 ms) late; 2% of frames are dropped and the previous one repeats |
| Wrist camera mount | off by up to 2 mm and 2° |

The expert and keyboard teleop correct for the calibration offset, as a person
steering by eye would, so expert demos still succeed while the recorded
observations carry all of the noise. The numbers are estimates; the first
session on the real arm should check the backlash and the camera delay.

To measure how a policy copes, sweep noise as you would latency:

```bash
python -m so101_sim eval --policy … --episodes 50 --noise none,real,heavy --latency-ms 0,488
```

## Shared datasets

Datasets and checkpoints are too big for git (several files are over GitHub's
100 MB limit), so they live on Hugging Face. Each entry names the commit of this
repo that generated it; check that commit out to regenerate it exactly.

| Dataset | Made with | Contents |
| --- | --- | --- |
| [`duylam1407/so101_sim_cube_box`](https://huggingface.co/datasets/duylam1407/so101_sim_cube_box) (public) | commit `6e41556`, `python -m so101_sim generate --name sim_cube_box --episodes 200 --randomize full` | 200 expert episodes, 55,067 frames, 30 fps, seeds 0–199, default start spread; one 3 cm cube, no noise, camera1 in the old `front` position (regenerate with `--camera1 front` at later commits); 200/200 expert successes; 662 MB. Hub revision `d926fb5`, tag `v3.0` |
| [`duylam1407/so101_sim_multi_object_noisy`](https://huggingface.co/datasets/duylam1407/so101_sim_multi_object_noisy) (public) | commit `8767d5b`, `python -m so101_sim generate --name sim_multi_object_noisy --episodes 300 --randomize full --objects all --scene mixed --noise real` | 300 expert episodes, 82,408 frames, 30 fps, seeds 0–299; `top` camera1; all five objects (52 cube, 57 block, 67 cylinder, 60 ball, 64 marker); scenes 89 `single`, 110 `colors`, 101 `kinds`, 33 distinct instructions; `real` noise in every episode; 300/300 expert successes; 1.0 GB. Each episode's scene and noise draw are in `meta/so101_sim_episodes.jsonl`. Hub revision `741c9dd`, tag `v3.0` |

Train new policies on `so101_sim_multi_object_noisy`: it has the camera position,
the noise and the language the real task will have. `so101_sim_cube_box` stays
for comparison with the first runs.

`lerobot-train` downloads a dataset by its id, so no copying or login is needed:

```bash
lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=duylam1407/so101_sim_multi_object_noisy \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=64 --steps=20000 --output_dir=outputs/train/smolvla_sim
```

Training runs so far:

| Run | Where | Settings | Status |
| --- | --- | --- | --- |
| `act_sim` | M1 Pro laptop, MPS | `so101_sim_cube_box`; ACT, `chunk_size` and `n_action_steps` 50, batch 8, 20k steps (2.9 epochs), checkpoint every 5k, `--dataset.video_backend=pyav` | done 2026-11-20, 5.7 h, final loss 0.096. Last checkpoint, 20 episodes each, async at 0 ms, `--camera1 front`: 6/20 without noise ([record](../../results/20261120-103142_sim-eval_act_sim-last_async_0ms.json)), 5/20 at `real` ([record](../../results/20261120-103723_sim-eval_act_sim-last_async_0ms_noise-real.json)), 2/20 at `heavy` ([record](../../results/20261120-103723_sim-eval_act_sim-last_async_0ms_noise-heavy.json)). Undertrained; the intervals overlap, so the noise effect is not yet measured. Checkpoints stay on the laptop |
| `smolvla_sim` | RTX 4090 | the command above, on `so101_sim_multi_object_noisy` | not started |

## Matching the real arm

**Units.** LeRobot 0.4.4 and 0.5.1 both default the SO-101 to `use_degrees=True`:
five joints in degrees from the middle of each calibrated range, gripper 0–100.
The Menagerie model puts its joint zeros at mid-range too, so arm joints map as
plain degrees, and the gripper maps its simulated range onto 0–100 with 0
closed. Both are assumptions until checked once against the real follower: put it
in a pose, compare its readings with the simulator's joint readout in the same
pose, and if a joint is mirrored or offset, write a joint map rather than editing
code:

```json
{"sign": [1, 1, 1, 1, 1], "offset_deg": [0, 0, 0, 0, 0], "gripper_closed": 0, "gripper_open": 100}
```

then pass `--joint-map my_arm.json` to any command.

**The team follower's map** is in
[`joint_maps/team_follower.json`](joint_maps/team_follower.json). It comes from
the safety layer's table calibration (2026-11-20, 13 jaw-tip touches, 4.6 mm
rms), which fits the same `so101.xml` to the real arm: there
model angle = LeRobot degrees + offset, so the map's `offset_deg` is minus those
offsets. `shoulder_lift` is off by 10.3°, `wrist_flex` by 6.4° and `elbow_flex`
by 0.9°. The fit cannot see `shoulder_pan` or `wrist_roll`, and it assumes no
joint is mirrored; check both with the pose comparison above. The datasets so
far use no map, so their states and actions are in the simulator's degrees:
fine for training and evaluating in simulation, but a policy taken from them to
the team arm would aim 10° off on the shoulder. Use `--joint-map
joint_maps/team_follower.json` for new datasets and for the app's *Real
follower* mirror.

**The team follower's safety limits** clip a few simulated commands. With the
map, `so101_sim_multi_object_noisy` asks for `shoulder_lift` above the arm's
+38° in 0.8% of frames (32 of 300 episodes, mostly while reaching the box) and
`elbow_flex` above +60° in 0.1% (4 episodes). The closed gripper is commanded
to 0, below the layer's floor of 2, in about a third of frames, so the real
grasp closes on 2. None of this stops an episode; the layer clips and logs it.

**Cameras.** `camera1` is a top view, because in `lerobot/smolvla_base`'s
training data camera1 is a top view. It sits 70 cm above the table and 30 cm in
front of the workspace (at x = 0.50 m), aimed back at the table 20 cm in front of
the base, so the arm is at the top of the image, with a 52° vertical field of
view (a Brio at 78° diagonal, cropped to 4:3). Straight down over the workspace
the raised arm hides the objects. `--camera1 front` gives the view datasets
before 2026-11-20 used: 62 cm in front of the base, 42 cm up. Measure the real
mount in ENG276 and pass `--camera1-pos X,Y,Z --camera1-target X,Y,Z` (metres;
x forward from the arm base, y to its left, z up). `camera2` is the wrist camera
on Menagerie's camera mount; check its view against the real one too.

**Episodes** start near a raised home pose with the jaws about 16 cm over the
table (`--home` changes it). The real arm's folded rest pose puts the jaws inside
the area where cubes spawn. On the real arm an episode starts wherever the arm
is, so at `light` and `full` each episode's start pose strays from home by up to
30° on `shoulder_pan` and `wrist_roll`, 15–20° on the other joints and 20 on the
gripper, redrawn if the jaws come within 10 cm of the table or the arm touches
the cube or box. `--start-spread 2` doubles that (the expert still succeeded on
150 of 150 layouts), and `--start-spread 0` or `--randomize none` starts every
episode at home. A seed gives the same layout at any spread.

**Servo limits.** The gripper gets half the arm servos' torque, because
LeRobot's follower writes `Max_Torque_Limit = 500` (50%) to the gripper.
`--max-step-deg 3` mirrors the lab script's `MAX_STEP`.

## Limits worth knowing

- **Sim-to-real.** The renders are clean CAD colours on a plain table. A policy
  trained only on simulated images will not transfer to the real cameras as is.
  Use simulated demos for the simulated success budget, and for real-arm runs
  train on real demos or mix the two.
- **Servo gains** are Menagerie's, not identified against our servos. The source
  models disagree by 56× on the position gain. If simulated success matters for
  the paper, fit them to a few recorded real trajectories first.
- **SmolVLA in the loop** goes through the same `predict_action_chunk` path that
  LeRobot's async policy server uses, and was exercised here with an ACT
  checkpoint. SmolVLA itself has not been run in this simulator yet.

## Layout

```text
so101_sim/
  env.py          scene, physics, cameras, randomization, success check
  objects.py      the objects, their sizes and grasps, colours, multi-object scenes
  noise.py        sensor, servo and camera noise of a real arm
  units.py        sim radians <-> LeRobot degrees and gripper 0-100
  ik.py           5-DoF inverse kinematics (position + pitch, roll set directly)
  expert.py       scripted pick-and-place
  recorder.py     LeRobot dataset writer
  generate.py     headless expert demo generation
  policy_eval.py  latency model, chunk executor, closed-loop evaluation
  leader.py       real SO-101 leader arm over USB
  follower.py     mirror the simulated arm on a real SO-101 follower over USB
  app/            local web server and the browser page
  assets/so101/   MuJoCo Menagerie robotstudio_so101 (Apache-2.0) + the task scene
tests/            pytest; CI runs them on Windows, macOS and Linux
```

`python -m pytest` runs the tests. `.github/workflows/so101-sim.yml` runs them,
the launchers and a rendering check on all three systems.

## Credits

The arm model is [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s
`robotstudio_so101` (commit `370c439`, Apache-2.0, licence in `assets/so101/LICENSE`),
derived from The Robot Studio's SO-ARM100 `so101_new_calib` model.
