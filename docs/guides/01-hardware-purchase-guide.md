# Hardware purchase guide

Read this before spending anything.

## Step 0: ask Dr. Liu what the lab already owns

The course curriculum has a
[LeRobot and SO-ARM101 page](https://lkk688.github.io/edgeAI/curriculum/05d_lerobot_so101/)
with device-specific notes (serial ports, camera device numbers, a
`so101_unified_teleop.py` helper script). That means a working SO-101 setup
already exists in the lab, wired to an Orin Nano, with the hostnames and
calibration files to match.

Ask before you order:

- Is there an SO-101 leader and follower pair the team can use or borrow?
- ~~Which Jetson are we assigned?~~ Answered: `sjsujetson-36`.
- Is there a department budget or a purchase process we should go through?
- Can we get the existing calibration files for those arms? The
  `so101_unified_teleop.py` helper is already public in the course repo at
  `jetson/robotics/`, so that one you can just download.

If the lab has arms, skip to the accessories section. Everything below assumes
you are buying from scratch.

## What we are optimizing for

Three things, in priority order:

1. **LeRobot compatibility.** The SO-101 has first-class support in LeRobot
   (`so101_follower`, `so101_leader`), and the professor's curriculum is built
   around it. Deviating from it means writing our own driver, our own
   calibration, and our own teleoperation loop. That is weeks of work that has
   nothing to do with the research question.
2. **Time.** This is a two-semester capstone. Anything that trades money for
   calendar time is worth it at these price points.
3. **Cost.** Real, but third. The whole arm budget is under the price of one
   depth camera.

## Recommended bill of materials

### Core purchase

| Item | Why | Approx. cost |
| --- | --- | --- |
| Hiwonder SO-ARM101 advanced kit, assembled | Both arms, both cameras, boom stand, powered hub, both supplies, clamps, tools | $460 + tax |
| 1 TB NVMe SSD (if the Jetson does not have one) | Checkpoints, sim assets, episode recordings | $60 – $80 |
| Inline power switch for the servo rail | Kill the arm without yanking a barrel jack | $10 |
| 1× spare STS3215 servo | Students strip gears. You will want this at 11pm | $22 |

Rough total, buying everything new: **$590 – $610**.

If you buy an arm-only kit instead, add a second camera ($25 to $70), a powered
USB hub ($25), and a camera mount ($15). Compare bundles on that basis, because
the sticker prices are not like for like.

### Vendor options for the arm

| Vendor | Configuration | Price | Ships from | Notes |
| --- | --- | --- | --- | --- |
| [PartaBot](https://partabot.com/products/so-arm101) | Electronics only $329, Full Kit assembled $479 (on sale from $550) | $329 – $479 | US | Boards, servos, USB-C cables, both power supplies, webcam, fasteners. Full Kit adds the PLA+ printed parts. Assembled is only offered together with Full Kit |
| [Seeed Studio](https://www.seeedstudio.com/SO-ARM101-Low-Cost-AI-Arm-Kit-Pro-p-6427.html) | Pro servo motor kit, $249.90, plus 3D printed parts, $29.90 | ~$280 | US / international | Motors, adapter board, cables. You assemble |
| [WowRobo](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1) | Pkg 1 printed parts + 12 servos $199, Pkg 2 unassembled $259, Pkg 3 assembled $299 | $199 – $299 | China, UPS express | Leader + follower, both power supplies, 4 clamps, one camera, USB-C cables. Taxes and duties prepaid, 1-3 days prep then 2-5 days transit. Hugging Face LeRobot hardware partner |
| [Hiwonder](https://www.hiwonder.com/products/lerobot-so-101), also on Amazon | Assembled, advanced kit | ~$460 + tax | US via Amazon | Both arms, wrist and external camera, weighted boom stand, 4-port powered hub, both supplies, 4 clamps, tools. STS3215 servos, 30 kg·cm at 12 V, 360° magnetic encoders, BusLinker V3.0 board |

**Recommendation: the Hiwonder advanced kit, roughly $460 plus tax on Amazon.**
Compare bundles all-in rather than on sticker price. WowRobo's $299 becomes
about $470 once shipping, a second camera, a powered hub, and a camera mount
are added. The Hiwonder kit lands near $500 with tax and includes all of those.

For about $30 more you get domestic returns, one order instead of four, and a
weighted boom camera stand rather than a clamp. Returns are the deciding
factor. Something will arrive dead or die in week two, and an Amazon
replacement takes two days where an international RMA can cost weeks of a
two-semester schedule.

Its servos are STS3215 at 30 kg·cm and 12 V with 360° magnetic encoders, which
is an upgrade over the stock potentiometer servos. Less position noise means
less noise in the success rate. The BusLinker V3.0 board drives fine from stock
LeRobot using the normal `so101_follower` and `so101_leader` classes, and
Hiwonder publishes
[their own setup documentation](https://docs.hiwonder.com/projects/LeRobot/en/latest/docs/SO-ARM101%20Open-Source%206-Axis%20Robotic%20Arm%20User%20Manual.html).
Their Windows GUI is an optional diagnostic, not a requirement.

**Buy WowRobo Package 3, $299 plus about $100 shipping, if** you want the
cheapest path and do not mind sourcing the hub and second camera yourself.
Leader, follower, and one camera, duties prepaid, UPS express in about a week,
and a Hugging Face LeRobot hardware partner.

What is in the box, confirmed from the vendor's own Package 3 photo:

- Both arms, assembled. The leader is the one with the pistol grip, the
  follower has the gripper jaws
- Two power supplies. The control boards are labelled, 5 V DC on the leader
  and 12 V DC on the follower, which is the standard SO-101 configuration and
  confirms the follower runs the higher-torque 12 V servos
- Four table clamps, two per arm
- One camera module on a printed bracket, for the wrist
- Two USB-C cables, one per control board

Never cross the two supplies. The 12 V brick on the leader's 5 V rail destroys
its servos. The labels are printed on the boards, so check before every
power-on until it becomes habit.

[OpenELAB](https://openelab.com/products/wowrobo-robotics-so-arm101-diykit)
and RCDrone resell the same WowRobo kit if it is out of stock.

**Buy PartaBot's Full Kit assembled, $479, if** you want a confirmed 12 V
follower, a US-domestic supply chain, or WowRobo is out of stock. It also
includes both power supplies and a webcam.

PartaBot's option names are not self-explanatory. Decoded:

- **Electronics only**, $329. Servos, the two control boards, cables, both
  power supplies, and fasteners. No plastic. You print the arm bodies yourself.
- **Full Kit** adds the PLA+ printed structural parts for both arms.
- **Assembled** is offered only in combination with Full Kit, which is why it
  greys out when Electronics only is selected.

Take the Full Kit over Electronics only. Printing it yourself means 25 to 30
pieces, 1 to 1.5 kg of filament, and 40 to 60 hours of print time before the
failed prints. The servo horn mounts and bearing seats are load-bearing with
real tolerances, so it is not a forgiving print job. Seeed lists printed parts
at $29.90 per arm, so roughly $60 buys a pair if you want a price anchor.

Take assembled over unassembled, from whichever vendor you pick. Two arms is
about a day each for a first timer, so the premium works out to single-digit
dollars per hour of work that contributes nothing to the project's
contribution. It also removes the most error-prone step in the whole build,
since an assembled arm arrives with servo IDs already assigned. Confirm with
the vendor that assembled covers both the leader and the follower.

Assembled means built, not brought up. You still calibrate both arms, write the
udev rules, and verify every joint responds. See
[Arm bring-up](04-arm-bringup.md).

If the budget is hard-capped, the fallback is the Seeed Pro motor kit plus
printed parts, and you assemble. Budget a full day per arm and expect at least
one servo to need re-IDing.

### A warning about Amazon and third-party resellers

SO-101 kits appear on Amazon and on a long tail of reseller sites, often at
prices that look competitive. Read the scope before you buy, because the
listing titles routinely do not match what ships.

The recurring traps:

- **"Servo Motor Kit"** is the motors-only package. No arm bodies. One Amazon
  listing is titled "SO-ARM101 Low-Cost AI Arm Servo Motor Kit Pro for LeRobot
  (Assembled Version)", which manages to claim both at once.
- **Follower-only kits.** Plenty of listings sell a single arm. We need the
  leader too, because demonstration collection is what the SO-101 fine-tune
  depends on.
- **Missing power supplies.** The two arms need different voltages and the
  cheap listings often omit both adapters.
- **No camera**, where the specialist kits include one.

Before ordering from a general marketplace, confirm all five: leader and
follower, arm bodies included, both power supplies, camera, and the return
window. If the listing does not state something, assume it is not in the box.

The specialist vendors publish photographs of the actual box contents, which is
worth more than a few dollars of savings on a kit you will be debugging at the
servo-ID level.

One legitimate reason to prefer a marketplace anyway: university procurement.
If the department can pay through an existing Amazon Business account but not
by international card, that can unblock funding that would otherwise come out
of your own pocket. Worth asking whoever handles the budget.

### Pro or Standard?

This distinction matters on the Seeed listings. Buy Pro. The Pro follower uses
12 V ST3215-C047 servos with per-joint gear ratios (1/345, 1/191, 1/147)
instead of a uniform 7.4 V set. More holding torque means less sag under
payload, and less sag means the arm actually reaches the pose the policy
commanded. For a project whose entire output is repeatable measurements, arm
droop is a direct source of noise in the success rate.

PartaBot does not use the Pro naming. Its kit ships 6 × 12 V motors for the
follower plus 6 × 7.4 V motors for the leader, which is the configuration you
want, so there is nothing to choose there.

WowRobo's Package 3 ships the same split. Its follower control board is
labelled 12 V DC and its leader 5 V DC, so there is nothing to choose there
either.

### Cameras: what to buy and what to avoid

The model consumes 224×224 RGB. Nothing more. Buy the cheapest thing that gives
a clean color V4L2 device with MJPEG output.

If the kit came with a camera, that one goes on the wrist and you buy one more
for the third-person view.

Good choices:

- Logitech C270 (~$25 each). 720p, MJPEG, tiny, well supported by V4L2. Fine
  for both views.
- Logitech C920 / C922 (~$60 each). 1080p MJPEG, better optics and low-light
  behavior. Worth it for the third-person camera.
- Arducam 1080p USB mini modules (~$30). Small and light, which matters for
  the wrist mount.

Get MJPEG. An uncompressed YUYV stream at 1080p eats USB bandwidth and forces
the CPU to move a lot of bytes on a board where CPU time is part of what we are
measuring.

**Do not buy an Intel RealSense.** It costs more than the arm, the model only
uses RGB, and `pyrealsense2` wheels are pinned to specific glibc versions.
The professor's own board notes document the resulting version fight on
JetPack 6.

**Do not use CSI cameras (Arducam IMX219 and similar) for this project.** They
emit raw Bayer, so LeRobot cannot open them directly. Making them work needs an
`nvarguscamerasrc` GStreamer pipeline bridged through `v4l2loopback` into a
fake V4L2 device. It works, the curriculum documents it, and it is one more
moving part between your policy and your measurement. USB webcams just appear
as `/dev/videoN`.

One more reason: the Orin Nano has no hardware H.264 encoder. LeRobot encodes
recorded episodes to video, which falls back to software `x264enc`. That CPU
load shows up in your latency percentiles if you record and infer at the same
time. Keep the camera path as cheap as possible.

### What you do not need to buy

- A depth camera. The policy is RGB only.
- A force/torque sensor. Out of scope.
- A second Jetson. Sim and training live on the host workstation.
- A GPU for the workstation, if the lab already has one. Check first.

## The compatibility catch nobody mentions

The published `Hy-Embodied-0.5-VLA-RoboTwin` checkpoint expects a **dual-arm**
observation: three camera streams named `top_head`, `hand_left`, `hand_right`,
plus a dual-arm end-effector state vector. RoboTwin 2.0's default embodiment is
`aloha-agilex`, a two-armed robot.

The SO-101 is one arm, six joints, controlled in joint space.

This is not a reason to change the purchase. The model's delta-chunk action
representation is explicitly designed for cross-embodiment transfer, and the
leader arm exists so we can collect SO-101 demonstrations and fine-tune onto
our embodiment. But it does mean two things for planning:

1. The physical-arm validation needs a fine-tuning run on our own recorded
   demos. That is what the leader arm is for. Do not treat it as optional.
2. Someone has to write the observation adapter that maps our two cameras and
   six joint positions into whatever the fine-tuned policy expects. Put that on
   the 295A schedule, not 295B.

See [Model and simulation](06-model-and-simulation.md) for how this shapes the
experiment plan.

## Ordering checklist

- [ ] Confirmed with Dr. Liu what the lab already has
- [ ] Confirmed which Jetson is assigned to the team and whether it has an NVMe
- [ ] Ordered arm (leader + follower, assembled, both power supplies included)
- [ ] Confirmed the kit's servos are STS3215 or Feetech, not a proprietary bus
- [ ] Ordered a second camera and a powered USB hub, unless the kit bundles them
- [ ] Ordered an inline barrel-jack switch for the follower's 12 V rail
- [ ] Confirmed the Jetson power supply is the 5 A USB-C or official barrel jack
- [ ] Confirmed a workstation with >= 24 GB VRAM on Linux, see
      [Model and simulation](06-model-and-simulation.md)
