# Work summary, Nov 19 to Nov 21, 2026

This note lists what we did from Nov 19 to Nov 21. It also lists the tasks that are still open for James and Lam. Each item names its PR, so the details are in the PR and its records in `results/`.

All timings on the Jetson below were taken with the clocks not pinned. They are first numbers, not sweep results.

## What we did

### Hardware and live loop (James)

- PR #12 runs the Brio front camera at 60 fps in MJPG. The commit message says the loop drops to 27 Hz. That is wrong for `lerobot-record`, because it reads the latest frame. The PR body has the fix.
- PR #13 adds `benchmarks/live_loop.py`. It times the SmolVLA loop with live cameras, in sync and async mode. Async with 5 steps keeps 30 Hz with no queue gaps.
- PR #15 probes INT4-weight layers on TensorRT 10.3. Single layers build, and the engine is 3.8 times smaller. At 512 tokens it is about 12% slower than FP16.

### Safety layer (Vi)

- PR #14 adds the `so101_safe_follower` safety layer. It has 18 unit tests, and they pass.
- We set the joint limits for the team follower. The shoulder lift goes from -45 to +38 degrees. The elbow flex goes up to +60 degrees, where the arm meets the table. The step cap counts from the last goal, and the follow error is capped at 10 degrees.
- We added a gripper height check (S-11). It uses forward kinematics, calibrated on the team arm with 4.6 mm rms error.
- The E-8 bench test passes on the team follower with all 9 cases, including the new case B-9.

### Simulation and baseline (Vi and Lam)

- PR #16 installs RoboTwin 2.0 on the workstation. It pins RoboTwin at 13c3c47 and lists 5 tasks with 100 episodes each.
- PR #17 adds the remote policy link from the simulator to the Jetson. With 224 px frames over Wi-Fi, the round trip p50 is 29 ms.
- PR #18 times the released Hy-VLA checkpoint on the RTX 4090. One chunk takes 450 ms in BF16. The backbone has 3785 M parameters and the expert has 862 M.
- PR #20 is the Hy-VLA RoboTwin baseline. It reaches 473 of 500 episodes, which is 94.6%. The weakest task is `place_a2b_left` at 79%.

### Demo data (James)

- PR #21 lists the first demo datasets. `pillow_case` has 10 episodes and `pillow_case_clean` has 9. We dropped episode 1 from the clean set because it pushed the pillow. The task is "Put the teal pillow on the red phone case."

### Hy-VLA export (Lam, draft PR #22)

- We split Hy-VLA into a backbone and an action expert. The split matches the full model exactly.
- The action expert exports to ONNX. It needed 8 patches for the export only. Two of them fix wrong TensorRT results: the `rotate_half` fusion and an FP16 overflow in RMSNorm.
- The backbone exports to ONNX too. It returns K and V for 32 layers and the prefix mask.
- We tried INT4 weights first. TensorRT 10.3 on the Orin unpacks INT4 weights to FP16 in GPU memory on every call. The INT4 backbone needs 5.3 GB of scratch memory, so it cannot build or run on the board. INT4 was also the slowest. One chunk of 8 layers took 88 ms in INT4, 33 ms in FP16 and 23 ms in INT8.
- So we moved to INT8 W8A8 with `export/quantize_int8.py`. Weights have one scale per output channel. Activations have one scale per tensor, from max calibration.
- Language layers 0 and 1 stay in FP16. With them in INT8, K cosine drops from 0.993 to 0.980 and V from 0.983 to 0.958.
- The vision tower is now mostly INT8 too. The qkv, attn.proj and fc1 weights are INT8. The fc2 weights and the merger stay in FP16. With all vision weights in INT8, K cosine is only 0.966 and V is 0.933.
- `export/split_backbone.py` cuts the backbone graph at chosen language layers. We use 4 parts: vision and layers 0 to 7, then 8 to 15, 16 to 23 and 24 to 31.
- `export/make_handoff.py` makes the handoff package for the engine builder. The first package is `hyvla-int8-v1`, with 11 files and 5.6 GiB. It has the 4 backbone parts, the expert, reference cases from ONNX Runtime and a manifest with SHA-256 hashes.

### TensorRT engines on the Jetson (James, PR #19)

- PR #19 adds the engine builder and runtime in `engine/`. We tested it first with a fake handoff package.
- The action expert runs as an FP16 engine. It is within 0.624% of the reference and takes 18.1 ms per step.
- The builder now runs a backbone made of several parts as one chain. INT8 graphs build with the int8 and fp16 flags, like `trtexec --int8 --fp16`.
- Big parts only build with a RAM cap, so the host copies go to swap: `systemd-run --user --scope -p MemoryHigh=1500M -p MemorySwapMax=infinity`. The 16 GB swap file on the Jetson is now in `/etc/fstab`. All 5 engines of the package build in 52 min.
- Engines are memory-mapped when they load. Before this change, the expert did not fit next to the backbone.
- The 2% max-difference check (TR-3) does not fit INT8. One value that lands on a different INT8 step gives a large max difference. So quantized packages now pass on cosine. Each output needs cosine 0.99 or more to the reference. K, V and actions also cannot be more than 0.002 of cosine farther from FP32 than the reference is.

### First full Hy-VLA run on the Jetson

`hyvla-int8-v1` passes the engine check on the board. These are the numbers from `results/20261121-015818_orin-nano-hyvla-int8-v1-package.json`.

| Item | Result |
|---|---|
| Cosine to the package reference | K 0.9964, V 0.9901, actions 0.9967 |
| Cosine to FP32 | K 0.992, V 0.981 and 0.983, actions 0.993 and 0.995 |
| Engines | 4 backbone parts (4181 MiB) and the expert (714 MiB) |
| Backbone | p50 431 ms |
| One expert step | p50 18.1 ms |
| One chunk (backbone and 10 expert steps) | p50 614 ms |
| Lowest free RAM with all engines loaded | 439 MiB |

Most of the backbone time is in part 1, which takes 363 ms on the GPU. Part 1 runs the vision tower, the embeddings and layers 0 to 7. The vision tower is the slow piece, because it runs on 18 frames (3 cameras and 6 history frames). Parts 2 to 4 take about 23 ms each.

There are two limits to these numbers. The 2 reference cases are also the calibration cases, so the accuracy is optimistic. Also, V cosine to the reference is 0.9901, just above the 0.99 line.

## Leftover tasks

The workbook dates come from the Workbook II schedule. S7 and M4 are both on Dec 13.

### James

| Task | Workbook deadline |
|---|---|
| Time the full pipeline with pinned clocks. Run `sudo jetson_clocks` first, then `engine/build_engines.py --handoff build/handoff/hyvla-int8-v1`. The engines are cached, so it takes about 2 min. | Dec 13 (S7 and M4) |
| Live demo on the arm with the full policy. It needs a fine-tune on our demos first. Free RAM is also tight, with 439 MiB left at the lowest. | Not in the workbook |

### Lam

| Task | Workbook deadline |
|---|---|
| Real INT8 calibration on RoboTwin frames, with more cases than the 2 test cases. Then make a new package. This also gives V more room above the 0.99 line. | Dec 13 (S7) |
| L3 check: run the INT8 model in RoboTwin and compare it with the 94.6% baseline. | Dec 13 (M4 simulation numbers) |
| Update `export/NOTES.md` and the body of PR #22 with the INT4 result, the INT8 plan, the vision result, the split and the handoff package. | Not in the workbook |
| Check whether the vision tower encodes each frame on its own. If it does, we can keep the features of old frames between chunks. Then only the 3 new frames need the vision tower, and most of the part 1 time goes away. | Not in the workbook |
