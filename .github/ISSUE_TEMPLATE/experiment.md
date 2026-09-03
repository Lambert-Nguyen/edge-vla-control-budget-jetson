---
name: Experiment
about: Log one sweep configuration and its measured results
title: "[exp] <policy> @ <precision>/<resolution>/<chunk>"
labels: experiment
assignees: ''
---

<!--
Open this issue with sections 1-3 filled in BEFORE running the sweep, then
come back and fill in 4-6 with the measurements. Close it when the results
are committed to results/.

Do not paste weights, datasets, or video here. Reference paths and commit SHAs.
-->

## 1. Question

<!-- One sentence: what does this run decide? e.g. "Does INT8 at 224px hold
     30 Hz sustained control while keeping RoboTwin success within 5 pts of
     the FP16 baseline?" -->

## 2. Configuration

- **Config file:** `configs/`
- **Code commit:** <!-- git rev-parse --short HEAD -->
- **Policy / architecture:** <!-- checkpoint identifier only — no weights in repo -->
- **Precision:** <!-- fp32 / fp16 / int8 / mixed — and what stayed in higher precision -->
- **Input resolution:**
- **Action chunk length:**
- **Target control rate (Hz):**
- **Runtime:** <!-- PyTorch eager / torch.compile / ONNX Runtime / TensorRT (+ version) -->
- **Batch size:**

### Environment

- **Target:** <!-- Jetson Orin Nano 8 GB / host workstation / RoboTwin sim -->
- **JetPack / L4T:**
- **CUDA / cuDNN / TensorRT:**
- **Power mode (`nvpmodel -q`):** <!-- e.g. 15W -->
- **`jetson_clocks`:** <!-- on / off -->
- **Cooling:** <!-- fan setting; note if the board thermally throttled -->
- **Other load on the board:** <!-- should be none for a timing run -->

## 3. Method

- **Warmup iterations discarded:**
- **Measured iterations:**
- **Repeats / seeds:**
- **What was included in "end-to-end" latency:** <!-- capture → preprocess →
     forward → postprocess → command sent? Be explicit; this is the number
     that gets compared across runs. -->

## 4. Measured results

| Metric | Value | Notes |
| --- | --- | --- |
| Latency p50 (ms) | | |
| Latency p90 (ms) | | |
| Latency p99 (ms) | | |
| Sustained control rate (Hz) | | |
| Missed deadlines (%) | | at the target rate above |
| Peak unified memory (GB) | | of 8 GB |
| Avg board power (W) | | |
| Peak board power (W) | | |
| Energy per inference (J) | | |
| Sim success rate (%) | | RoboTwin 2.0, n = |
| Real success rate (%) | | SO-101, n = |
| Thermal throttling observed | | yes / no |

- **Results file(s):** `results/`
- **Sim tasks evaluated:**
- **Trials per task:**

## 5. Observations

<!-- What actually happened. Failure modes, OOM, accuracy regressions, anything
     that would make a reader distrust the numbers above. Negative results are
     worth writing down carefully — they constrain the design space too. -->

## 6. Conclusion

- **Answer to the question in section 1:**
- **Keep this operating point?** <!-- yes / no / needs another run -->
- **Follow-up experiments:** <!-- link new issues -->

## Checklist

- [ ] Config committed to `configs/`
- [ ] Results committed to `results/` as CSV/JSON, referencing the config and commit
- [ ] No weights, datasets, recordings, or video added to the repo
- [ ] Board power mode and clock state recorded above
- [ ] Warmup excluded from the reported latency percentiles
- [ ] Comparable baseline run identified <!-- link the issue it is compared against -->
