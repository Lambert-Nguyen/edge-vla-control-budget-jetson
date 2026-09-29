# First benchmark

Goal: one real measurement, written to `results/`, that a teammate can
reproduce from the commit and the board state recorded inside it.

Do this before touching the real model. If the harness is wrong, every number
after it is wrong, and you will not find out until the report.

## What we are actually measuring

Four quantities, for every configuration in the sweep:

| Quantity | Why it matters | Where it comes from |
| --- | --- | --- |
| End-to-end action latency | Sets how fast the loop can close | `time.perf_counter` around a synchronized forward pass |
| Sustained control rate | Whether the arm ever stalls waiting | action chunk length divided by latency |
| Peak unified memory | Whether the config fits in 8 GB at all | `torch.cuda.max_memory_allocated` plus board-level RSS |
| Board power | The other edge budget | INA3221 rails via `tegrastats` or `jtop` |

And one more that lives in [simulation](06-model-and-simulation.md): task
success rate, so that a speedup is always reported next to its accuracy cost.

## The control-rate idea, since it is the part people get wrong

A tabletop arm wants a new command roughly every 33 ms (30 Hz). A VLA forward
pass takes far longer than that. The fix is action chunking: each inference
returns `k` future actions, and the arm plays them out while the next inference
runs.

So the loop closes at `k / latency` Hz, not `1 / latency` Hz. With a chunk of
8 and a 200 ms inference, that is 40 Hz of commands, which clears a 30 Hz
budget.

Two consequences worth keeping in your head:

- The number that matters is **p99 latency, not mean**. The mean tells you the
  typical case. A single 600 ms outlier drains the chunk buffer and the arm
  stops mid-motion. Our harness reports p50, p90, and p99, and judges the
  budget on p99.
- Longer chunks buy control rate but cost accuracy. Each chunk is computed from
  an observation that is one inference cycle old, so by the last action in the
  chunk the world has moved on. That trade is one axis of the sweep.

## Run it

On the Jetson, in the CUDA environment:

```bash
source ~/lerobot-py310-cuda/bin/activate
export LD_LIBRARY_PATH=$HOME/lerobot-py310-cuda/cudss-lib:$LD_LIBRARY_PATH
cd ~/edge-vla

python3 benchmarks/hello_latency.py
```

Output looks like:

```text
====================================================================
dummy-fp16   224px   chunk=8
====================================================================
  p50    4.21 ms    p90    4.55 ms    p99    6.10 ms
  mean   4.30 ms    sd     0.34 ms    max    6.80 ms
  peak gpu memory 0.05 GiB allocated, 0.08 GiB reserved
  control rate    1899.2 Hz typical, 1311.5 Hz worst case
  budget 30 Hz: MET

wrote results/20260921-143002_dummy-fp16.json
```

Those numbers are meaningless as science. The model is a four-layer convnet
standing in for a 4B-parameter VLA. What matters is that the record came out
complete.

## Read the JSON

```bash
python3 -m json.tool results/*_dummy-fp16.json
```

Every record carries:

- `config` holds resolution, chunk, precision, iteration counts, target rate
- `latency` holds n, mean, stdev, min, p50, p90, p99, max
- `memory` holds peak allocated and reserved
- `control_rate` holds sustained and worst case, and whether the budget was met
- `board` holds model, L4T, nvpmodel output, jetson_clocks GPU line
- `git` holds commit, branch, and whether the tree was dirty

If `git.dirty` is `true`, the run is exploratory. Commit first, then re-run,
before any number goes near the report. [CONTRIBUTING.md](../../CONTRIBUTING.md)
is explicit about this and the harness warns you.

## Prove the harness is honest

Three checks. Do them once, now, and you can trust the harness for the rest of
the project.

**1. Warmup actually matters.**

```bash
python3 benchmarks/hello_latency.py --warmup 0 --iters 50 --label no-warmup
```

p99 should be noticeably worse. The first calls include CUDA context creation
and kernel autotuning. If discarding warmup changes nothing, your synchronize
calls are in the wrong place.

**2. Resolution moves latency in the expected direction.**

```bash
for r in 112 224 336; do
  python3 benchmarks/hello_latency.py --resolution $r --label "res-$r"
done
```

Latency should rise with resolution. If it does not, you are timing queueing
rather than execution, which means a missing `torch.cuda.synchronize()`.

With clocks unpinned there is a second cause, and it is the one we hit on
2026-09-28. The GPU governor raises the clock as the work grows. p50 stayed at
2.1 to 2.6 ms from 112 to 672 px while `VDD_IN` climbed from 6.5 W to 14.9 W and
the GPU clock went from 306-408 MHz to 1020 MHz. Only once the clock saturated
did latency follow the pixels: 13.8 ms at 1344 px and 30.7 ms at 2048 px, a
0.45 ratio for 0.43 of the pixels. The harness was timing execution
correctly, and check 1 confirms the synchronize is in place. The lesson:
take this check with `jetson_clocks` on, and never read a latency without the
power and clock columns beside it. The records are in `results/`, labelled
`res-*`.

**3. The power mode changes the answer.**

```bash
sudo jetson_clocks --store            # save the current, unpinned clock state first
sudo nvpmodel -m 0 && sudo jetson_clocks
python3 benchmarks/hello_latency.py --iters 2000 --label pm0-15w
sudo nvpmodel -m 2 && sudo jetson_clocks
python3 benchmarks/hello_latency.py --iters 2000 --label pm2-maxn
sudo jetson_clocks --restore          # unpins the clocks again
```

The two records should differ, and each should have captured its own mode. This
is the check that catches teammates comparing numbers taken in different modes.

Put the board back where you found it when you are done. Ours ships in mode 2
with `jetson_clocks` off. `nvpmodel -m 2` alone does not unpin the clocks, so
`--restore` (or a reboot) is the part that undoes it.

`bash benchmarks/sanity_checks.sh` does all of this in one go. It runs this
check in all three modes, repeats check 2 with the clocks pinned, and restores
the board on exit, even after Ctrl-C or a failure. It asks for the sudo password
once, and `--dry-run` tries it without sudo.

## Power modes on our board

`sjsujetson-36` has three:

| Mode | Name | What it is |
| --- | --- | --- |
| 0 | 15W | The original pre-Super Orin Nano envelope. Our tightest power budget |
| 1 | 25W | The module TDP, capped. Deterministic |
| 2 | MAXN SUPER | Uncapped. Highest clocks, throttles back when it exceeds TDP |

It arrived in mode 2, MAXN SUPER.

### Which one to benchmark in

Run all three at one configuration early, as a cheap experiment, and see how
far apart they actually are on this board. Modes 1 and 2 share the same 25 W
TDP, and 2 only removes the cap on instantaneous clocks, so the gap between
them may be small. Measure rather than assume.

Then pick one as the primary mode for the main sweep and hold it fixed.
Sweeping power mode on top of precision, resolution, and chunk length triples
the grid for little insight.

Two arguments worth weighing when you pick:

- **MAXN SUPER gives the best latency**, which makes the "does the loop close
  fast enough" answer most favorable. It is also uncapped, so it throttles when
  it exceeds TDP. Throttling mid-run inflates p99, and p99 is the number this
  whole project hinges on.
- **Mode 1 at 25 W is deterministic.** A capped envelope is also the more
  honest story for a robot that runs on a battery, which is the deployment this
  project is about.

My suggestion: take the main sweep in mode 1, and report the best configuration
at all three modes as a power-versus-control-rate result. That gives a clean
primary sweep plus the power characterization the abstract promises, without
tripling the work.

Whatever you choose, write the decision down and make sure all three of you use
it. The harness records the mode in every result, so a mismatch is detectable
after the fact, but it is cheaper to agree up front.

### Watch for thermal throttling

In MAXN SUPER especially, a long run can heat the board until it clocks down.
The symptom is latency that creeps up over a sweep and a p99 that gets worse
the longer you run.

```bash
tegrastats --interval 1000     # watch the temperature and GPU frequency columns
```

If frequency drops partway through a run, the numbers from that run are not
comparable to a cold-start run. Re-run, or let the board settle first and say
so in the result.

## Power measurement

The harness reads the power rails as of 2026-09-28. `hello_latency.py` starts
a sampler thread after warmup and stops it after the timed loop. The thread
reads the INA3221 rails (`VDD_IN`, `VDD_CPU_GPU_CV`, `VDD_SOC`), the GPU clock
and the `tj` temperature straight from sysfs, with no root and no jtop. The
record's `power` block holds mean and peak watts per rail, energy per inference
(`VDD_IN` mean × mean latency, idle draw included), the GPU clock range and the
peak temperature. On a machine without the sensor it is `null`, so the script
still runs on the workstation.

Two things to know:

- Sampling is every 50 ms by default (`--power-interval-ms`). Each current
  read is an I2C transaction of about 0.7 ms, and the sensor itself updates
  about every 8 ms. Over 3000 iterations p99 was 4.90 ms with sampling and
  4.98 ms without (`--no-power`), so it does not perturb the measurement.
- A timed loop shorter than about a second yields too few samples, and the
  harness warns below 20. Raise `--iters` when the power number matters.

The other ways to get the number, for ad hoc checks:

```bash
# text stream, easy to parse, sample while a run is in flight
tegrastats --interval 100 --logfile /tmp/tegra.log
```

```python
# python API from jetson-stats, Jetson only
from jtop import jtop

with jtop() as jetson:
    while jetson.ok():
        print(jetson.power["tot"])   # total board power, milliwatts
        print(jetson.stats["Temp tj"])
```

The pattern that works: start a sampler thread before the measurement loop,
stop it after, and report mean and peak board power alongside the latency
percentiles in the same JSON record. Sampling only before and after misses the
peak entirely.

Guard the import. `jtop` is Jetson-only, and `benchmarks/` has to stay
importable on the host workstation for analysis.

## Where this goes next

`hello_latency.py` is the template, not the product. The real harness differs
in three ways:

1. It loads a real policy from a path given by a config file, never a constant.
2. It feeds real camera frames, or recorded frames replayed at the right rate,
   so preprocessing is inside the measured region. Preprocessing on the CPU is
   part of end-to-end latency and it is easy to accidentally exclude.
3. It samples power for the duration of the run.

Keep the measurement logic separate from the inference code it measures. That
separation is the reason `benchmarks/` and `src/` are different directories.

## Sweep bookkeeping

The experiment axes from the abstract:

- precision: FP16, INT8, INT4 / W4A16
- runtime: PyTorch eager, TensorRT engine
- input resolution
- flow-matching denoising steps (the published model uses 10 Euler steps at
  inference, and fewer steps is a direct latency lever)
- action chunk length

That is a large grid. Do not run it by hand. One YAML per point in `configs/`,
one record per run in `results/`, and open the
[experiment issue](../../.github/ISSUE_TEMPLATE/experiment.md) with the config
before you run the sweep, as CONTRIBUTING requires.

## Done when

- [ ] `benchmarks/hello_latency.py` runs clean on the Jetson
- [ ] A record exists in `results/` with a non-dirty git commit
- [ ] The three honesty checks above behave as described
- [ ] The team agrees on the target control rate, and it is written down
- [ ] Power sampling is on someone's plate as a tracked issue
