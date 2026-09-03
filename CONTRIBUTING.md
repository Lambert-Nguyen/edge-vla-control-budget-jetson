# Contributing

This is a three-person capstone repo with a shared Jetson and a shared arm.
The process below exists so that (a) nobody's work-in-progress breaks a
benchmark run someone else is in the middle of, and (b) every number that ends
up in the report can be traced back to the commit and config that produced it.

## Ground rules

1. **`main` is always runnable.** Never commit directly to `main`. All changes
   land through a pull request.
2. **No weights, no data, no video.** Model checkpoints, TensorRT engines,
   datasets, episode recordings, and large logs never enter git — see
   [.gitignore](.gitignore). If `git add` wants to stage a file over ~5 MB,
   stop and ask why. Never use `git add -f` to get around the ignore rules.
3. **`results/` holds CSV and JSON only.** Plots are regenerated from those
   files; they are not the source of truth.
4. **A measurement without its config is not a result.** Every record written
   to `results/` must identify the config file and the git commit it came
   from.

## Branch-per-feature workflow

One branch per feature, fix, or experiment. Branches are short-lived — open a
PR within a few days rather than accumulating a month of work.

```bash
git checkout main
git pull origin main
git checkout -b feat/trt-int8-engine-build
# ... work, committing as you go ...
git push -u origin feat/trt-int8-engine-build
```

Name branches `<type>/<short-description>` in kebab-case, using the same
`<type>` values as commit messages:

| Prefix | Use for |
| --- | --- |
| `feat/` | New capability — a harness, an export path, a policy wrapper |
| `fix/` | Bug fix |
| `exp/` | An experiment sweep and the configs/results it produces |
| `docs/` | Report sections, meeting notes, README |
| `chore/` | Dependencies, tooling, repo housekeeping |
| `refactor/` | Restructuring with no behavior change |

Rebase onto `main` to pick up others' work; do not merge `main` into your
branch repeatedly.

```bash
git fetch origin
git rebase origin/main
```

## Pull requests

- **Every PR needs approval from at least one teammate before merge.** No
  self-merging, including for docs. The reviewer is confirming they could
  re-run what you did, not just that the code looks fine.
- Fill in [the PR template](.github/pull_request_template.md). If the PR
  changes anything that could move a measurement — preprocessing, precision,
  batching, control loop timing, the sim task set — say so explicitly, because
  previously recorded results may no longer be comparable.
- Keep PRs reviewable. If a branch touches more than a few hundred lines
  across unrelated areas, split it.
- CI is not set up yet, so the reviewer is the only check. Run the code and
  paste real output into the PR description; don't describe what you expect it
  to do.
- Squash-merge into `main`. The squashed commit message follows the convention
  below.

### Review expectations

Reviewers should check:

- Does this assume a specific checkpoint, absolute path, or the reviewer's
  own machine? It shouldn't — model paths come from config or CLI arguments.
- Would this silently invalidate results already in `results/`?
- Does anything ignored by `.gitignore` sneak in?
- On hardware changes: are the workspace and torque limits still enforced?

## Commit messages

Conventional-Commits style:

```text
<type>(<scope>): <imperative summary, <= 72 chars>

<body: why the change was made, and anything a teammate would need to
reproduce it — board state, JetPack version, config used>

<footer: Refs #12>
```

`<type>` is one of `feat`, `fix`, `exp`, `docs`, `chore`, `refactor`, `perf`,
`test`. `<scope>` is usually a top-level directory: `src`, `bench`, `sim`,
`hardware`, `configs`, `results`, `docs`.

Examples:

```text
feat(src): add ONNX export path with configurable input resolution

fix(bench): stop counting the first inference in latency percentiles

The first call includes CUDA context creation and cuDNN autotuning, which
inflated p99 by ~400 ms. Now discards a configurable warmup count.

exp(results): int8 sweep at 224px, chunk sizes 1/4/8

Ran on Orin Nano, JetPack 6.0, nvpmodel 15W, jetson_clocks on.
Config: configs/int8_224_chunk_sweep.yaml
Refs #23
```

Rules:

- Imperative mood ("add", not "added" or "adds").
- No period at the end of the summary line.
- Explain **why** in the body. The diff already shows what changed.
- For anything measurement-related, record the board's power mode and clock
  state in the body — a number without them is not reproducible.

## Experiments

Experiments are tracked as issues, using the
[experiment template](.github/ISSUE_TEMPLATE/experiment.md). Open the issue
with the config *before* running the sweep, then fill in the measured results
and close it. This gives us a chronological log to write the report from.

## Working on shared hardware

The Jetson and the arm are shared. Before a timing or power run, post in the
team channel — background load from someone else's build will show up in your
percentiles. Never leave the arm powered and unattended, and confirm the
workspace limits are loaded before any closed-loop policy run.
