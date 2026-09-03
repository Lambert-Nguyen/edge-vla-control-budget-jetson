<!--
Every PR needs approval from at least one teammate before merge. See
CONTRIBUTING.md. Squash-merge, and make the squashed message follow the
commit convention.
-->

## Summary

<!-- What changed and why. One or two sentences. -->

## Type

- [ ] `feat` — new capability
- [ ] `fix` — bug fix
- [ ] `exp` — experiment configs and/or results
- [ ] `docs` — report, notes, README
- [ ] `chore` / `refactor` / `perf` / `test`

## Related

<!-- Closes #, Refs # -->

## How this was verified

<!-- Paste real output — not a description of what you expect. Say where it
     ran: Jetson, host workstation, or sim. CI is not set up, so the reviewer
     has only this and the diff to go on. -->

```text

```

- Ran on: <!-- Jetson Orin Nano / host / RoboTwin sim / SO-101 arm -->

## Effect on measurements

<!-- Anything that touches preprocessing, precision, batching, control loop
     timing, or the sim task set can change results that are already recorded. -->

- [ ] No effect — this cannot change any recorded measurement
- [ ] Changes measurements. Which previously recorded results are now stale, and are they being re-run?

<!-- If checked, explain here: -->

## Checklist

- [ ] Branched from `main`, rebased on latest `main`
- [ ] No model weights, checkpoints, engines, datasets, recordings, or video added
      (`git diff --stat origin/main...HEAD` reviewed for large files)
- [ ] Anything added to `results/` is CSV/JSON and references its config and commit
- [ ] No hardcoded checkpoint paths, absolute paths, or machine-specific values —
      those come from `configs/` or CLI arguments
- [ ] Jetson-only imports are guarded so host-side code still runs
- [ ] For hardware changes: workspace and torque limits still enforced, and the
      change was tested with the arm powered down first where possible
- [ ] Docs updated if setup or the run procedure changed

## Notes for the reviewer

<!-- Anything specific to look at, known limitations, or follow-up left undone. -->
