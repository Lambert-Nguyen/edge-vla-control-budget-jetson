# Contributing

## Workflow
1. Create a feature branch from the active development branch.
2. Implement one focused change per branch.
3. Open a pull request (PR) and link any related issue.
4. Merge only after approval and required checks.

## Branch-Per-Feature
- Use one branch per feature, fix, or experiment setup change.
- Suggested naming:
  - `feature/<short-description>`
  - `fix/<short-description>`
  - `docs/<short-description>`

## Pull Request Review
- Every PR must be reviewed by at least one teammate before merge.
- Teammate reviewers for this project:
  - Lam Nguyen
  - Vi Thi Tuong Nguyen
  - James Pham

## Commit Message Convention
Use concise, imperative commit messages with a scoped prefix:
- `feat: add RoboTwin latency harness scaffold`
- `fix: correct Jetson telemetry parser`
- `docs: update experiment logging template`

Recommended format:
`<type>: <short imperative summary>`

Common types:
- `feat`
- `fix`
- `docs`
- `refactor`
- `test`
- `chore`
