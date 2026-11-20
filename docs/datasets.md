# Datasets

Datasets stay on the Jetson and are never uploaded or put in git. LeRobot keeps
them in `~/.cache/huggingface/lerobot/local/<name>`. This file lists them, so the
team knows which one to use and how it was made.

## pillow_case_clean (use this one)

| | |
|---|---|
| Task text | `Put the teal pillow on the red phone case.` |
| Episodes | 9, each 20 s at 30 Hz (5391 frames) |
| Recorded | Nov 20, 2026, by James, with `so101_lab.sh record` |
| Arm | team SO-ARM101 Pro, follower `team_follower`, leader `team_leader` |
| camera1 | Logitech Brio, top front view, 640x480, 60 fps MJPG |
| camera2 | kit wrist camera, 640x480, 30 fps |
| Made from | `pillow_case` without episode 1 |

The teal pillow and the red phone case stand in for a cube and a box until we
have those. When we get them, they go in a new dataset with its own task text.

The Brio settings for these demos were manual focus 30, exposure 160 (16 ms),
gain 80 and white balance 6000 K, with the desk lamp on. Use the same light and
the same camera position for more demos and for policy runs on this task.

### Why episode 1 was dropped

In episode 1 the gripper caught only the edge of the pillow and pushed it onto
the phone case instead of lifting it. A policy should not learn that. The wrist
camera shows it: at 8 s the pillow hangs from one edge, and at 11 s the gripper
is already open. In the other episodes the gripper opens to 37 to 78 (of 100),
closes on the pillow and carries it.

The clean copy was made with:

```bash
lerobot-edit-dataset --repo_id local/pillow_case --new_repo_id local/pillow_case_clean \
    --operation.type delete_episodes --operation.episode_indices "[1]"
```

The clean copy numbers its episodes 0 to 8 again, so old episode 2 is now
episode 1.

## pillow_case

The full recording, 10 episodes. Keep it, but train on `pillow_case_clean`.

## pillow_case_test

A 10 s test before the real demos. Only for checking the setup. The Brio picture
in it is too dark (no gain, 4000 K).

## Next

Nine demos are enough to check the pipeline, but too few to fine-tune SmolVLA
well. We plan to record a few dozen more with the same setup.
