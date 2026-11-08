# Meeting Log

**Project:** Real-Time Control Budgets for Vision-Language-Action Manipulation Policies on the NVIDIA Jetson Orin Nano
**Course:** CMPE 295A, Fall 2026 (instruction began 2026-08-19)
**Team:** Lam Nguyen, Vi Thi Tuong Nguyen, James Pham
**Advisor:** Dr. Kaikai Liu

One note per meeting, named `YYYY-MM-DD-advisor.md` or `YYYY-MM-DD-team.md`, following the course template ([TEMPLATE.md](TEMPLATE.md)). Each note has five sections: accomplished since the last meeting, accomplished during the meeting, issues and blockers, action items, and plans for the next meeting. Action items name an owner and a due date, and the next note reports on them.

Team-only meetings are informal discussions held in person, by text and in the team's Facebook Messenger group chat. Their start and end times were not recorded, and the notes say so. Credentials and door codes are never written here, because this repository is public. Advisor-meeting photos are in [photos/](photos/), with location data removed.

## Meetings, Fall 2026

| Date | Type | Mode | Key outcomes | Note |
| --- | --- | --- | --- | --- |
| 2026-09-01 | Advisor | In person, office hours | Received the Jetson Orin Nano. Pivoted from autonomous driving to robotic-arm manipulation | [notes](2026-09-01-advisor.md) |
| 2026-09-02 | Team | Informal | Abstract finalized paragraph by paragraph, technical claims corrected. Repository created | [notes](2026-09-02-team.md) |
| 2026-09-14 | Team | Informal | Model memory estimates reviewed. Board bring-up started; access questions sent to Dr. Liu | [notes](2026-09-14-team.md) |
| 2026-09-15 | Advisor | In person, office hours | Brief status check on the abstract, repository and model-memory plan. Confirmed that the provisioned Jetson should be audited before changes | [notes](2026-09-15-advisor.md) |
| 2026-09-22 | Team | Informal | Setup guides merged (PR #2). Board treated as pre-provisioned. SO-101-class arm chosen; lab arm loan requested | [notes](2026-09-22-team.md) |
| 2026-09-28 | Team | Informal | Board environments verified. Harness checks; clocks locked for all experiments. Hiwonder servo issue found | [notes](2026-09-28-team.md) |
| 2026-09-29 | Advisor | In person, office hours | 295A baseline / 295B paper plan. Simulate the follower first, then test on the lab arm. Buy own arms. Simulation on an RTX 4090. Lab access form | [notes](2026-09-29-advisor.md) |
| 2026-10-07 | Team | Not recorded | Self-assembly versus fully assembled arm compared; no order yet. Lab configuration, admin approvals and experiment settings remain open. Simulator and SmolVLA status require verification (October 12 recap) | [notes](2026-10-07-team.md) |
| 2026-10-14 | Team | Not recorded | Date confirmed; retrospective review draft ties S3 tools/access work and S4 baseline preparation to Workbook I. Discussion details need confirmation | [notes](2026-10-14-team.md) |
| 2026-10-22 | Team / lab | In person, SJSU lab | Leader and follower checks, calibration, teleoperation and cameras worked with the Jetson; arm joint values read on the board. SmolVLA bring-up results traced to PR #6 | [notes](2026-10-22-team.md) |
| 2026-10-28 | Team | Not recorded | TensorRT test engine and pinned SmolVLA baselines; MAXN SUPER/30 Hz settings; Workbook II and three design specs define the handoff and S5 work | [notes](2026-10-28-team.md) |
| 2026-11-04 | Team | Not recorded | Own SO-ARM101 Pro kit received and calibrated; James has the camera. Follow-ups cover live-camera/async integration, the kit's safety profile and the Hy-VLA/RoboTwin baseline | [notes](2026-11-04-team.md) |

The October 14 and October 22 records were reconstructed on October 24 using the team's recap, repository evidence and *Project Workbook - Workbook I* (dated October 1), especially Chapters 3–5 and 8–9. Workbook targets are plans, not proof of completion. Proposed follow-up deadlines are labeled in each note. The October 22 lab session replaced that week's Wednesday meeting; it is not an advisor meeting.

The October 28 and November 4 notes were reconstructed on November 7 using the confirmed Wednesday team schedule, the sources below and the team's confirmation that its own arm arrived and was calibrated and James has the camera. Meeting times and modes were not supplied. Completed work is distinguished from planned specifications; unreported software/test outcomes remain unconfirmed. No advisor meeting is inferred from a team meeting or document date.

## Sources for the October 28 and November 4 notes

- *Project Workbook - Workbook II*, dated October 29, 2026: Chapters 6–9, including implemented blocks (Table 6), test outcomes (Tables 7–8), implementation progress (Tables 11–12), and sprint/milestone targets (Tables 13–14).
- James Pham, *TensorRT Engine Builder and Runtime on the Jetson*, design specification v1.1, October 28: current test-engine results, the planned two-engine runtime, reference checks, engine cache and board-memory tests.
- Lam Nguyen, *Hy-VLA Export and Quantization on the Workstation*, design specification v1.1, October 28: two-graph export, precision/memory budget, L0–L3 checks and the manifest/reference handoff to James.
- Vi Thi Tuong Nguyen, *Safety Layer for the SO-101 Follower Arm*, design specification v1.1, October 28: completed lab checks/step cap, planned CHECK/RUN/HOLD/STOP/PARK states, watchdog, profiles and E-8 tests.
- Repository evidence: [PR #8](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/8), [October 26 JSON records](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/tree/fd8dc20cd5628ad91cdc30e7a65aacc16ced26f7/results), [documented sweep settings](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/blob/c709cff2bf5088f7454804f2ec118e92e37984a4/README.md#sweep-setting), and [PR #9](https://github.com/Lambert-Nguyen/edge-vla-control-budget-jetson/pull/9).

## Advisor correspondence (email)

Between meetings, the team and Dr. Liu also kept in touch by email. These are not meetings, and are listed here only so the notes can be traced.

| Date | From | Summary |
| --- | --- | --- |
| 2026-09-14 | James (for the team) | Board access questions: setup guide, flashing, login, SSH, network |
| 2026-09-15 | Dr. Liu | Follow-up to the advisor check-in: the board is already set up. Links to the [edgeAI tutorial](https://lkk688.github.io/edgeAI/) and [sjsujetsontool guide](https://lkk688.github.io/edgeAI/curriculum/00_sjsujetsontool_guide/). Account details (not recorded here) |
| 2026-09-22 | James (for the team) | Can the team borrow a lab SO-101 pair? Is the Hiwonder SO-ARM101 kit suitable? |
| 2026-09-29 | Dr. Liu | Arm and sim-to-real resources (see the [2026-09-29 notes](2026-09-29-advisor.md)) and the ENG276 lab-access form |
| 2026-09-29 | James (for the team) | Lab access requested. Does the lab's SO-ARM101 include the leader arm? |
| 2026-09-30 | Dr. Liu | Lab-access form signed. Invitation to ENG276 during a NASA Ames visit to meet a former student who worked on the lab's arm |

## Before Fall 2026

How the team and advisor came together, for context:

- **2026-04-24:** Lam emailed Dr. Liu, on the recommendation of Prof. Dan Harkey, asking him to advise the team.
- **Late April 2026:** introductory meeting with the team during Dr. Liu's Thursday office hours. He suggested NVIDIA CUDA-BEVFusion and Xiaomi MindDrive as directions.
- **2026-05-12:** the team proposed a tentative title, "Efficient On-Device Deployment of Vision-Language-Action Models for Real-Time Autonomous Driving on NVIDIA Jetson Platforms".
- **2026-05-21:** Dr. Liu registered the project for CMPE 295A, Fall 2026.
