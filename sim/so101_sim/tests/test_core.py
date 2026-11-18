"""Simulator, units, expert and latency logic. No LeRobot, no rendering."""
import math

import numpy as np
import pytest

from so101_sim.config import FEATURE_NAMES, SimConfig
from so101_sim.env import SO101Env
from so101_sim.expert import run_expert_episode
from so101_sim.policy_eval import ChunkExecutor, ChunkPolicy, LatencyModel, rollout, wilson, ExpertChunkPolicy
from so101_sim.units import JointMap


@pytest.fixture(scope="module")
def env():
    e = SO101Env(SimConfig(randomize="light"), render=False)
    yield e
    e.close()


def test_feature_names_match_lerobot_so101():
    assert FEATURE_NAMES == ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                             "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")


def test_units_round_trip():
    jm = JointMap(sign=[1, -1, 1, 1, -1], offset_deg=[0, 5, -3, 0, 2])
    q = np.array([0.3, -0.5, 1.0, -0.2, 2.0, 0.4])
    assert np.allclose(jm.to_sim(jm.to_lerobot(q)), q, atol=1e-5)
    plain = JointMap()
    v = plain.to_lerobot(np.array([math.radians(30), 0, 0, 0, 0, -0.174533]))
    assert v[0] == pytest.approx(30, abs=1e-4) and v[5] == pytest.approx(0, abs=1e-3)


def test_reset_is_deterministic(env):
    env.reset(seed=7, render=False)
    a = env.cube_pose()[0].copy(), env.box_pos().copy()
    env.reset(seed=8, render=False)
    env.reset(seed=7, render=False)
    b = env.cube_pose()[0], env.box_pos()
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_cube_and_box_never_overlap(env):
    for s in range(50):
        env.reset(seed=s, render=False)
        rel = np.abs(env.cube_pose()[0][:2] - env.box_pos()[:2])
        assert rel.max() > env.box_outer_half + env.cube_half


def test_start_pose_varies_but_stays_safe_and_keeps_the_layout():
    from so101_sim.config import HOME_POSE, START_MIN_HEIGHT, START_SPREAD
    e = SO101Env(SimConfig(randomize="light"), render=False)
    fixed = SO101Env(SimConfig(randomize="light", start_spread=0.0), render=False)
    starts = []
    for s in range(30):
        e.reset(seed=s, render=False)
        fixed.reset(seed=s, render=False)
        start = e.jmap.to_lerobot(e.start)
        assert np.all(np.abs(start - HOME_POSE) <= np.array(START_SPREAD) + 1e-3)
        assert e.tcp_pos()[2] >= START_MIN_HEIGHT - 0.005       # after settling
        assert np.allclose(e.cube_pose()[0], fixed.cube_pose()[0], atol=1e-3)   # same layout
        assert np.allclose(e.box_pos(), fixed.box_pos())
        assert np.allclose(fixed.start, fixed.home)
        starts.append(start)
    assert np.all(np.std(starts, axis=0) > 4)                   # every joint varies
    e.reset(seed=3, render=False)
    again = e.start.copy()
    e.reset(seed=3, render=False)
    assert np.allclose(e.start, again)                          # and a seed repeats it
    none = SO101Env(SimConfig(randomize="none"), render=False)
    none.reset(seed=5, render=False)
    assert np.allclose(none.start, none.home)


@pytest.mark.parametrize("level", ["none", "light", "full"])
def test_expert_solves_the_task(level):
    e = SO101Env(SimConfig(randomize=level), render=False)
    wins = 0
    for s in range(6):
        for _ in run_expert_episode(e, s):
            pass
        wins += e.is_success()
        assert np.all(np.isfinite(e.data.qpos)), "simulation went unstable"
    assert wins >= 5


class Ramp(ChunkPolicy):
    """Chunk whose actions encode the step they are meant for: [t, 0, 0, 0, 0, 0]."""
    n_action_steps = 10

    def chunk(self, obs, step, task):
        return np.array([[step + i, 0, 0, 0, 0, 0] for i in range(self.n_action_steps)], np.float32)


def drive(mode, latency_ms, steps=40, threshold=0.5):
    ex = ChunkExecutor(Ramp(), LatencyModel.constant(latency_ms), fps=30, mode=mode, threshold=threshold)
    ex.reset(np.full(6, -1, np.float32))
    return [float(ex.next_action(t, None, "")[0]) for t in range(steps)], ex


def test_async_drops_elapsed_actions():
    out, ex = drive("async", latency_ms=100)          # 3 steps at 30 Hz
    assert out[:3] == [-1, -1, -1]                    # waiting for the first chunk
    # Once chunks flow, the action played at step t was planned for step t.
    assert all(v == t for t, v in enumerate(out) if v >= 0)
    assert min(ex.staleness_ms) == pytest.approx(100.0)


def test_sync_holds_then_plays_whole_chunk():
    out, ex = drive("sync", latency_ms=100)
    assert out[:3] == [-1, -1, -1]
    assert out[3:13] == list(range(0, 10))            # the whole chunk, planned at step 0, late by 3
    assert out[13:16] == [9, 9, 9]                    # holds the last action while replanning
    # Stalls: steps 0-2, 13-15, 26-28 and 39 (each replan costs 3 steps).
    assert ex.stalls == 10


def test_zero_latency_async_never_stalls():
    out, ex = drive("async", latency_ms=0)
    assert out == list(range(40)) and ex.stalls == 0


def test_latency_from_harness_record(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"latency": {"chunk": {"p50_ms": 488.0, "stdev_ms": 4.5, "min_ms": 484, "max_ms": 516}}}')
    lat = LatencyModel.from_result(p)
    rng = np.random.default_rng(0)
    xs = [lat.sample_s(rng) for _ in range(200)]
    assert 0.484 <= min(xs) and max(xs) <= 0.516


def test_expert_as_policy_succeeds_at_jetson_latency(env):
    r = rollout(env, ExpertChunkPolicy(), LatencyModel.constant(488), seed=100001, mode="async", render=False)
    assert r.success


def test_wilson_interval():
    lo, hi = wilson(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97


# Layouts the expert used to fail: the cube or box far enough out that pointing the
# jaws straight down at hover or carry height needs more wrist_flex than the joint
# has. The fixed jaw then landed on the cube, or the cube on the box rim.
@pytest.mark.parametrize("level,seed", [("light", 243), ("light", 273), ("light", 315), ("light", 337),
                                        ("full", 256), ("full", 323), ("full", 100250)])
def test_expert_handles_far_layouts(level, seed):
    e = SO101Env(SimConfig(randomize=level), render=False)
    for _ in run_expert_episode(e, seed, extra_s=0.5):
        pass
    assert e.is_success()


def test_steepest_pitch_reaches_exactly_and_keeps_the_arm_configuration(env):
    from so101_sim.ik import LIFT, ArmIK
    ik = ArmIK(env.model, env.qadr, env.dadr)
    env.reset(seed=0, render=False)
    q0 = env.qpos_arm()
    target = np.array([0.23, 0.19, 0.12])          # carry height over a far box
    _, err_down = ik.solve(q0, target, math.pi / 2)
    q, pitch, err = ik.solve_steepest(q0, target)
    assert err_down > 0.01                          # straight down is out of reach
    assert err < 1e-3 and math.radians(45) < pitch < math.pi / 2
    assert q[LIFT] > ik.lo[LIFT] + 0.2             # not flipped to shoulder_lift at its limit
    near = np.array([0.20, 0.0, 0.05])              # reachable straight down: unchanged
    q_n, pitch_n, _ = ik.solve_steepest(q0, near)
    assert pitch_n == math.pi / 2 and np.allclose(q_n, ik.solve(q0, near, math.pi / 2)[0])


def test_policy_label():
    from so101_sim.cli import policy_label
    assert policy_label("outputs/train/smolvla_sim/checkpoints/020000/pretrained_model") == "smolvla_sim-020000"
    assert policy_label("outputs\\train\\act\\checkpoints\\last\\pretrained_model\\") == "act-last"
    assert policy_label("lerobot/smolvla_base") == "smolvla_base"
    assert policy_label("expert") == "expert"


class FakeFollower:
    """Stands in for LeRobot's SO101Follower: records what it is sent."""

    def __init__(self, pose):
        self.pose = {k: float(v) for k, v in zip(FEATURE_NAMES, pose)}
        self.sent, self.connected, self.fail = [], True, False

    def get_observation(self):
        return dict(self.pose)

    def send_action(self, action):
        if self.fail:
            raise ConnectionError("port vanished")
        self.sent.append(np.array([action[k] for k in FEATURE_NAMES]))
        self.pose = dict(action)
        return action

    def disconnect(self):
        self.connected = False


def test_follower_mirror_catches_up_slowly_then_follows_capped():
    from so101_sim.follower import FollowerMirror
    rest = np.array([0.0, -100.0, 95.0, 70.0, 0.0, 2.0])         # folded, as on the bench
    robot = FakeFollower(rest)
    m = FollowerMirror("PORT", "id", robot=robot)
    home = np.array([0.0, -40.0, 20.0, 70.0, 0.0, 20.0])
    for _ in range(200):
        m.send(home)
        if not m.catching_up:
            break
    steps = np.abs(np.diff(np.vstack([rest] + robot.sent), axis=0))
    assert np.allclose(robot.sent[-1], home) and steps.max() <= 1.0 + 1e-9   # 1 deg/step to start
    n_catch = len(robot.sent)
    # following: a fast move is capped at 3 deg/step and does not drop back to catch-up
    for i in range(1, 40):
        m.send(home + np.array([6.0 * min(i, 5), 0, 0, 0, 0, 0]))
        assert not m.catching_up
    steps = np.abs(np.diff(np.vstack(robot.sent[n_catch - 1:]), axis=0))
    assert steps.max() <= 3.0 + 1e-9 and np.allclose(robot.sent[-1][0], 30.0)
    # a jump in the simulator (a new scene) goes back to catching up
    m.send(home)
    assert m.catching_up and abs(robot.sent[-1][0] - 29.0) < 1e-9
    m.close()
    assert not robot.connected
