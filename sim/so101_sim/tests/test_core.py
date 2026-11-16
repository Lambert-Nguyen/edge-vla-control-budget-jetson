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
