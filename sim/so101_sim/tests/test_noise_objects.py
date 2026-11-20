"""Objects, multi-object scenes and noise. No LeRobot."""
import numpy as np
import pytest

from so101_sim import objects
from so101_sim.cli import parse_objects
from so101_sim.config import OBJECT_KINDS, SimConfig, noise_label, parse_noise
from so101_sim.env import SO101Env
from so101_sim.expert import run_expert_episode
from so101_sim.noise import QUANT_DEG
from so101_sim.policy_eval import ExpertChunkPolicy, LatencyModel, rollout


def play(env, seed):
    for _ in run_expert_episode(env, seed, extra_s=0.5):
        pass
    return env.is_success()


def test_original_cube_is_untouched():
    e = SO101Env(SimConfig(randomize="full"), render=False)
    e.reset(seed=3, render=False)
    assert e.original_cube and e.noise is None
    assert e.task == "Pick up the cube and put it in the box."
    assert e.obj.kind == "cube" and e.obj.mass == pytest.approx(0.03)
    for d in e.distractors:                      # parked under the table, no contacts
        assert e.data.xpos[d["body"]][2] < -0.1 and e.model.geom_contype[d["geom"]] == 0


@pytest.mark.parametrize("kind", OBJECT_KINDS)
def test_expert_picks_every_object(kind):
    e = SO101Env(SimConfig(randomize="full", objects=(kind,)), render=False)
    assert all(play(e, s) for s in range(4))
    assert e.task == f"Pick up the {objects.NAMES[kind]} and put it in the box."


@pytest.mark.parametrize("kind", OBJECT_KINDS)
def test_objects_rest_on_the_table(kind):
    e = SO101Env(SimConfig(randomize="light", objects=(kind,)), render=False)
    e.reset(seed=1, render=False)
    z = e.cube_pose()[0][2]
    assert z == pytest.approx(e.obj.half_height, abs=2e-3)


def test_layout_does_not_depend_on_objects_or_noise():
    def layout(**kw):
        e = SO101Env(SimConfig(randomize="full", **kw), render=False)
        e.reset(seed=11, render=False)
        return e.box_pos()[:2].copy(), e.start.copy()
    box, start = layout()
    for kw in ({"noise": 1.0}, {"objects": OBJECT_KINDS}, {"objects": ("ball",), "noise": 2.0}):
        b, s = layout(**kw)
        assert np.allclose(b, box) and np.allclose(s, start)


@pytest.mark.parametrize("scene", ["colors", "kinds"])
def test_multi_object_scenes(scene):
    e = SO101Env(SimConfig(randomize="full", objects=OBJECT_KINDS, scene=scene), render=False)
    for seed in range(6):
        e.reset(seed=seed, render=False)
        info = e.episode_info()
        others = info.get("distractors", [])
        assert others, "at least one distractor"
        if scene == "colors":
            assert all(o["kind"] == e.obj.kind for o in others)
            colors = [e.target_color] + [o["color"] for o in others]
            assert len(set(colors)) == len(colors)
            assert e.target_color in e.task
        else:
            kinds = [e.obj.kind] + [o["kind"] for o in others]
            assert len(set(kinds)) == len(kinds)
            assert {o["color"] for o in others} == {e.target_color}
        # every object on the table, apart from the others and outside the box
        pos = [e.cube_pose()[0]] + [e.data.xpos[d["body"]].copy() for d in e.distractors if d["spec"]]
        assert all(p[2] > 0 for p in pos)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                assert np.linalg.norm(pos[i][:2] - pos[j][:2]) > 0.05
        assert not e.distractor_in_box()
        assert play(e, seed) and not e.distractor_in_box()


def test_noise_off_is_exactly_the_clean_simulator():
    a = SO101Env(SimConfig(randomize="full"), render=False)
    b = SO101Env(SimConfig(randomize="full", noise=0.0), render=False)
    sa = [obs.state.copy() for obs, _ in run_expert_episode(a, 5)]
    sb = [obs.state.copy() for obs, _ in run_expert_episode(b, 5)]
    assert np.array_equal(np.array(sa), np.array(sb))


def test_noise_on_state_and_actuation():
    clean = SO101Env(SimConfig(randomize="light"), render=False)
    noisy = SO101Env(SimConfig(randomize="light", noise=1.0), render=False)
    clean.reset(seed=2, render=False)
    noisy.reset(seed=2, render=False)
    n = noisy.noise
    assert n is not None and np.abs(n.offset).max() > 0 and not np.allclose(n.kp_scale, 1)
    s = noisy.state()
    # arm joints sit on encoder ticks; readings differ from the truth by about the offset
    assert np.allclose(np.round(s[:5] / QUANT_DEG) * QUANT_DEG, s[:5])
    assert not np.allclose(s, clean.state(), atol=0.05)
    # the expert compensates the calibration offset, so it still succeeds
    assert all(play(noisy, seed) for seed in range(4))
    # the same seed draws the same noise
    noisy.reset(seed=2, render=False)
    assert np.allclose(noisy.noise.offset, n.offset)


def test_backlash_holds_small_reversals():
    e = SO101Env(SimConfig(randomize="none", noise=1.0), render=False)
    e.reset(seed=0, render=False)
    n = e.noise
    n.action_sigma[:] = 0.0
    up = e.start + 10.0 * n.backlash                  # far more than any calibration offset
    first = n.command(up)                              # moving up takes up the slack
    back = n.command(up - 0.5 * n.backlash)            # a reversal inside the play: gears stay
    assert np.allclose(first[:5], back[:5])
    down = n.command(up - 2.0 * n.backlash)            # past the play: they follow
    assert np.all(down[:5] < first[:5])


def test_camera_noise_is_late_and_degraded():
    e = SO101Env(SimConfig(randomize="light", noise=1.0, width=160, height=120), render=True)
    try:
        e.reset(seed=4)
        delay = e.noise.cameras["camera1"]["delay"]
        assert delay >= 1
        clean_frames, noisy_frames = [], []
        for _ in range(delay + 3):
            clean_frames.append(e.render_camera("camera1").astype(int))
            noisy_frames.append(e.step(e.state()).images["camera1"].astype(int))
        # the noisy image is not the clean render of the same step
        assert np.abs(noisy_frames[-1] - clean_frames[-1]).mean() > 1.0
    finally:
        e.close()


def test_expert_as_policy_with_noise_and_distractors():
    e = SO101Env(SimConfig(randomize="full", objects=OBJECT_KINDS, scene="kinds", noise=1.0), render=False)
    r = rollout(e, ExpertChunkPolicy(), LatencyModel.constant(488), seed=100002, mode="async", render=False)
    assert r.success and not r.wrong_object_in_box and r.task == e.task
    assert r.scene["scene"] == "kinds" and r.scene["distractors"]


def test_parsers():
    assert parse_noise("none") == 0 and parse_noise("real") == 1 and parse_noise("heavy") == 2
    assert parse_noise("0.5") == 0.5 and noise_label(1.0) == "real" and noise_label(0.5) == "0.5"
    with pytest.raises(ValueError):
        parse_noise("-1")
    assert parse_objects("all") == OBJECT_KINDS
    assert parse_objects("Ball, cube,ball") == ("ball", "cube")
    with pytest.raises(SystemExit):
        parse_objects("banana")


def test_mixed_scene_draws_all_three():
    e = SO101Env(SimConfig(randomize="full", objects=OBJECT_KINDS, scene="mixed"), render=False)
    seen = set()
    for seed in range(20):
        e.reset(seed=seed, render=False)
        seen.add(e.episode_info()["scene"])
    assert seen == {"single", "colors", "kinds"}
