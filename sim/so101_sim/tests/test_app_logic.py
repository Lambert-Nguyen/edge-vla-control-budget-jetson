"""App logic without a browser: commands in, simulator state out.

Rendering is replaced by blank images so these run fast on any machine, including CI
runners without a GPU.
"""
import numpy as np
import pytest

from so101_sim.app.server import SimApp
from so101_sim.config import SimConfig

lerobot = pytest.importorskip("lerobot")


@pytest.fixture
def app(tmp_path, monkeypatch):
    cfg = SimConfig(width=64, height=48, episode_s=2.0)
    a = SimApp(cfg, dataset="app_test", root=str(tmp_path / "ds"), seed=3)
    blank = np.zeros((48, 64, 3), np.uint8)
    monkeypatch.setattr(a.env, "render_camera", lambda key: blank)
    monkeypatch.setattr(a, "publish", lambda obs: None)
    a.new_scene(3)
    a.enter_mode("keyboard", quiet=True)
    yield a
    a.shutdown()


def run(app, n):
    for _ in range(n):
        while not app.cmds.empty():
            app.handle(app.cmds.get())
        app.tick()


def test_keyboard_moves_the_jaws(app):
    import time
    start = app.env.tcp_pos().copy()
    app.handle({"type": "keys", "keys": ["fwd", "down"]})
    for _ in range(20):
        app.keys_t = time.monotonic()
        run(app, 1)
    moved = app.env.tcp_pos() - start
    assert moved[0] > 0.02 and moved[2] < -0.02


def hold(app, keys, seconds):
    import time
    app.handle({"type": "keys", "keys": keys})
    for _ in range(int(seconds * app.cfg.fps)):
        app.keys_t = time.monotonic()
        run(app, 1)
    app.handle({"type": "keys", "keys": []})
    run(app, 15)


def test_keyboard_moves_where_the_keys_say_at_the_wrist_limit(app):
    # The jaws cannot point straight down far out or high up: wrist_flex runs out.
    # The jaws should still go where the keys say, tilting less, and tilting
    # should not move them.
    start = app.kb["pos"].copy()
    hold(app, ["pitch_down"], 1.5)                       # well past what the wrist allows
    assert np.allclose(app.kb["pos"], start, atol=1e-3)
    assert np.linalg.norm(app.env.tcp_pos() - start) < 0.01
    hold(app, ["down"], 0.5)
    hold(app, ["fwd"], 1.5)                               # far out: the tilt has to give
    far = app.kb["pos"].copy()
    assert far[0] - start[0] > 0.12
    hold(app, ["back"], 1.0)
    back = app.kb["pos"] - far
    assert back[0] < -0.10 and abs(back[2]) < 0.005      # back means back, not up
    assert np.linalg.norm(app.env.tcp_pos() - app.kb["pos"]) < 0.01
    hold(app, ["pitch_down"], 1.0)                        # asks for more than the wrist has
    tilt = app.kb["pitch_now"]
    hold(app, ["pitch_up"], 0.3)                          # answers at once: no wind-up to undo
    assert app.kb["pitch_now"] < tilt - np.radians(15)


def test_sliders_set_joints(app):
    app.handle({"type": "mode", "mode": "sliders"})
    target = [20.0, -30.0, 30.0, 60.0, 10.0, 50.0]
    app.handle({"type": "joints", "values": target})
    run(app, 45)
    assert np.allclose(app.env.state()[:5], target[:5], atol=2.0)


def test_record_save_discard_and_time_limit(app, tmp_path):
    app.handle({"type": "record"})
    run(app, 10)
    app.handle({"type": "save"})
    assert app.saved_episodes == 1 and not app.recording
    app.handle({"type": "record"})
    run(app, 5)
    app.handle({"type": "discard"})
    assert app.saved_episodes == 1
    app.handle({"type": "record"})
    run(app, app.cfg.max_steps + 2)          # saves itself at the time limit
    assert app.saved_episodes == 2 and not app.recording
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/app_test", root=str(tmp_path / "ds"))
    assert ds.num_episodes == 2 and ds.num_frames == 10 + app.cfg.max_steps


def test_expert_batch_records_successful_demos(app, tmp_path):
    app.handle({"type": "fast", "value": True})
    app.handle({"type": "batch", "n": 2})
    for _ in range(2000):
        run(app, 1)
        if app.batch_left == 0:
            break
    assert app.batch_left == 0 and app.batch_stats[0] == 2
    assert app.saved_episodes == 2


def test_policy_mode_with_expert_and_latency(app):
    app.handle({"type": "mode", "mode": "policy", "policy": "expert", "latency_ms": 300, "policy_mode": "async"})
    assert app.mode == "policy"
    run(app, 400)
    assert app.executor.inferences >= 2
    assert app.success_streak > 0, "expert-as-policy should finish the task"


def test_bad_leader_port_falls_back_to_keyboard(app):
    app.handle({"type": "mode", "mode": "leader", "leader_port": "/dev/does-not-exist"})
    assert app.mode == "keyboard"
    assert any("could not switch to leader" in line for line in app.log_lines)


def test_failed_save_drops_the_episode_and_the_next_one_is_clean(app, tmp_path, monkeypatch):
    app.handle({"type": "record"})
    run(app, 8)
    real = app.rec.ds.save_episode
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            app.rec.ds.writer.episode_buffer.pop("size")    # what LeRobot does before encoding
            raise RuntimeError("encoder died")
        return real(*a, **kw)

    monkeypatch.setattr(app.rec.ds, "save_episode", flaky)
    app.handle({"type": "save"})
    assert app.saved_episodes == 0 and not app.recording
    assert any("could not save the episode" in line for line in app.log_lines)
    app.handle({"type": "record"})
    run(app, 6)
    app.handle({"type": "save"})
    assert app.saved_episodes == 1
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/app_test", root=str(tmp_path / "ds"))
    assert ds.num_episodes == 1 and ds.num_frames == 6


def test_follower_mirrors_the_simulated_arm(app, monkeypatch):
    import so101_sim.follower as follower
    from test_core import FakeFollower
    robot = FakeFollower(app.env.state())
    real = follower.FollowerMirror
    monkeypatch.setattr(follower, "FollowerMirror", lambda port, fid, **kw: real(port, fid, robot=robot, **kw))
    app.handle({"type": "follower", "on": True})                 # no port yet: refused
    assert app.follower is None
    app.handle({"type": "follower", "on": True, "port": "COM9", "id": "f1"})
    assert app.follower is not None and app.follower_port == "COM9"
    app.handle({"type": "mode", "mode": "expert"})
    run(app, 60)
    assert len(robot.sent) == 60
    assert np.abs(robot.sent[-1] - app.last_action).max() < 10   # keeping up with the expert
    app.handle({"type": "fast", "value": True})                  # too fast for a real arm: it waits
    run(app, 5)
    assert len(robot.sent) == 60
    app.handle({"type": "fast", "value": False})
    robot.fail = True                                             # unplugged: mirror stops, app goes on
    run(app, 3)
    assert app.follower is None and not robot.connected and app.step == 68   # the app went on
    app.handle({"type": "follower", "on": True})
    robot.fail, robot.connected = False, True
    app.handle({"type": "follower", "on": False})
    assert app.follower is None and not robot.connected
