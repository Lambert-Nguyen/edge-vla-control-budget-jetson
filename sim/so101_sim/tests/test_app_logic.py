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
