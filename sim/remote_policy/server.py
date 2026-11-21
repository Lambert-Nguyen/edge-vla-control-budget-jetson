#!/usr/bin/env python3
"""Policy server on the Jetson for the remote simulation link.

The simulator on the workstation sends an observation, and this server runs
the policy on the board and sends back one action chunk. So a simulation
score measures the policy as it runs on the Jetson.

Backends:
- dummy: returns a chunk of zeros. Tests the link and its round trip time.
- hold: returns the present state as every action. Tests a RoboTwin episode
  end to end without a model.
- smolvla: SmolVLA in PyTorch on the GPU. Takes up to two images and the
  joint state, like on the arm.

A Hy-VLA backend (TensorRT engines) comes with the engine builder.

    python3 sim/remote_policy/server.py --backend dummy
    python3 sim/remote_policy/server.py --backend hold
    python3 sim/remote_policy/server.py --backend smolvla --steps 5

The server serves one client at a time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from sim.remote_policy.protocol import DEFAULT_PORT, pack, unpack  # noqa: E402


class DummyBackend:
    name = "dummy"

    def __init__(self, chunk: int = 50, action_dim: int | None = None):
        self.chunk, self.action_dim = chunk, action_dim

    def reset(self) -> None:
        pass

    def infer(self, obs: dict, task: str):
        import numpy as np

        dim = self.action_dim or len(obs.get("state", [0] * 6))
        return np.zeros((self.chunk, dim), dtype=np.float32)

    def info(self) -> dict:
        return {"backend": self.name, "chunk": self.chunk}


class HoldBackend:
    """Returns the present state as every action: the arm holds still.

    For RoboTwin link tests with action_type ee: the 16-d end-effector state
    is a valid ee action, so an episode runs end to end without a model.
    """

    name = "hold"

    def __init__(self, chunk: int = 40):
        self.chunk = chunk

    def reset(self) -> None:
        pass

    def infer(self, obs: dict, task: str):
        import numpy as np

        return np.tile(np.asarray(obs["state"], dtype=np.float32), (self.chunk, 1))

    def info(self) -> dict:
        return {"backend": self.name, "chunk": self.chunk}


class SmolVLABackend:
    """SmolVLA with the same pipeline as lerobot-record. Images keyed by order."""

    name = "smolvla"

    def __init__(self, policy_path: str, steps: int | None):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = torch.device("cuda")
        self.policy = SmolVLAPolicy.from_pretrained(policy_path).to(self.device).eval()
        if steps:
            self.policy.config.num_steps = steps
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": "cuda"}})
        self.image_keys = list(self.policy.config.image_features)
        self.n = self.policy.config.n_action_steps
        self.path = policy_path

    def reset(self) -> None:
        self.policy.reset()

    def infer(self, obs: dict, task: str):
        from lerobot.utils.control_utils import prepare_observation_for_inference

        images = [v for k, v in sorted(obs.items()) if k != "state"]
        frame = {"observation.state": obs["state"].astype("float32")}
        for key, img in zip(self.image_keys, images):
            frame[key] = img
        with self.torch.inference_mode():
            batch = prepare_observation_for_inference(frame, self.device, task, "so101_follower")
            batch = self.pre(batch)
            actions = self.policy.predict_action_chunk(batch)[0, : self.n]
            return self.post(actions).cpu().numpy()

    def info(self) -> dict:
        return {"backend": self.name, "policy": self.path, "num_steps": self.policy.config.num_steps,
                "chunk": self.n, "image_keys": self.image_keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("dummy", "hold", "smolvla"), default="dummy")
    ap.add_argument("--policy", default="lerobot/smolvla_base")
    ap.add_argument("--steps", type=int, help="denoising steps (smolvla)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    from websockets.sync.server import serve

    if args.backend == "smolvla":
        backend = SmolVLABackend(args.policy, args.steps)
    else:
        backend = DummyBackend() if args.backend == "dummy" else HoldBackend()
    if args.backend == "smolvla":  # warm up, so the first call is not slow
        import numpy as np

        warm = {"state": np.zeros(6, np.float32), "a": np.zeros((480, 640, 3), np.uint8),
                "b": np.zeros((480, 640, 3), np.uint8)}
        for _ in range(2):
            backend.infer(warm, "warm up")
        backend.reset()

    def handle(ws):
        peer = ws.remote_address
        print(f"client {peer} connected", flush=True)
        for data in ws:
            try:
                msg = unpack(data)
                kind = msg.get("type")
                if kind == "info":
                    reply = {"ok": True, **backend.info()}
                elif kind == "reset":
                    backend.reset()
                    reply = {"ok": True}
                elif kind == "infer":
                    t = time.perf_counter()
                    actions = backend.infer(msg["obs"], msg.get("task", ""))
                    reply = {"ok": True, "actions": actions, "server_ms": (time.perf_counter() - t) * 1e3}
                else:
                    reply = {"ok": False, "error": f"unknown message type {kind!r}"}
            except Exception as e:  # report to the client, keep serving
                reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            ws.send(pack(reply))
        print(f"client {peer} left", flush=True)

    print(f"policy server ({backend.name}) on ws://{args.host}:{args.port}", flush=True)
    with serve(handle, args.host, args.port, max_size=None, compression=None) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
