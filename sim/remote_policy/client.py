"""Client side of the remote policy link. It runs in the simulator process.

    from sim.remote_policy.client import RemotePolicy
    policy = RemotePolicy("sjsujetson-36.local")
    policy.reset()
    actions = policy.infer({"head": rgb, "wrist": rgb2, "state": q}, "pick up the cube")

It keeps one websocket open and records the round trip time of every call,
so a simulation run can report how long the link and the board took.
"""

from __future__ import annotations

import time

from websockets.sync.client import connect

from .protocol import DEFAULT_PORT, pack, unpack


class RemotePolicy:
    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout_s: float = 30.0):
        self.url = f"ws://{host}:{port}"
        self.ws = connect(self.url, max_size=None, open_timeout=timeout_s, compression=None)
        self.timeout_s = timeout_s
        self.round_trip_ms: list[float] = []
        self.server_ms: list[float] = []

    def _call(self, msg: dict) -> dict:
        self.ws.send(pack(msg))
        reply = unpack(self.ws.recv(timeout=self.timeout_s))
        if not reply.get("ok"):
            raise RuntimeError(f"policy server: {reply.get('error')}")
        return reply

    def info(self) -> dict:
        return self._call({"type": "info"})

    def reset(self) -> None:
        self._call({"type": "reset"})

    def infer(self, obs: dict, task: str):
        t = time.perf_counter()
        reply = self._call({"type": "infer", "obs": obs, "task": task})
        self.round_trip_ms.append((time.perf_counter() - t) * 1e3)
        self.server_ms.append(float(reply.get("server_ms", 0.0)))
        return reply["actions"]

    def close(self) -> None:
        self.ws.close()
