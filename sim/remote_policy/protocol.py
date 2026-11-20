"""Messages between the simulator (client) and the policy server.

Each message is one msgpack frame over a websocket. NumPy arrays are packed
with msgpack-numpy, so images go as raw uint8 bytes with no extra encoding.

Client to server:
    {"type": "info"}
    {"type": "reset"}
    {"type": "infer", "obs": {name: array, ...}, "task": str}

Server to client:
    {"ok": True, ...}                       for info and reset
    {"ok": True, "actions": array (T, D),   for infer: one action chunk
     "server_ms": float}                    time spent in the policy on the server
    {"ok": False, "error": str}             on any error
"""

from __future__ import annotations

import msgpack
import msgpack_numpy

DEFAULT_PORT = 8765


def pack(msg: dict) -> bytes:
    return msgpack.packb(msg, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode, raw=False)
