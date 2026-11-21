"""Messages between the simulator (client) and the policy server.

Each message is one msgpack frame over a websocket. NumPy arrays are packed
in msgpack-numpy's format, so images go as raw uint8 bytes with no extra
encoding.

Client to server:
    {"type": "info"}
    {"type": "reset"}
    {"type": "infer", "obs": {name: array, ...}, "task": str}

Server to client:
    {"ok": True, ...}                       for info and reset
    {"ok": True, "actions": array (T, D),   for infer: one action chunk
     "server_ms": float}                    time spent in the policy on the server
    {"ok": False, "error": str}             on any error

Only plain numeric arrays are accepted. msgpack-numpy's own decoder unpickles
object arrays, and the server listens on the network without a password, so
anyone who can reach its port could run code on the Jetson. unpack() refuses
object and structured dtypes before msgpack-numpy sees them.
"""

from __future__ import annotations

import msgpack
import msgpack_numpy
import numpy as np

DEFAULT_PORT = 8765


class UnsafeMessage(ValueError):
    """A message with an array type that could run code when decoded."""


def pack(msg: dict) -> bytes:
    return msgpack.packb(msg, default=msgpack_numpy.encode, use_bin_type=True)


def _decode(obj: dict):
    if b"nd" in obj:
        kind = obj.get(b"kind", b"")
        if kind not in (b"", ""):
            raise UnsafeMessage(f"array kind {kind!r} is not allowed")
        type_ = obj.get(b"type")
        try:
            dtype = np.dtype(type_.decode() if isinstance(type_, bytes) else type_)
        except TypeError as e:
            raise UnsafeMessage(f"bad array type {type_!r}") from e
        if dtype.hasobject or dtype.kind not in "biufc":
            raise UnsafeMessage(f"array type {dtype} is not allowed")
    return msgpack_numpy.decode(obj)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=_decode, raw=False)
