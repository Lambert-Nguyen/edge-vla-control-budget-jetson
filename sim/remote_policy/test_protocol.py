"""The link's messages: arrays round trip, and nothing in a message can run code.

    python -m pytest sim/remote_policy/test_protocol.py
"""

from __future__ import annotations

import pickle

import msgpack
import numpy as np
import pytest

from sim.remote_policy.protocol import UnsafeMessage, pack, unpack

RAN = []


class Payload:
    """Unpickling this records that code ran."""

    def __reduce__(self):
        return (RAN.append, ("ran",))


def test_observation_round_trip():
    rng = np.random.default_rng(0)
    obs = {"head": rng.integers(0, 255, (168, 224, 3), dtype=np.uint8),
           "state": rng.normal(size=14).astype(np.float32)}
    msg = unpack(pack({"type": "infer", "obs": obs, "task": "pick up the cube"}))
    assert msg["type"] == "infer" and msg["task"] == "pick up the cube"
    for k, v in obs.items():
        assert msg["obs"][k].dtype == v.dtype and np.array_equal(msg["obs"][k], v)


def test_reply_round_trip():
    actions = np.arange(40 * 16, dtype=np.float32).reshape(40, 16)
    msg = unpack(pack({"ok": True, "actions": actions, "server_ms": np.float64(12.5), "flag": np.bool_(True)}))
    assert np.array_equal(msg["actions"], actions) and msg["server_ms"] == 12.5 and msg["flag"]


def evil(array: dict) -> bytes:
    return msgpack.packb({"type": "infer", "obs": {"x": array}}, use_bin_type=True)


@pytest.mark.parametrize("array", [
    # what msgpack-numpy itself sends for an object array: it would unpickle data
    {b"nd": True, b"kind": b"O", b"type": "|O", b"shape": [1], b"data": pickle.dumps(Payload())},
    # the same, with the kind left out
    {b"nd": True, b"kind": b"", b"type": "|O", b"shape": [1], b"data": pickle.dumps(Payload())},
    # a structured array
    {b"nd": True, b"kind": b"V", b"type": [["a", "<f4"]], b"shape": [1], b"data": b"\0" * 4},
])
def test_unsafe_arrays_are_refused(array):
    with pytest.raises(UnsafeMessage):
        unpack(evil(array))
    assert RAN == []
