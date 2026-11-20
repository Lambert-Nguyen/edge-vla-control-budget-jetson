#!/usr/bin/env python3
"""Time the remote policy link from the workstation.

Sends observations like the simulator will (camera images and a joint state)
to the policy server on the Jetson, and reports the round trip time, the time
the policy took on the board, and the rest, which is the network and packing.

    python3 sim/remote_policy/link_test.py --host sjsujetson-36.local --cameras 3 --size 480x640
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from sim.remote_policy.client import RemotePolicy  # noqa: E402


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(p / 100 * len(s) + 0.5) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cameras", type=int, default=2)
    ap.add_argument("--size", default="480x640", help="image HxW (default: %(default)s)")
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import numpy as np

    h, w = (int(x) for x in args.size.split("x"))
    rng = np.random.default_rng(0)
    obs = {f"cam{i}": rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for i in range(args.cameras)}
    obs["state"] = np.zeros(args.state_dim, np.float32)
    mb = sum(v.nbytes for v in obs.values()) / 1e6

    policy = RemotePolicy(args.host, args.port)
    info = policy.info()
    policy.reset()
    for _ in range(3):
        policy.infer(obs, "warm up")
    policy.round_trip_ms.clear()
    policy.server_ms.clear()
    actions = None
    for _ in range(args.iters):
        actions = policy.infer(obs, "Pick up the cube and put it in the box.")
    policy.close()

    rt, sv = policy.round_trip_ms, policy.server_ms
    net = [a - b for a, b in zip(rt, sv)]
    print(f"server: {info}")
    print(f"observation {mb:.2f} MB ({args.cameras} x {h}x{w} images), reply {actions.shape} {actions.dtype}")
    for name, xs in (("round trip", rt), ("on the board", sv), ("link + packing", net)):
        print(f"  {name:15s} p50 {pct(xs, 50):7.1f} ms   p99 {pct(xs, 99):7.1f} ms   mean {statistics.fmean(xs):7.1f} ms")
    if args.json:
        args.json.write_text(json.dumps({"server": info, "obs_mb": mb, "round_trip_ms": rt,
                                         "server_ms": sv, "link_ms": net}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
