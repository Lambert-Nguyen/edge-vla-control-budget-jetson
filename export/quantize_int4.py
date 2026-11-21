#!/usr/bin/env python3
"""INT4-weight version of an exported Hy-VLA graph (round to nearest, block 128).

Each MatMul or Gemm weight becomes INT4 with one scale per block of 128 values
along the input dimension, stored as a DequantizeLinear node (opset 21, block
quantization). This is the format that TensorRT 10.3 builds on the Jetson
(benchmarks/int4_probe.py). Activations stay as they are. Weights that reach a
MatMul through a Transpose (the vision tower) are quantized too. Other large
weights (the token embedding table, and weights whose input size is not a
multiple of 128) are stored in FP16. Small weights keep their type.

It handles one weight at a time and streams the weights file, so a 15 GB graph
needs only a little RAM. (ModelOpt ran out of 30 GB of RAM on the backbone.)

    python export/quantize_int4.py --onnx build/onnx/hyvla-bb/backbone.onnx \\
        --out build/onnx/hyvla-bb-w4/backbone.onnx

This is round to nearest (RTN), the simplest method and the first quantized
point. AWQ-style scaling with calibration frames is the next step (export spec).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BLOCK = 128
ALIGN = 4096


class DataWriter:
    """Writes tensors to one external data file and points the TensorProto at it."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "wb")
        self.offset = 0

    def put(self, t: TensorProto, raw: bytes) -> None:
        pad = (-self.offset) % ALIGN
        if pad:
            self.f.write(b"\0" * pad)
            self.offset += pad
        self.f.write(raw)
        t.ClearField("raw_data")
        t.data_location = TensorProto.EXTERNAL
        del t.external_data[:]
        for k, v in (("location", self.path.name), ("offset", str(self.offset)), ("length", str(len(raw)))):
            e = t.external_data.add()
            e.key, e.value = k, v
        self.offset += len(raw)

    def close(self) -> None:
        self.f.close()


def load_array(t: TensorProto, base: Path) -> np.ndarray:
    if t.data_location == TensorProto.EXTERNAL:
        info = {e.key: e.value for e in t.external_data}
        with open(base / info["location"], "rb") as f:
            f.seek(int(info.get("offset", 0)))
            raw = f.read(int(info["length"])) if "length" in info else f.read()
        del t.external_data[:]
        t.data_location = TensorProto.DEFAULT
        t.raw_data = raw
    a = numpy_helper.to_array(t)
    t.ClearField("raw_data")
    return a


def quant_int4(w: np.ndarray, axis: int):
    """Symmetric INT4 per block of BLOCK along `axis` (the input dimension). Returns q, scale."""
    w = np.moveaxis(w.astype(np.float32), axis, 0)  # (K, N)
    k, n = w.shape
    blocks = w.reshape(k // BLOCK, BLOCK, n)
    scale = np.abs(blocks).max(axis=1) / 7.0  # (K/B, N)
    scale[scale == 0] = 1.0
    q = np.clip(np.round(blocks / scale[:, None, :]), -8, 7).astype(np.int8).reshape(k, n)
    return np.moveaxis(q, 0, axis), np.moveaxis(scale, 0, axis)


def pack_int4(q: np.ndarray) -> bytes:
    nib = (q.reshape(-1).astype(np.int16) & 0xF).astype(np.uint8)
    if nib.size % 2:
        nib = np.append(nib, 0)
    return (nib[0::2] | (nib[1::2] << 4)).tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-size", type=int, default=1 << 20, help="leave weights with fewer values in FP32")
    ap.add_argument("--block", type=int, default=128, choices=(32, 64, 128),
                    help="values per scale along the input dimension (default: %(default)s)")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="TEXT",
                    help="keep weights whose name contains TEXT in FP16, not INT4 (e.g. visual)")
    args = ap.parse_args()
    global BLOCK
    BLOCK = args.block

    t0 = time.time()
    base = args.onnx.resolve().parent
    m = onnx.load(str(args.onnx), load_external_data=False)
    inits = {t.name: t for t in m.graph.initializer}
    users: dict[str, list] = {}
    for n in m.graph.node:
        for i, name in enumerate(n.input):
            users.setdefault(name, []).append((n, i))

    # Which weights become INT4, and along which axis is the input dimension K.
    plan = {}
    for n in m.graph.node:
        if n.op_type not in ("MatMul", "Gemm") or len(n.input) < 2 or n.input[1] not in inits:
            continue
        w = inits[n.input[1]]
        if len(w.dims) != 2 or len(users.get(n.input[1], [])) != 1:
            continue
        trans_b = n.op_type == "Gemm" and any(a.name == "transB" and a.i == 1 for a in n.attribute)
        axis = 1 if trans_b else 0
        if w.dims[axis] % BLOCK or w.dims[0] * w.dims[1] < args.min_size:
            continue
        plan[n.input[1]] = axis
    # Weight -> Transpose(perm 1, 0) -> MatMul, as in the vision tower. The stored
    # weight is (N, K), so the input dimension K is axis 1.
    for name, t in inits.items():
        u = users.get(name, [])
        if len(t.dims) != 2 or len(u) != 1 or u[0][0].op_type != "Transpose" or name in plan:
            continue
        tr = u[0][0]
        perm = [list(a.ints) for a in tr.attribute if a.name == "perm"]
        perm = perm[0] if perm else [1, 0]  # no perm attribute means reverse the axes
        nxt = users.get(tr.output[0], [])
        # The vision tower runs once per camera, so one weight can feed several MatMuls.
        if perm != [1, 0] or not nxt or any(op.op_type != "MatMul" or i != 1 for op, i in nxt):
            continue
        if t.dims[1] % BLOCK or t.dims[0] * t.dims[1] < args.min_size:
            continue
        plan[name] = 1
    # Other large FP32 weights (embedding tables, weights whose K is not a multiple of
    # BLOCK) are stored in FP16 with a Cast back to FP32.
    for name in [n for n in plan if any(x in n for x in args.exclude)]:
        del plan[name]
    gathers = {name for name, t in inits.items() if name not in plan and t.data_type == TensorProto.FLOAT
               and int(np.prod(t.dims or [1])) >= args.min_size}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = DataWriter(args.out.with_suffix(".onnx.data"))
    new_nodes, new_inits = [], []
    before = after = 0
    for t in list(m.graph.initializer):
        a = load_array(t, base)
        before += a.nbytes
        if t.name in plan:
            axis = plan[t.name]
            q, scale = quant_int4(a, axis)
            qt = TensorProto(name=t.name + "_int4", data_type=TensorProto.INT4, dims=list(q.shape))
            raw = pack_int4(q)
            writer.put(qt, raw)
            st = numpy_helper.from_array(scale.astype(np.float32), t.name + "_scale")
            sraw = st.raw_data
            writer.put(st, sraw)
            after += len(raw) + len(sraw)
            new_inits += [qt, st]
            new_nodes.append(helper.make_node("DequantizeLinear", [qt.name, st.name], [t.name],
                                              name=t.name + "_dq", axis=axis, block_size=BLOCK))
            m.graph.initializer.remove(t)
        elif t.name in gathers:
            h = numpy_helper.from_array(a.astype(np.float16), t.name + "_fp16")
            raw = h.raw_data
            writer.put(h, raw)
            after += len(raw)
            new_inits.append(h)
            new_nodes.append(helper.make_node("Cast", [h.name], [t.name], name=t.name + "_cast", to=TensorProto.FLOAT))
            m.graph.initializer.remove(t)
        else:
            raw = numpy_helper.from_array(a, t.name).raw_data
            if len(raw) < 64 * 1024:  # small constants (shapes, axes) stay in the graph file
                t.raw_data = raw
            else:
                writer.put(t, raw)
            after += len(raw)
        del a
    writer.close()
    m.graph.initializer.extend(new_inits)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes + nodes)  # weights first, so the graph stays in order
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = max(op.version, 21)
    if m.ir_version < 10:
        m.ir_version = 10
    onnx.save(m, str(args.out))
    print(f"{len(plan)} weights to INT4 (block {BLOCK}), {len(gathers)} other large weights to FP16")
    print(f"weights {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB, in {time.time() - t0:.0f} s")
    print(f"wrote {args.out} and {writer.path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
