#!/usr/bin/env python3
"""INT8 version of an exported Hy-VLA graph: INT8 weights and INT8 activations (W8A8).

TensorRT 10.3 on the Orin runs INT4 weights by unpacking them to FP16 in GPU
memory, so the INT4 backbone needs 5.3 GB of scratch (export/NOTES.md). With
INT8 on both inputs of a MatMul, TensorRT can use its INT8 kernels instead.

Each MatMul or Gemm weight gets one INT8 scale per output channel. The
activation that goes into it gets one scale per tensor, from the largest |x| seen
in calibration runs with ONNX Runtime. Both are stored as QuantizeLinear and
DequantizeLinear nodes (explicit quantization), which TensorRT builds with
--int8 --fp16. Other weights stay as they are.

    python export/quantize_int8.py --onnx build/onnx/hyvla-bb/backbone_part2_part1.onnx \\
        --calib build/onnx/hyvla-bb/chunk_L8_inputs.npz --out build/onnx/hyvla-bb-w8/chunk_L8.onnx

The calibration file holds the graph inputs for each case (case00/<input>, ...).
This is max calibration, the simplest method. It is a first point to compare with
INT4, not the final INT8 calibration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_int4 import DataWriter, load_array  # noqa: E402


def plan_weights(m, inits, users, min_size):
    """MatMul/Gemm nodes with a 2-D weight: weight name -> (node, output-channel axis)."""
    plan = {}
    for n in m.graph.node:
        if n.op_type not in ("MatMul", "Gemm") or len(n.input) < 2 or n.input[1] not in inits:
            continue
        w = inits[n.input[1]]
        if len(w.dims) != 2 or len(users.get(n.input[1], [])) != 1 or w.dims[0] * w.dims[1] < min_size:
            continue
        trans_b = n.op_type == "Gemm" and any(a.name == "transB" and a.i == 1 for a in n.attribute)
        plan[n.input[1]] = (n, 0 if trans_b else 1)
    return plan


def calibrate(path: Path, acts: list[str], calib: dict, cases: list[str]) -> dict[str, float]:
    import onnxruntime as ort

    m = onnx.load(str(path), load_external_data=False)
    have = {o.name for o in m.graph.output}
    for a in acts:
        if a not in have:
            m.graph.output.append(helper.make_tensor_value_info(a, TensorProto.FLOAT, None))
    tmp = path.with_name(path.stem + "_calib.onnx")
    onnx.save(m, str(tmp))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in s.get_inputs()]
    outs = [o.name for o in s.get_outputs()]
    amax = {a: 0.0 for a in acts}
    for c in cases:
        r = dict(zip(outs, s.run(None, {n: calib[f"{c}/{n}"] for n in names})))
        for a in acts:
            amax[a] = max(amax[a], float(np.abs(r[a]).max()))
    tmp.unlink()
    return amax


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-size", type=int, default=1 << 20)
    args = ap.parse_args()

    t0 = time.time()
    base = args.onnx.resolve().parent
    m = onnx.load(str(args.onnx), load_external_data=False)
    inits = {t.name: t for t in m.graph.initializer}
    users: dict[str, list] = {}
    for n in m.graph.node:
        for x in n.input:
            users.setdefault(x, []).append(n)
    plan = plan_weights(m, inits, users, args.min_size)
    acts = sorted({n.input[0] for n, _ in plan.values()})
    calib = dict(np.load(args.calib))
    cases = sorted({k.split("/")[0] for k in calib})
    amax = calibrate(args.onnx, acts, calib, cases)
    print(f"{len(plan)} weights to INT8, {len(acts)} activations calibrated on {len(cases)} cases", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = DataWriter(args.out.with_suffix(".onnx.data"))
    new_nodes, new_inits = [], []
    before = after = 0
    for t in list(m.graph.initializer):
        a = load_array(t, base)
        before += a.nbytes
        if t.name in plan:
            _, axis = plan[t.name]
            w = a.astype(np.float32)
            scale = np.abs(w).max(axis=1 - axis) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.round(w / (scale[None, :] if axis == 1 else scale[:, None])), -127, 127).astype(np.int8)
            qt = numpy_helper.from_array(q, t.name + "_int8")
            raw = qt.raw_data
            writer.put(qt, raw)
            st = numpy_helper.from_array(scale.astype(np.float32), t.name + "_scale")
            # ONNX Runtime's fused INT8 Gemm wants a zero point with the same shape as the scale.
            zt = numpy_helper.from_array(np.zeros(scale.shape, np.int8), t.name + "_zero")
            after += len(raw) + len(st.raw_data)
            new_inits += [qt, st, zt]
            new_nodes.append(helper.make_node("DequantizeLinear", [qt.name, st.name, zt.name], [t.name],
                                              name=t.name + "_dq", axis=axis))
            m.graph.initializer.remove(t)
        else:
            raw = numpy_helper.from_array(a, t.name).raw_data
            if len(raw) < 64 * 1024:
                t.raw_data = raw
            else:
                writer.put(t, raw)
            after += len(raw)
        del a
    writer.close()

    # Activations: x -> QuantizeLinear -> DequantizeLinear -> MatMul.
    for x in acts:
        s = max(amax[x], 1e-8) / 127.0
        sn, zn = x + "_act_scale", x + "_act_zero"
        new_inits += [numpy_helper.from_array(np.array(s, np.float32), sn),
                      numpy_helper.from_array(np.array(0, np.int8), zn)]
        q, dq = x + "_q", x + "_dq"
        new_nodes += [helper.make_node("QuantizeLinear", [x, sn, zn], [q], name=x + "_quant"),
                      helper.make_node("DequantizeLinear", [q, sn, zn], [dq], name=x + "_dequant")]
        for n, _ in plan.values():
            if n.input[0] == x:
                n.input[0] = dq

    m.graph.initializer.extend(new_inits)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    # Weight DQ nodes first. Activation Q/DQ right after the node that makes x.
    wdq = [n for n in new_nodes if n.op_type == "DequantizeLinear" and n.input[0].endswith("_int8")]
    act = {n.input[0]: [] for n in new_nodes if n.op_type == "QuantizeLinear"}
    for n in new_nodes:
        if n.op_type == "QuantizeLinear":
            act[n.input[0]].append(n)
        elif n.op_type == "DequantizeLinear" and not n.input[0].endswith("_int8"):
            act[n.input[0][: -len("_q")]].append(n)
    graph_inputs = {i.name for i in m.graph.input}
    order = wdq + [n for x in act if x in graph_inputs for n in act[x]]
    for n in nodes:
        order.append(n)
        for o in n.output:
            order += act.get(o, [])
    m.graph.node.extend(order)
    onnx.save(m, str(args.out))
    print(f"weights {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB, in {time.time() - t0:.0f} s")
    print(f"wrote {args.out} and {writer.path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
