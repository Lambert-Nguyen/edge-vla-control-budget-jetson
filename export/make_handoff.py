#!/usr/bin/env python3
"""Make the handoff package for the engine builder (engine/build_engines.py).

The package is one folder that we copy to the Jetson. It holds:

- the backbone as standalone parts (backbone_p1.onnx ... with their .onnx.data).
  TensorRT 10.3 cannot build the whole backbone as one engine on the board, so
  the quantized graph is cut at the given language layers (export/split_backbone.py).
- the action expert graph (expert.onnx and expert.onnx.data).
- reference.npz with the reference cases.
- manifest.json with the inputs, outputs, precision and SHA-256 of every file.

The reference outputs come from ONNX Runtime (CPU) on the same graphs that are in
the package. So the board check (TR-3) only tests the TensorRT build. For each
case the file also holds the FP32 outputs (k_fp32, v_fp32, actions_fp32), so the
board can report how far the quantized model is from FP32.

    python export/make_handoff.py --backbone build/onnx/hyvla-bb-w8v/backbone.onnx --cuts 8 16 24 \\
        --expert build/onnx/hyvla-expert-rt/expert.onnx \\
        --cases build/onnx/hyvla-bb/backbone_reference.npz \\
        --states build/onnx/hyvla-expert-rt/expert_reference.npz \\
        --precision int8 --out build/handoff/hyvla-int8-v1

--cases is the FP32 backbone reference from export/export_backbone.py --reference.
It has the inputs and the FP32 k, v of each case. --states gives the robot state
of each case (from export/export_onnx.py --reference).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parent
DTYPES = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.FLOAT16: "float16", onnx.TensorProto.INT64: "int64",
          onnx.TensorProto.INT32: "int32", onnx.TensorProto.BOOL: "bool", onnx.TensorProto.INT8: "int8"}


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def io_spec(path: Path) -> dict:
    m = onnx.load(str(path), load_external_data=False)
    f = lambda v: [[d.dim_value for d in v.type.tensor_type.shape.dim], DTYPES[v.type.tensor_type.elem_type]]  # noqa: E731
    return {"inputs": {v.name: f(v) for v in m.graph.input}, "outputs": {v.name: f(v) for v in m.graph.output}}


def split(full: Path, cuts: list[int], out: Path) -> list[str]:
    """Cut the graph at each layer and save every part with its own weights file."""
    tmp, made = [], []
    rest = full
    for layer in cuts:
        subprocess.run([sys.executable, str(HERE / "split_backbone.py"), "--onnx", str(rest), "--layer", str(layer)],
                       check=True)
        p1 = rest.with_name(rest.stem + "_part1.onnx")
        rest = rest.with_name(rest.stem + "_part2.onnx")
        tmp += [p1, rest]
        made.append(p1)
    made.append(rest)
    names = []
    for i, p in enumerate(made, 1):
        name = f"backbone_p{i}.onnx"
        m = onnx.load(str(p))  # with the weights
        onnx.save(m, str(out / name), save_as_external_data=True, location=name + ".data", size_threshold=1024)
        del m
        names.append(name)
        print(f"  {name}: {(out / (name + '.data')).stat().st_size / 2**20:.0f} MiB of weights", flush=True)
    for p in tmp:
        p.unlink()
    return names


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_flow(expert, state, mask, k, v, noise, steps):
    """The Euler loop of Hy-VLA sample_actions: t goes from 1 to 0 in `steps` steps."""
    x, t, dt = noise.copy(), 1.0, -1.0 / steps
    v1 = None
    for _ in range(steps):
        vel = expert.run(None, {"state": state, "prefix_pad_mask": mask, "k": k, "v": v, "x_t": x,
                                "t": np.array([t], np.float32)})[0]
        v1 = vel if v1 is None else v1
        x = x + dt * vel
        t += dt
    return v1, x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, type=Path, help="the quantized (or FP16/FP32) backbone graph")
    ap.add_argument("--cuts", type=int, nargs="*", default=[8, 16, 24], help="first language layer of each new part")
    ap.add_argument("--expert", required=True, type=Path)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--precision", default="int8", help="precision of the backbone parts (int8, fp16, int4_weight)")
    ap.add_argument("--steps", type=int, default=10, help="Euler steps (Hy-VLA num_steps)")
    ap.add_argument("--info", nargs="*", default=[], metavar="KEY=VALUE", help="notes for the manifest")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    t0 = time.time()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists. Use a new name for a new package.")
    args.out.mkdir(parents=True)

    print("backbone parts:", flush=True)
    parts = split(args.backbone.resolve(), args.cuts, args.out)
    link_or_copy(args.expert, args.out / "expert.onnx")
    link_or_copy(args.expert.with_suffix(".onnx.data"), args.out / "expert.onnx.data")

    cases = dict(np.load(args.cases))
    states = dict(np.load(args.states))
    names = sorted({k.split("/")[0] for k in cases if k.endswith("/k")})
    bb_in = io_spec(args.out / parts[0])["inputs"]
    pools = {c: {n: cases[f"{c}/{n}"] for n in bb_in} for c in names}
    print(f"reference: {len(names)} cases, ONNX Runtime on the package graphs", flush=True)
    for p in parts:  # one part at a time, to keep the RAM low
        s = session(args.out / p)
        outs = [o.name for o in s.get_outputs()]
        for c in names:
            r = s.run(None, {i.name: pools[c][i.name] for i in s.get_inputs()})
            pools[c].update(zip(outs, r))
        del s
    expert = session(args.out / "expert.onnx")
    rng = np.random.default_rng(0)
    ref = {}
    cos = lambda a, b: float((a.astype(np.float64) * b).sum() / np.sqrt((a.astype(np.float64) ** 2).sum()  # noqa: E731
                                                                        * (b.astype(np.float64) ** 2).sum()))
    for i, c in enumerate(names):
        state = states[f"case{i:02d}/state"]
        noise = rng.standard_normal((1, 40, 32)).astype(np.float32)
        p = pools[c]
        v1, acts = run_flow(expert, state, p["prefix_pad_mask"], p["k"], p["v"], noise, args.steps)
        _, acts32 = run_flow(expert, state, cases[f"{c}/prefix_pad_mask"], cases[f"{c}/k"], cases[f"{c}/v"],
                             noise, args.steps)
        for n in bb_in:
            ref[f"{c}/{n}"] = cases[f"{c}/{n}"]
        ref.update({f"{c}/state": state, f"{c}/noise": noise, f"{c}/k": p["k"], f"{c}/v": p["v"],
                    f"{c}/prefix_pad_mask": p["prefix_pad_mask"], f"{c}/velocity_t1": v1, f"{c}/actions": acts,
                    f"{c}/k_fp32": cases[f"{c}/k"], f"{c}/v_fp32": cases[f"{c}/v"], f"{c}/actions_fp32": acts32})
        print(f"  {c}: vs FP32 k cosine {cos(p['k'], cases[f'{c}/k']):.4f}, v {cos(p['v'], cases[f'{c}/v']):.4f}, "
              f"actions {cos(acts, acts32):.4f}", flush=True)
    np.savez(args.out / "reference.npz", **ref)

    graphs = {}
    for i, p in enumerate(parts, 1):
        graphs[f"backbone_p{i}"] = {"file": p, "data": p + ".data", **io_spec(args.out / p)}
    full = io_spec(args.backbone)
    graphs["backbone"] = {"parts": [f"backbone_p{i}" for i in range(1, len(parts) + 1)], **full}
    graphs["expert"] = {"file": "expert.onnx", "data": "expert.onnx.data", **io_spec(args.out / "expert.onnx")}
    files = [f for g in graphs.values() for f in (g.get("file"), g.get("data")) if f] + ["reference.npz"]
    precision = {f"backbone_p{i}": args.precision for i in range(1, len(parts) + 1)}
    precision.update({"backbone": args.precision, "expert": "fp16"})
    try:
        commit = subprocess.run(["git", "-C", str(HERE), "describe", "--always", "--dirty"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    k_shape = full["outputs"]["k"][0]
    manifest = {
        "name": args.out.name,
        "model": "Hy-VLA (Hy-Embodied-0.5-VLA), RoboTwin checkpoint",
        "opset": [o.version for o in onnx.load(str(args.backbone), load_external_data=False).opset_import][0],
        "precision": precision,
        "graphs": graphs,
        "flow": {"steps_default": args.steps, "chunk": 40, "action_dim": 32, "kv_shape": k_shape,
                 "context": {"k": "k", "v": "v", "prefix_pad_mask": "prefix_pad_mask"}},
        "reference": {"file": "reference.npz", "cases": len(names), "noise_seed": 0,
                      "source": "ONNX Runtime CPU on the package graphs. *_fp32 keys are the FP32 model."},
        "export_commit": commit,
        "export_info": dict(kv.split("=", 1) for kv in args.info),
        "sha256": {},
    }
    print("hashing ...", flush=True)
    manifest["sha256"] = {f: sha256(args.out / f) for f in files}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum((args.out / f).stat().st_size for f in files) / 2**30
    print(f"wrote {args.out}: {len(files)} files, {total:.2f} GiB, in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
