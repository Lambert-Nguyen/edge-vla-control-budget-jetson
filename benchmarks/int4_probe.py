#!/usr/bin/env python3
"""Does TensorRT 10.3 on the Orin run INT4-weight layers, and is it faster?

Risk R-2 of the export and engine builder design specs. Lam's main path is an
INT4-weight backbone with Q/DQ nodes in the ONNX graph. Before that export
exists, this probe checks the board side on a small graph with the shape of
Hy-VLA's language layers (hidden 2048, MLP 6144, SiLU gate):

- fp16: the MLP blocks with FP16 weights.
- int4: the same blocks, weights stored as INT4 with one FP16 scale per block
  of 128 (DequantizeLinear, opset 21, block quantization), FP16 activations.

Both graphs are built with trtexec on the board. The probe reports whether
the build works, the engine size, the GPU time, and the output error of each
engine against a NumPy reference of the same (dequantized) weights. So the
error is the build's error, not the quantization's.

    python3 benchmarks/int4_probe.py --tokens 64 --blocks 4 --json results/<name>.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from hello_latency import board_state, git_state  # noqa: E402

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
HIDDEN, MLP, BLOCK = 2048, 6144, 128


def quant_int4(w, block: int):
    """Symmetric INT4 per block of `block` rows (the K axis). Returns q, scale, dequantized."""
    import numpy as np

    k, n = w.shape
    wb = w.reshape(k // block, block, n)
    scale = np.abs(wb).max(axis=1) / 7.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(wb / scale[:, None, :]), -8, 7).astype(np.int8)
    deq = (q * scale[:, None, :]).reshape(k, n)
    return q.reshape(k, n), scale.astype(np.float16), deq.astype(np.float16)


def build_graph(kind: str, weights: list, tokens: int, path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits = [], []
    x = "x"
    for b, (wg, wu, wd) in enumerate(weights):
        names = {}
        for tag, w in (("gate", wg), ("up", wu), ("down", wd)):
            name = f"b{b}_{tag}"
            if kind == "fp16":
                inits.append(numpy_helper.from_array(w["deq"], name))
                names[tag] = name
            else:
                q = TensorProto(name=name + "_q", data_type=TensorProto.INT4, dims=list(w["q"].shape))
                nib = (w["q"].reshape(-1).astype(np.int16) & 0xF).astype(np.uint8)
                if nib.size % 2:
                    nib = np.append(nib, 0)
                q.raw_data = (nib[0::2] | (nib[1::2] << 4)).tobytes()  # two values per byte, low first
                inits.append(q)
                inits.append(numpy_helper.from_array(w["scale"], name + "_s"))
                nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_s"], [name],
                                              axis=0, block_size=BLOCK))
                names[tag] = name
        nodes += [
            helper.make_node("MatMul", [x, names["gate"]], [f"b{b}_g"]),
            helper.make_node("Sigmoid", [f"b{b}_g"], [f"b{b}_sg"]),
            helper.make_node("Mul", [f"b{b}_g", f"b{b}_sg"], [f"b{b}_silu"]),
            helper.make_node("MatMul", [x, names["up"]], [f"b{b}_u"]),
            helper.make_node("Mul", [f"b{b}_silu", f"b{b}_u"], [f"b{b}_h"]),
            helper.make_node("MatMul", [f"b{b}_h", names["down"]], [f"b{b}_d"]),
            helper.make_node("Add", [x, f"b{b}_d"], [f"b{b}_out"]),
        ]
        x = f"b{b}_out"
    nodes.append(helper.make_node("Identity", [x], ["y"]))
    graph = helper.make_graph(
        nodes, f"mlp_{kind}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, tokens, HIDDEN])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, tokens, HIDDEN])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def reference(weights: list, x):
    """FP32 NumPy forward pass with the dequantized weights."""
    import numpy as np

    x = x.astype(np.float32)
    for wg, wu, wd in weights:
        g = x @ wg["deq"].astype(np.float32)
        u = x @ wu["deq"].astype(np.float32)
        h = g / (1 + np.exp(-g)) * u
        x = x + h @ wd["deq"].astype(np.float32)
    return x


def trtexec(onnx_path: Path, engine: Path, flags: list[str], x_path: Path, out_json: Path) -> dict:
    cmd = [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={engine}", *flags,
           f"--loadInputs=x:{x_path}", f"--exportOutput={out_json}",
           "--warmUp=500", "--duration=5", "--useSpinWait"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    gpu = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, "
                    r"median = ([\d.]+) ms, percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, "
                    r"percentile\(99%\) = ([\d.]+) ms", log)
    res = {"ok": p.returncode == 0 and engine.exists(), "flags": flags}
    if gpu:
        res.update(gpu_mean_ms=float(gpu.group(3)), gpu_median_ms=float(gpu.group(4)),
                   gpu_p99_ms=float(gpu.group(7)))
    if engine.exists():
        res["engine_mb"] = engine.stat().st_size / 1e6
    if not res["ok"]:
        res["error"] = "\n".join(line for line in log.splitlines() if "Error" in line or "ERROR" in line)[-1500:]
    return res


def read_output(out_json: Path):
    import numpy as np

    for item in json.loads(out_json.read_text()):
        if item["name"] == "y":
            dims = [int(d) for d in item["dimensions"].split("x")]
            return np.array(item["values"], dtype=np.float32).reshape(dims)
    raise KeyError("y")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=64, help="sequence length (default: %(default)s)")
    ap.add_argument("--blocks", type=int, default=4, help="MLP blocks (default: %(default)s)")
    ap.add_argument("--keep", type=Path, help="keep the ONNX files and engines here")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import numpy as np

    rng = np.random.default_rng(0)
    weights = []
    for _ in range(args.blocks):
        trio = []
        for k, n in ((HIDDEN, MLP), (HIDDEN, MLP), (MLP, HIDDEN)):
            w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
            q, s, deq = quant_int4(w, BLOCK)
            trio.append({"q": q, "scale": s, "deq": deq})
        weights.append(trio)
    x = (rng.standard_normal((1, args.tokens, HIDDEN)) * 0.5).astype(np.float16)
    ref = reference(weights, x)

    work = args.keep or Path(tempfile.mkdtemp(prefix="int4_probe_"))
    work.mkdir(parents=True, exist_ok=True)
    x_path = work / "x.bin"
    x.tofile(x_path)
    runs = {}
    for kind, flag_sets in (("fp16", [["--fp16"]]), ("int4", [["--stronglyTyped"], ["--fp16", "--int4"]])):
        onnx_path = work / f"mlp_{kind}.onnx"
        build_graph(kind, weights, args.tokens, onnx_path)
        for flags in flag_sets:
            tag = kind + "".join(f.replace("--", "_") for f in flags)
            r = trtexec(onnx_path, work / f"{tag}.engine", flags, x_path, work / f"{tag}.json")
            if r["ok"]:
                y = read_output(work / f"{tag}.json")
                err = np.abs(y - ref)
                r["max_abs_err"] = float(err.max())
                r["rel_err"] = float(err.max() / np.abs(ref).max())
            runs[tag] = r

    weight_mb_fp16 = sum(w["deq"].nbytes for trio in weights for w in trio) / 1e6
    print("=" * 72)
    print(f"INT4 probe: {args.blocks} MLP blocks {HIDDEN}->{MLP}->{HIDDEN}, {args.tokens} tokens, "
          f"FP16 weights {weight_mb_fp16:.0f} MB")
    print("=" * 72)
    for tag, r in runs.items():
        if r["ok"]:
            print(f"  {tag:22s} built  engine {r.get('engine_mb', 0):6.1f} MB   gpu median "
                  f"{r.get('gpu_median_ms', float('nan')):6.3f} ms   p99 {r.get('gpu_p99_ms', float('nan')):6.3f} ms   "
                  f"max err {r['rel_err'] * 100:.3f}% of max")
        else:
            print(f"  {tag:22s} FAILED\n{r.get('error', '')}")
    board = board_state()
    print(f"  board {board['nvpmodel_mode']}, clocks {'pinned' if board['clocks'].get('jetson_clocks_active') else 'scaling'}")

    if args.json:
        report = {"kind": "trt_int4_probe", "timestamp": datetime.now(timezone.utc).isoformat(),
                  "config": {"tokens": args.tokens, "blocks": args.blocks, "hidden": HIDDEN, "mlp": MLP,
                             "block_size": BLOCK},
                  "weight_mb_fp16": weight_mb_fp16, "runs": runs, "board": board, "git": git_state()}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
