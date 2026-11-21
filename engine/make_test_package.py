#!/usr/bin/env python3
"""Make a small fake handoff package with the same interface as the Hy-VLA one.

Until Lam's real export exists, this lets the engine builder run its whole
path on the board: the manifest and hashes, the engine cache, the build, the
2% reference check and the Euler loop. The graphs are tiny, but their inputs
and outputs have the names and roles of the real ones:

- backbone: images, img_masks, lang_tokens, lang_masks -> k, v, prefix_pad_mask
- expert: state, prefix_pad_mask, k, v, x_t, t -> velocity (one Euler step)

The reference outputs are computed in NumPy (FP32) from the same weights.

    python3 engine/make_test_package.py --out build/handoff/test-fp16
    python3 engine/make_test_package.py --out build/handoff/test-w4 --int4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trt_runtime import sha256  # noqa: E402

CAMS, RES, PATCH = 3, 32, 8          # 3 cameras, 32 px, 8 px patches -> 16 tokens per camera
TOKENS, VOCAB = 16, 64               # task tokens
D, LAYERS, KV_HEADS, HEAD = 128, 2, 2, 64
STATE, H, A = 32, 40, 32             # state size, chunk, action size (padded, like Hy-VLA)
PREFIX = CAMS * (RES // PATCH) ** 2 + TOKENS
BLOCK = 64


def silu(x):
    return x / (1 + np.exp(-x))


def make_weights(rng):
    w = lambda *s: (rng.standard_normal(s) / np.sqrt(s[0])).astype(np.float32)  # noqa: E731
    return {"patch": w(3 * PATCH * PATCH, D), "emb": (rng.standard_normal((VOCAB, D)) * 0.5).astype(np.float32),
            "wk": w(D, LAYERS * KV_HEADS * HEAD), "wv": w(D, LAYERS * KV_HEADS * HEAD),
            "wq": w(A, KV_HEADS * HEAD), "ws": w(STATE, KV_HEADS * HEAD), "wt": w(1, KV_HEADS * HEAD),
            "wo": w(KV_HEADS * HEAD, A)}


def backbone_np(W, images, img_masks, lang_tokens, lang_masks):
    p = images.reshape(1, CAMS, 3, RES // PATCH, PATCH, RES // PATCH, PATCH)
    p = p.transpose(0, 1, 3, 5, 2, 4, 6).reshape(1, CAMS * (RES // PATCH) ** 2, 3 * PATCH * PATCH)
    vis = p @ W["patch"]
    txt = W["emb"][lang_tokens]
    x = np.concatenate([vis, txt], axis=1)
    pad = np.concatenate([np.repeat(img_masks, (RES // PATCH) ** 2, axis=1), lang_masks], axis=1)
    k = (x @ W["wk"]).reshape(1, PREFIX, LAYERS, KV_HEADS, HEAD).transpose(2, 0, 3, 1, 4)
    v = (x @ W["wv"]).reshape(1, PREFIX, LAYERS, KV_HEADS, HEAD).transpose(2, 0, 3, 1, 4)
    return k, v, pad


def expert_np(W, state, pad, k, v, x_t, t):
    q = x_t @ W["wq"] + (state @ W["ws"])[:, None, :] + t[:, None, None] * W["wt"][None]
    q = q.reshape(1, H, KV_HEADS, HEAD).transpose(0, 2, 1, 3)
    km, vm = k.mean(axis=0), v.mean(axis=0)                                  # mean over layers
    s = q @ km.transpose(0, 1, 3, 2) / np.sqrt(HEAD)
    s = np.where(pad[:, None, None, :], s, -1e4)
    s = np.exp(s - s.max(-1, keepdims=True))
    s = s / s.sum(-1, keepdims=True)
    o = (s @ vm).transpose(0, 2, 1, 3).reshape(1, H, KV_HEADS * HEAD)
    return silu(o) @ W["wo"]


def build_onnx(W, out: Path, int4: bool) -> dict:
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh

    def weight(name, arr, nodes, inits, quant):
        if not quant:
            inits.append(nh.from_array(arr.astype(np.float16), name))
            return name
        k, n = arr.shape
        blocks = arr.reshape(k // BLOCK, BLOCK, n)
        scale = np.abs(blocks).max(axis=1) / 7.0
        q = np.clip(np.round(blocks / scale[:, None, :]), -8, 7).astype(np.int8).reshape(k, n)
        W[name + "_deq"] = (q.reshape(k // BLOCK, BLOCK, n) * scale[:, None, :]).reshape(k, n)
        t = T(name=name + "_q", data_type=T.INT4, dims=[k, n])
        nib = (q.reshape(-1).astype(np.int16) & 0xF).astype(np.uint8)
        t.raw_data = (nib[0::2] | (nib[1::2] << 4)).tobytes()
        inits += [t, nh.from_array(scale.astype(np.float16), name + "_s")]
        nodes.append(h.make_node("DequantizeLinear", [name + "_q", name + "_s"], [name], axis=0, block_size=BLOCK))
        return name

    def const(name, arr, inits):
        inits.append(nh.from_array(np.asarray(arr), name))
        return name

    # ---- backbone ----
    n, i = [], []
    n += [h.make_node("Cast", ["images"], ["img16"], to=T.FLOAT16),
          h.make_node("Reshape", ["img16", const("s1", np.array([1, CAMS, 3, RES // PATCH, PATCH, RES // PATCH, PATCH]), i)], ["p0"]),
          h.make_node("Transpose", ["p0"], ["p1"], perm=[0, 1, 3, 5, 2, 4, 6]),
          h.make_node("Reshape", ["p1", const("s2", np.array([1, CAMS * (RES // PATCH) ** 2, 3 * PATCH * PATCH]), i)], ["p2"]),
          h.make_node("MatMul", ["p2", weight("patch", W["patch"], n, i, int4)], ["vis"]),
          h.make_node("Gather", [weight("emb", W["emb"], n, i, False), "lang_tokens"], ["txt"], axis=0),
          h.make_node("Concat", ["vis", "txt"], ["x"], axis=1),
          h.make_node("Unsqueeze", ["img_masks", const("ax2", np.array([2]), i)], ["im0"]),
          h.make_node("Expand", ["im0", const("s4", np.array([1, CAMS, (RES // PATCH) ** 2]), i)], ["im1"]),
          h.make_node("Reshape", ["im1", const("s5", np.array([1, CAMS * (RES // PATCH) ** 2]), i)], ["im2"]),
          h.make_node("Concat", ["im2", "lang_masks"], ["prefix_pad_mask"], axis=1)]
    for kv in ("k", "v"):
        n += [h.make_node("MatMul", ["x", weight("w" + kv, W["w" + kv], n, i, int4)], [kv + "0"]),
              h.make_node("Reshape", [kv + "0", const(kv + "s", np.array([1, PREFIX, LAYERS, KV_HEADS, HEAD]), i)], [kv + "1"]),
              h.make_node("Transpose", [kv + "1"], [kv], perm=[2, 0, 3, 1, 4])]
    backbone = h.make_graph(n, "backbone", [
        h.make_tensor_value_info("images", T.FLOAT, [1, CAMS, 3, RES, RES]),
        h.make_tensor_value_info("img_masks", T.BOOL, [1, CAMS]),
        h.make_tensor_value_info("lang_tokens", T.INT64, [1, TOKENS]),
        h.make_tensor_value_info("lang_masks", T.BOOL, [1, TOKENS])], [
        h.make_tensor_value_info("k", T.FLOAT16, [LAYERS, 1, KV_HEADS, PREFIX, HEAD]),
        h.make_tensor_value_info("v", T.FLOAT16, [LAYERS, 1, KV_HEADS, PREFIX, HEAD]),
        h.make_tensor_value_info("prefix_pad_mask", T.BOOL, [1, PREFIX])], i)

    # ---- expert (FP16, like the plan) ----
    n, i = [], []
    n += [h.make_node("Cast", ["x_t"], ["x16"], to=T.FLOAT16),
          h.make_node("Cast", ["state"], ["s16"], to=T.FLOAT16),
          h.make_node("Cast", ["t"], ["t16"], to=T.FLOAT16),
          h.make_node("MatMul", ["x16", weight("wq", W["wq"], n, i, False)], ["q0"]),
          h.make_node("MatMul", ["s16", weight("ws", W["ws"], n, i, False)], ["sq"]),
          h.make_node("Unsqueeze", ["sq", const("ax1", np.array([1]), i)], ["sq1"]),
          h.make_node("Unsqueeze", ["t16", const("ax12", np.array([1, 2]), i)], ["t3"]),
          h.make_node("Mul", ["t3", weight("wt", W["wt"], n, i, False)], ["tq"]),
          h.make_node("Add", ["q0", "sq1"], ["q1"]), h.make_node("Add", ["q1", "tq"], ["q2"]),
          h.make_node("Reshape", ["q2", const("qs", np.array([1, H, KV_HEADS, HEAD]), i)], ["q3"]),
          h.make_node("Transpose", ["q3"], ["q"], perm=[0, 2, 1, 3]),
          h.make_node("ReduceMean", ["k", const("ax0", np.array([0]), i)], ["km"], keepdims=0),
          h.make_node("ReduceMean", ["v", "ax0"], ["vm"], keepdims=0),
          h.make_node("Transpose", ["km"], ["kt"], perm=[0, 1, 3, 2]),
          h.make_node("MatMul", ["q", "kt"], ["s0"]),
          h.make_node("Mul", ["s0", const("isq", np.array(1 / np.sqrt(HEAD), np.float16), i)], ["s1"]),
          h.make_node("Unsqueeze", ["prefix_pad_mask", const("ax12b", np.array([1, 2]), i)], ["pm"]),
          h.make_node("Where", ["pm", "s1", const("neg", np.array(-1e4, np.float16), i)], ["s2"]),
          h.make_node("Softmax", ["s2"], ["p"], axis=-1),
          h.make_node("MatMul", ["p", "vm"], ["o0"]),
          h.make_node("Transpose", ["o0"], ["o1"], perm=[0, 2, 1, 3]),
          h.make_node("Reshape", ["o1", const("os", np.array([1, H, KV_HEADS * HEAD]), i)], ["o2"]),
          h.make_node("Sigmoid", ["o2"], ["o3"]), h.make_node("Mul", ["o2", "o3"], ["o4"]),
          h.make_node("MatMul", ["o4", weight("wo", W["wo"], n, i, False)], ["vel16"]),
          h.make_node("Cast", ["vel16"], ["velocity"], to=T.FLOAT)]
    expert = h.make_graph(n, "expert", [
        h.make_tensor_value_info("state", T.FLOAT, [1, STATE]),
        h.make_tensor_value_info("prefix_pad_mask", T.BOOL, [1, PREFIX]),
        h.make_tensor_value_info("k", T.FLOAT16, [LAYERS, 1, KV_HEADS, PREFIX, HEAD]),
        h.make_tensor_value_info("v", T.FLOAT16, [LAYERS, 1, KV_HEADS, PREFIX, HEAD]),
        h.make_tensor_value_info("x_t", T.FLOAT, [1, H, A]),
        h.make_tensor_value_info("t", T.FLOAT, [1])],
        [h.make_tensor_value_info("velocity", T.FLOAT, [1, H, A])], i)

    files = {}
    for name, g in (("backbone", backbone), ("expert", expert)):
        m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)])
        m.ir_version = 10
        onnx.checker.check_model(m)
        onnx.save(m, str(out / f"{name}.onnx"))
        files[name] = f"{name}.onnx"
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--int4", action="store_true", help="INT4 weights in the backbone, as Q/DQ nodes")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    W = make_weights(rng)
    files = build_onnx(W, args.out, args.int4)
    if args.int4:  # the reference uses the same dequantized weights as the graph
        for key in ("patch", "wk", "wv"):
            W[key] = W[key + "_deq"]
    Wr = {k: v.astype(np.float16).astype(np.float32) for k, v in W.items()}

    ref = {}
    for c in range(args.cases):
        images = rng.random((1, CAMS, 3, RES, RES)).astype(np.float32)
        img_masks = np.ones((1, CAMS), bool)
        lang_tokens = rng.integers(0, VOCAB, (1, TOKENS))
        lang_masks = np.arange(TOKENS)[None] < rng.integers(4, TOKENS + 1)
        state = (rng.random((1, STATE)) - 0.5).astype(np.float32)
        noise = rng.standard_normal((1, H, A)).astype(np.float32)
        k, v, pad = backbone_np(Wr, images, img_masks, lang_tokens, lang_masks)
        x, t, dt = noise.copy(), 1.0, -1.0 / args.steps
        v0 = expert_np(Wr, state, pad, k, v, x, np.array([t], np.float32))
        for _ in range(args.steps):
            x = x + dt * expert_np(Wr, state, pad, k, v, x, np.array([t], np.float32))
            t += dt
        for name, arr in (("images", images), ("img_masks", img_masks), ("lang_tokens", lang_tokens),
                          ("lang_masks", lang_masks), ("state", state), ("noise", noise), ("k", k), ("v", v),
                          ("prefix_pad_mask", pad), ("velocity_t1", v0), ("actions", x)):
            ref[f"case{c:02d}/{name}"] = arr
    np.savez(args.out / "reference.npz", **ref)

    hashed = [*files.values(), "reference.npz"]
    manifest = {
        "name": args.out.name, "model": "fake test graphs (engine/make_test_package.py)",
        "opset": 21, "precision": {"backbone": "int4_weight" if args.int4 else "fp16", "expert": "fp16"},
        "graphs": {
            "backbone": {"file": files["backbone"], "inputs": {"images": [[1, CAMS, 3, RES, RES], "float32"],
                         "img_masks": [[1, CAMS], "bool"], "lang_tokens": [[1, TOKENS], "int64"],
                         "lang_masks": [[1, TOKENS], "bool"]},
                         "outputs": {"k": [[LAYERS, 1, KV_HEADS, PREFIX, HEAD], "float16"],
                                     "v": [[LAYERS, 1, KV_HEADS, PREFIX, HEAD], "float16"],
                                     "prefix_pad_mask": [[1, PREFIX], "bool"]}},
            "expert": {"file": files["expert"], "inputs": {"state": [[1, STATE], "float32"],
                       "prefix_pad_mask": [[1, PREFIX], "bool"], "k": [[LAYERS, 1, KV_HEADS, PREFIX, HEAD], "float16"],
                       "v": [[LAYERS, 1, KV_HEADS, PREFIX, HEAD], "float16"], "x_t": [[1, H, A], "float32"],
                       "t": [[1], "float32"]}, "outputs": {"velocity": [[1, H, A], "float32"]}}},
        "flow": {"steps_default": args.steps, "chunk": H, "action_dim": A,
                 "context": {"k": "k", "v": "v", "prefix_pad_mask": "prefix_pad_mask"}},
        "reference": {"file": "reference.npz", "cases": args.cases, "noise_seed": 0},
        "sha256": {f: sha256(args.out / f) for f in hashed},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out}: {', '.join(hashed)}, manifest.json ({manifest['precision']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
