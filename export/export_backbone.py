#!/usr/bin/env python3
"""Export the Hy-VLA backbone to ONNX, and check it (level L0).

The backbone reads the three camera stacks (6 history frames each, 224 px, already
resized and scaled to [-1, 1] by prepare_images) and the 64 task tokens, and
returns the key/value cache of all 32 layers and the prefix padding mask. The
action expert (export_onnx.py) reads exactly these.

    ~/hyvla/Hy-Embodied-0.5-VLA/.venv/bin/python export/export_backbone.py \\
        --ckpt ~/hyvla/ckpt/Hy-Embodied-0.5-VLA-RoboTwin --out build/onnx/hyvla-fp32

It uses all patches of export_onnx.py (they are the same math), and three more
for the backbone:

1. flash_attn has no ONNX export. The vision tower (the plain ViT blocks and the
   memory encoder) calls flash_attn_func(q, k, v, softmax_scale). The export
   uses plain attention with the same inputs and the same math.
2. The prefix modality mask (image tokens or text tokens) gets its pattern from
   one run before the export, like the expert mask.
3. embed_prefix casts the image tokens to bfloat16. The export keeps the dtype
   of the model.

The graph is FP32, about 15 GB of weights. It is too big for the board in FP16
(about 7.6 GB), so the board gets an INT4-weight version (next step). L0 here
compares ONNX Runtime with PyTorch on K and V.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import export_patches  # noqa: E402
from wrappers import BackboneWrapper  # noqa: E402

PREFIX_PATTERN: list[bool] | None = None  # filled by one run before the export


def _flash_attn_plain(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    """flash_attn_func with plain ops. q, k, v and the output are (B, N, H, d)."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, N, d)
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    if causal:
        n = scores.shape[-1]
        scores = scores.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), 1), float("-inf"))
    out = scores.softmax(dim=-1).to(v.dtype) @ v
    return out.transpose(1, 2)


def _embed_prefix_source_patch(hv):
    """embed_prefix without the bfloat16 cast, and with the modality mask pattern attached."""
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(hv.HyVLAFlowMatching.embed_prefix))
    old = "img_emb = self.dual_tower.embed_image(img).to(dtype=torch.bfloat16)"
    assert src.count(old) == 1, "the Hy-VLA code changed: look at the bfloat16 cast in embed_prefix"
    src = src.replace(old, "img_emb = self.dual_tower.embed_image(img).to(dtype=bos_emb.dtype)")
    old = "    return embs, pad_masks, att_masks, modality_mask, image_idx_ranges, image_full_ranges"
    assert src.count(old) == 1, "the Hy-VLA code changed: look at the end of embed_prefix"
    src = src.replace(old, "    modality_mask.static_pattern = _PREFIX_PATTERN()\n" + old)
    ns: dict = {}
    glb = dict(hv.__dict__, _PREFIX_PATTERN=lambda: PREFIX_PATTERN)
    exec(compile(src, hv.__file__, "exec"), glb, ns)
    return ns["embed_prefix"]


@contextlib.contextmanager
def backbone_patches():
    import hy_vla.hunyuan_vl_mot.modeling_hunyuan_vl_mot as mot
    import hy_vla.modeling_hy_vla as hv
    import hy_vla.space_time_attention as sta

    saved = (sta.flash_attn_func, mot.flash_attn_func, hv.HyVLAFlowMatching.embed_prefix)
    sta.flash_attn_func = _flash_attn_plain
    mot.flash_attn_func = _flash_attn_plain
    hv.HyVLAFlowMatching.embed_prefix = _embed_prefix_source_patch(hv)
    try:
        with export_patches():
            yield
    finally:
        sta.flash_attn_func, mot.flash_attn_func, hv.HyVLAFlowMatching.embed_prefix = saved


class BackboneExport(nn.Module):
    """Three camera stacks and the task tokens -> K, V of all layers and the prefix mask."""

    def __init__(self, policy):
        super().__init__()
        self.wrapper = BackboneWrapper(policy)

    def forward(self, top_head, hand_left, hand_right, lang_tokens, lang_masks):
        images = [top_head, hand_left, hand_right]
        masks = [torch.ones(top_head.shape[0], dtype=torch.bool, device=top_head.device)] * 3
        k, v, pad = self.wrapper(images, masks, lang_tokens, lang_masks)
        return k.float(), v.float(), pad


def make_inputs(policy, cfg, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    res = cfg.resize_imgs_with_padding[0]
    img = lambda: torch.rand(1, 6, 3, res, res, device="cuda", generator=g).to(torch.bfloat16)  # noqa: E731
    tasks = ["beat the block with the hammer", "click the bell", "stack two blocks", "adjust the bottle"]
    batch = {"observation.images.top_head": img(), "observation.images.hand_left": img(),
             "observation.images.hand_right": img(), "observation.state": torch.zeros(1, cfg.max_state_dim,
             device="cuda", dtype=torch.bfloat16), "task": [tasks[seed % len(tasks)]]}
    with torch.no_grad():
        images, _ = policy.prepare_images(batch)
        tok, tmask, _ = policy.prepare_language(batch)
    return (*images, tok, tmask)


def main() -> int:
    global PREFIX_PATTERN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", type=int, default=2)
    ap.add_argument("--reference", type=int, default=0, metavar="N",
                    help="save the first N cases (inputs and ONNX outputs) to <out>/backbone_reference.npz")
    ap.add_argument("--skip-export", action="store_true", help="only run L0 on an existing backbone.onnx")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    from hy_vla import HyVLA, HyVLAConfig

    cfg = HyVLAConfig.from_pretrained(args.ckpt)
    policy = HyVLA.from_pretrained(args.ckpt, config=cfg)
    policy.enable_video_encoder_if_needed()
    policy = policy.to(device="cuda", dtype=torch.bfloat16).eval()
    names = ["top_head", "hand_left", "hand_right", "lang_tokens", "lang_masks"]
    cases = [make_inputs(policy, cfg, s) for s in range(args.cases)]

    with torch.no_grad():  # the original backbone in BF16, and the prefix mask pattern
        ref_orig = [BackboneWrapper(policy)(list(c[:3]), [torch.ones(1, dtype=torch.bool, device="cuda")] * 3,
                                            c[3], c[4]) for c in cases]
        f = policy.model
        _, _, _, modality, _, _ = f.embed_prefix(list(cases[0][:3]), [torch.ones(1, dtype=torch.bool,
                                                 device="cuda")] * 3, cases[0][3], cases[0][4])
        PREFIX_PATTERN = [bool(x) for x in modality[0].tolist()]
    print(f"prefix: {len(PREFIX_PATTERN)} tokens, {sum(PREFIX_PATTERN)} image tokens", flush=True)

    # FP32 for the export. The expert tower is not used by the backbone graph.
    policy.model.dual_tower.expert.to("cpu")
    policy = policy.to(dtype=torch.float32)
    torch.cuda.empty_cache()
    model = BackboneExport(policy).eval()
    cases = [tuple(a.float() if a.is_floating_point() else a for a in c) for c in cases]
    with torch.no_grad(), backbone_patches():
        ref = [model(*c) for c in cases]
    same = max(float((a.float() - b[0].float()).abs().max() / b[0].float().abs().max())
               for (a, _, _), b in zip(ref, ref_orig))
    print(f"FP32 export version vs the original BF16 backbone: K max diff {100 * same:.3f}% (BF16 rounding)",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "backbone.onnx"
    export_s = None
    if not args.skip_export:
        t0 = time.time()
        with torch.no_grad(), backbone_patches():
            torch.onnx.export(model, cases[0], str(path), dynamo=True, opset_version=18, input_names=names,
                              output_names=["k", "v", "prefix_pad_mask"], external_data=True)
        export_s = time.time() - t0
    import gc

    del model, policy, f, ref_orig  # free RAM and GPU memory before ONNX Runtime loads 15 GB
    gc.collect()
    torch.cuda.empty_cache()
    m = onnx.load(str(path), load_external_data=False)
    ops: dict[str, int] = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    size_mb = sum(p.stat().st_size for p in args.out.glob("backbone.onnx*")) / 1e6
    print(f"exported{'' if export_s is None else f' in {export_s:.0f} s'}: {len(m.graph.node)} nodes, "
          f"{size_mb:.0f} MB", flush=True)
    print("  op types: " + ", ".join(f"{k} {v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])), flush=True)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    worst = {"k": 0.0, "v": 0.0, "prefix_pad_mask": 0.0}
    saved = {}
    for i, (c, r) in enumerate(zip(cases, ref)):
        feeds = {n: a.detach().cpu().numpy() for n, a in zip(names, c) if n in in_names}
        outs = sess.run(None, feeds)
        for key, o, rr in zip(("k", "v", "prefix_pad_mask"), outs, r):
            rr = rr.cpu().numpy()
            if key == "prefix_pad_mask":
                worst[key] = max(worst[key], float((o != rr).mean()))
            else:
                worst[key] = max(worst[key], float(np.abs(o - rr).max() / np.abs(rr).max()))
        if i < args.reference:
            saved.update({f"case{i:02d}/{n}": a for n, a in feeds.items()})
            saved.update({f"case{i:02d}/{key}": o for key, o in zip(("k", "v", "prefix_pad_mask"), outs)})
    if saved:
        np.savez(args.out / "backbone_reference.npz", **saved)
    ok = worst["k"] <= 0.01 and worst["v"] <= 0.01 and worst["prefix_pad_mask"] == 0
    print(f"L0: ONNX Runtime vs PyTorch on {len(cases)} cases: K {100 * worst['k']:.4f}%, V {100 * worst['v']:.4f}%"
          f", mask {100 * worst['prefix_pad_mask']:.1f}% (limit 1%) -> {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"  ONNX inputs: {in_names}", flush=True)

    if args.json:
        import subprocess

        def g(*a):
            try:
                return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), *a],
                                               text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                return None
        args.json.write_text(json.dumps({
            "kind": "hyvla_onnx_export", "graph": "backbone", "precision": "fp32", "export_s": export_s,
            "nodes": len(m.graph.node), "size_mb": size_mb, "op_types": ops, "onnx_inputs": in_names,
            "prefix_tokens": len(PREFIX_PATTERN), "image_tokens": sum(PREFIX_PATTERN),
            "patched_vs_original_k_max_diff": same,
            "l0": {"cases": len(cases), "max_rel_diff": worst, "limit": 0.01, "pass": ok},
            "git": {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))},
        }, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
