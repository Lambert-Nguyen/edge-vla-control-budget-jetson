#!/usr/bin/env python3
"""Export the Hy-VLA action expert (one Euler step) to ONNX, and check it (level L0).

Run in the Hy-VLA uv env on the workstation:

    ~/hyvla/Hy-Embodied-0.5-VLA/.venv/bin/python export/export_onnx.py \\
        --ckpt ~/hyvla/ckpt/Hy-Embodied-0.5-VLA-RoboTwin --out build/onnx/hyvla-fp32

The graph is exported in FP32. TensorRT builds it in FP16 on the Jetson (the
--fp16 flag of the engine builder), which is the usual way and keeps the ONNX
check simple: ONNX Runtime has little BF16 support.

Three things in the Hy-VLA code stop torch.export. They are patched only while
exporting, so the Hy-VLA files stay as they are (see export/NOTES.md):

1. denoise_step deep-copies the K/V cache. ExpertStepExport builds a new cache
   dict on every call, so it does the same work without the copy.
2. The rotary embedding has @dynamic_rope_update, which checks the sequence
   length against the data. It only changes anything past 262144 positions,
   and we use 281, so the export uses the plain cos/sin code.
3. mask_apply sends image and text tokens to different weights with boolean
   indexing. With fixed shapes the mask is known, so the export version uses
   fixed index lists.

Level L0 then runs the same inputs through PyTorch (FP32) and ONNX Runtime and
reports the largest difference as a share of the largest velocity (limit 1%).
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
from wrappers import BackboneWrapper, ExpertStep  # noqa: E402


# ---- patches used only while exporting -----------------------------------------


def _static_mask_apply(hidden_states, mask, text_funcs, vision_funcs, out_dims=None, padding_mask=None):
    """mask_apply with the mask read once as Python values, so the shapes are fixed."""
    B, S, D = hidden_states.shape
    # During torch.export the mask values are not known, so the code that makes
    # the mask attaches its pattern (a list of bools, one per token) to it.
    pattern = getattr(mask, "static_pattern", None)
    m = list(pattern) * B if pattern is not None else [bool(v) for v in mask.reshape(-1).tolist()]
    if all(m) or not any(m):
        funcs = vision_funcs if all(m) else text_funcs
        return [fn(hidden_states).reshape(B, S, -1) for fn in funcs]
    dev = hidden_states.device
    vis = torch.tensor([i for i, v in enumerate(m) if v], device=dev)
    txt = torch.tensor([i for i, v in enumerate(m) if not v], device=dev)
    flat = hidden_states.reshape(B * S, D)
    outs = []
    for tf, vf in zip(text_funcs, vision_funcs):
        t = tf(flat.index_select(0, txt))
        v = vf(flat.index_select(0, vis))
        out = torch.zeros(B * S, t.shape[-1], dtype=t.dtype, device=dev)
        out = out.index_copy(0, txt, t).index_copy(0, vis, v.to(t.dtype))
        outs.append(out.reshape(B, S, -1))
    return outs


def _plain_rope_forward(self, x, position_ids):
    """HunYuanVLMoTRotaryEmbedding.forward without @dynamic_rope_update."""
    inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    freqs = (inv @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


def _embed_suffix_any_dtype(self, state, noisy_actions, timestep):
    """HyVLAFlowMatching.embed_suffix with the model's dtype in place of bfloat16.

    The original casts the state, the time embedding and the noisy actions to
    bfloat16, so an FP32 model fails. Same math otherwise.
    """
    import torch.nn.functional as F
    from hy_vla.modeling_hy_vla import create_sinusoidal_pos_embedding

    dt = self.action_in_proj.weight.dtype
    state_block = self.state_proj(state).to(dt)[:, None, :]
    bsize, device = state_block.shape[0], state_block.device
    time_emb = create_sinusoidal_pos_embedding(timestep, self.config.proj_width, min_period=4e-3,
                                               max_period=4.0, device=device).type(dt)
    action_emb = self.action_in_proj(noisy_actions.to(dt))
    time_emb = time_emb[:, None, :].expand_as(action_emb)
    action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2))))
    n = self.config.n_action_steps
    embs = torch.cat([state_block, action_time_emb], dim=1)
    pad_masks = torch.ones(bsize, 1 + n, dtype=torch.bool, device=device)
    att = torch.tensor([1] + [1] + [0] * (n - 1), dtype=torch.bool, device=device)[None, :].expand(bsize, 1 + n)
    modality = torch.ones(bsize, 1 + n, dtype=torch.bool, device=device)
    modality.static_pattern = [True] * (1 + n)  # all action-expert tokens
    return embs, pad_masks, att, modality


@contextlib.contextmanager
def export_patches():
    import hy_vla.hunyuan_vl_mot.modeling_hunyuan_vl_mot as mot
    import hy_vla.modeling_dual_tower as dt

    import hy_vla.modeling_hy_vla as hv

    saved = (dt.mask_apply, mot.HunYuanVLMoTRotaryEmbedding.forward, hv.HyVLAFlowMatching.embed_suffix)
    dt.mask_apply = _static_mask_apply
    mot.HunYuanVLMoTRotaryEmbedding.forward = _plain_rope_forward
    hv.HyVLAFlowMatching.embed_suffix = _embed_suffix_any_dtype
    try:
        yield
    finally:
        dt.mask_apply, mot.HunYuanVLMoTRotaryEmbedding.forward, hv.HyVLAFlowMatching.embed_suffix = saved


class ExpertStepExport(nn.Module):
    """ExpertStep without the cache deepcopy. Same inputs and output."""

    def __init__(self, policy):
        super().__init__()
        self.flow = policy.model

    def forward(self, state, prefix_pad_mask, k, v, x_t, t):
        from hy_vla.modeling_hy_vla import make_att_2d_masks

        f = self.flow
        cache = {i: {"key_states": k[i], "value_states": v[i]} for i in range(k.shape[0])}
        embs, pad, att, modality = f.embed_suffix(state, x_t, t.expand(state.shape[0]))
        b, s, p = pad.shape[0], pad.shape[1], prefix_pad_mask.shape[1]
        full = torch.cat([prefix_pad_mask[:, None, :].expand(b, s, p), make_att_2d_masks(pad, att)], dim=2)
        pos = torch.sum(prefix_pad_mask, dim=-1)[:, None] + torch.cumsum(pad, dim=1) - 1
        out, _, _, _ = f.dual_tower.forward(attention_mask=full, position_ids=pos, past_key_values=cache,
                                            inputs_embeds=[None, embs], use_cache=True, fill_kv_cache=False,
                                            modality_masks=[None, modality])
        return f.action_out_proj(out[1][:, -f.config.n_action_steps:]).float()


# ---- export and check -------------------------------------------------------------


def make_inputs(policy, cfg, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    res = cfg.resize_imgs_with_padding[0]
    dtype = next(policy.parameters()).dtype
    img = lambda: torch.rand(1, 6, 3, res, res, device="cuda", generator=g).to(dtype)  # noqa: E731
    batch = {"observation.images.top_head": img(), "observation.images.hand_left": img(),
             "observation.images.hand_right": img(),
             "observation.state": (torch.rand(1, cfg.max_state_dim, device="cuda", generator=g) - 0.5).to(dtype),
             "task": ["beat the block with the hammer"]}
    with torch.no_grad():
        images, masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        tok, tmask, _ = policy.prepare_language(batch)
        k, v, pad = BackboneWrapper(policy)(images, masks, tok, tmask)
    x = torch.randn(1, cfg.n_action_steps, cfg.max_action_dim, device="cuda", generator=g)
    t = torch.tensor([1.0], device="cuda")
    return (state, pad, k, v, x, t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", type=int, default=4, help="L0 test inputs (default: %(default)s)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--reference", type=int, default=0, metavar="N",
                    help="save the first N cases (inputs and ONNX outputs) to <out>/expert_reference.npz")
    args = ap.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    from hy_vla import HyVLA, HyVLAConfig

    cfg = HyVLAConfig.from_pretrained(args.ckpt)
    policy = HyVLA.from_pretrained(args.ckpt, config=cfg)
    policy.enable_video_encoder_if_needed()
    # The backbone needs BF16 (flash_attn in the memory encoder), so the test
    # inputs come from the BF16 model. Then the model goes to FP32 for the export.
    policy = policy.to(device="cuda", dtype=torch.bfloat16).eval()
    cases = [make_inputs(policy, cfg, s) for s in range(args.cases)]
    with torch.no_grad():  # the original expert step, in BF16, the way the checkpoint runs
        ref_orig = [ExpertStep(policy)(*c).float() for c in cases]
    cases = [tuple(a.float() if a.is_floating_point() else a for a in c) for c in cases]
    policy = policy.to(dtype=torch.float32)
    torch.cuda.empty_cache()
    step = ExpertStepExport(policy).eval()
    names = ["state", "prefix_pad_mask", "k", "v", "x_t", "t"]
    with torch.no_grad(), export_patches():
        ref = [step(*c) for c in cases]
    same = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(ref, ref_orig))
    print(f"FP32 export version vs the original BF16 expert step: max diff {100 * same:.3f}% "
          f"of the largest velocity (BF16 rounding)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "expert.onnx"
    t0 = time.time()
    with torch.no_grad(), export_patches():
        torch.onnx.export(step, cases[0], str(path), dynamo=True, opset_version=18,
                          input_names=names, output_names=["velocity"], external_data=True)
    export_s = time.time() - t0
    model = onnx.load(str(path), load_external_data=False)
    ops: dict[str, int] = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    size_mb = sum(p.stat().st_size for p in args.out.glob("expert.onnx*")) / 1e6
    print(f"exported in {export_s:.0f} s: {len(model.graph.node)} nodes, {size_mb:.0f} MB", flush=True)
    print("  op types: " + ", ".join(f"{k} {v}" for k, v in sorted(ops.items(), key=lambda kv: -kv[1])), flush=True)

    sess = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    worst, saved = 0.0, {}
    for i, (c, r) in enumerate(zip(cases, ref)):
        feeds = {n: a.detach().cpu().numpy() for n, a in zip(names, c) if n in in_names}
        out = sess.run(None, feeds)[0]
        rr = r.cpu().numpy()
        worst = max(worst, float(np.abs(out - rr).max() / np.abs(rr).max()))
        if i < args.reference:
            saved.update({f"case{i:02d}/{n}": a for n, a in feeds.items()})
            saved[f"case{i:02d}/velocity"] = out
    if saved:
        np.savez(args.out / "expert_reference.npz", **saved)
        print(f"saved {args.reference} reference cases to {args.out / 'expert_reference.npz'}", flush=True)
    ok = worst <= 0.01
    print(f"L0: ONNX Runtime vs PyTorch on {len(cases)} cases: max diff {100 * worst:.4f}% of the largest "
          f"velocity (limit 1%) -> {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"  ONNX inputs: {in_names} (unused ones are dropped by the exporter)", flush=True)

    if args.json:
        import subprocess

        def g(*a):
            try:
                return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), *a],
                                               text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                return None
        args.json.write_text(json.dumps({
            "kind": "hyvla_onnx_export", "graph": "expert", "precision": "fp32",
            "export_s": export_s, "nodes": len(model.graph.node), "size_mb": size_mb, "op_types": ops,
            "onnx_inputs": in_names, "patched_vs_original_max_diff": same,
            "l0": {"cases": len(cases), "max_rel_diff": worst, "limit": 0.01, "pass": ok},
            "git": {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))},
        }, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
