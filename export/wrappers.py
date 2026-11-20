#!/usr/bin/env python3
"""The two parts of Hy-VLA that we export: the backbone and one expert step.

Hy-VLA works like pi0. The backbone (the VLM tower) reads the camera frames
and the task tokens once per chunk and fills a key/value cache in each of its
32 layers. Then the action expert runs num_steps Euler steps. Each step reads
that cache, the state, the noisy actions x_t and the time t, and returns the
velocity v_t. So:

- BackboneWrapper: images, image masks, task tokens and masks
  -> K and V of all layers (stacked) and the prefix padding mask.
- ExpertStep: state, prefix padding mask, K, V, x_t, t -> v_t.
- euler_loop(): the loop that calls ExpertStep. The same short function is
  used here, in the RoboTwin adapter and in the runtime on the Jetson, so the
  simulation tests the loop that runs on the board.

The tokenizer and the image resize stay outside, on the CPU (export spec).

    python export/wrappers.py --ckpt ~/hyvla/ckpt/Hy-Embodied-0.5-VLA-RoboTwin --check

--check runs the released model both ways on the same inputs and the same
noise, the normal sample_actions() and the split, and prints the largest
difference. It must be zero or close to it before any ONNX export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn


def _flow(policy):
    """HyVLAFlowMatching inside the HyVLA policy."""
    return policy.model


class BackboneWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.flow = _flow(policy)

    def forward(self, images, img_masks, lang_tokens, lang_masks):
        from hy_vla.modeling_hy_vla import make_att_2d_masks

        f = self.flow
        embs, pad_masks, att_masks, modality, idx_ranges, full_ranges = f.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        f._apply_visual_segment_mask(att_2d, idx_ranges, full_ranges)
        _, cache, _, _ = f.dual_tower.forward(
            attention_mask=att_2d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True, modality_masks=[modality, None])
        layers = sorted(cache)
        k = torch.stack([cache[i]["key_states"] for i in layers])
        v = torch.stack([cache[i]["value_states"] for i in layers])
        return k, v, pad_masks


class ExpertStep(nn.Module):
    """One flow-matching step of the action expert."""

    def __init__(self, policy):
        super().__init__()
        self.flow = _flow(policy)

    def forward(self, state, prefix_pad_masks, k, v, x_t, t):
        cache = {i: {"key_states": k[i], "value_states": v[i]} for i in range(k.shape[0])}
        v_t, _ = self.flow.denoise_step(state, prefix_pad_masks, cache, x_t, t.expand(state.shape[0]))
        return v_t


def euler_loop(expert_step, state, prefix_pad_masks, k, v, noise, num_steps: int):
    """x goes from noise at t = 1 to actions at t = 0, in num_steps equal steps."""
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
    t = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
    x = noise.clone()
    for _ in range(num_steps):
        x = x + dt * expert_step(state, prefix_pad_masks, k, v, x, t)
        t = t + dt
    return x


def check(ckpt: str, seed: int, json_path: Path | None) -> int:
    from hy_vla import HyVLA, HyVLAConfig

    cfg = HyVLAConfig.from_pretrained(ckpt)
    policy = HyVLA.from_pretrained(ckpt, config=cfg)
    policy.enable_video_encoder_if_needed()
    policy = policy.to(device="cuda", dtype=torch.bfloat16).eval()
    flow = _flow(policy)

    g = torch.Generator(device="cuda").manual_seed(seed)
    res = cfg.resize_imgs_with_padding[0]
    img = lambda: torch.rand(1, 6, 3, res, res, device="cuda", generator=g).to(torch.bfloat16)  # noqa: E731
    batch = {"observation.images.top_head": img(), "observation.images.hand_left": img(),
             "observation.images.hand_right": img(),
             "observation.state": (torch.rand(1, cfg.max_state_dim, device="cuda", generator=g) - 0.5).to(torch.bfloat16),
             "task": ["beat the block with the hammer"]}
    noise = torch.randn(1, cfg.n_action_steps, cfg.max_action_dim, device="cuda", generator=g)

    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens, lang_masks, _ = policy.prepare_language(batch)
        ref = flow.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
        k, v, pad = BackboneWrapper(policy)(images, img_masks, lang_tokens, lang_masks)
        out = euler_loop(ExpertStep(policy), state, pad, k, v, noise, cfg.num_steps)

    n = cfg.action_feature.shape[0]
    diff = (out[..., :n].float() - ref[..., :n].float()).abs().max().item()
    scale = ref[..., :n].float().abs().max().item()
    shapes = {"k": list(k.shape), "v": list(v.shape), "prefix_len": int(pad.shape[1]),
              "lang_tokens": list(lang_tokens.shape), "state": list(state.shape), "noise": list(noise.shape),
              "images": [list(i.shape) for i in images]}
    cache_mb = (k.numel() + v.numel()) * k.element_size() / 1e6
    print(f"split vs sample_actions: max abs diff {diff:.3e} (largest action {scale:.3f}, "
          f"{100 * diff / max(scale, 1e-9):.4f}% of it)")
    print(f"K and V: {shapes['k']} each (layer, batch, token, kv head, head dim), {k.dtype}, {cache_mb:.1f} MB together. Prefix {shapes['prefix_len']} tokens.")
    print(f"images {shapes['images']}, task tokens {shapes['lang_tokens']}, state {shapes['state']}, "
          f"noise {shapes['noise']}")
    ok = diff <= 1e-3 * max(scale, 1e-9)
    print("PASS" if ok else "FAIL: the split does not reproduce sample_actions")
    if json_path:
        import subprocess

        def g(*a):
            try:
                return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), *a],
                                               text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                return None
        git = {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}
        json_path.write_text(json.dumps({"kind": "hyvla_split_check", "git": git, "max_abs_diff": diff, "max_abs_action": scale, "pass": ok,
                                         "shapes": shapes, "kv_cache_mb": cache_mb, "seed": seed,
                                         "num_steps": cfg.num_steps, "dtype": "bfloat16"}, indent=2) + "\n")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--check", action="store_true", help="compare the split with sample_actions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.check:
        return check(args.ckpt, args.seed, args.json)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
