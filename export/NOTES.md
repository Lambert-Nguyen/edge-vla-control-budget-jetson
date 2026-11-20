# Hy-VLA ONNX export: status and blockers

Last update: Nov 20, 2026. Workstation, Hy-VLA uv env with torch 2.7.1,
onnx 1.19.1, onnxruntime-gpu 1.23.2 and onnxscript.

## Done

- `wrappers.py` splits the model into `BackboneWrapper` and `ExpertStep`.
  With the same noise, the split gives the same actions as `sample_actions`
  (max difference 0, see `results/20261120-122742_rtx4090-hyvla-split-check.json`).
- Shapes with 3 cameras, 6 history frames at 224 px and 64 task tokens:
  K and V are `[32, 1, 240, 4, 128]` each (layer, batch, token, kv head,
  head dim). The prefix is 240 tokens. The expert adds 41 tokens (1 state
  token and 40 actions).
- K comes out of the backbone in float32 and V in bfloat16.

## Blockers found when exporting ExpertStep

We tried `torch.onnx.export` on `ExpertStep` with both exporters.

1. `copy.deepcopy` of the cache in `denoise_step` breaks `torch.export`
   ("argument of type UntypedStorage"). Fix: the wrapper passes a new cache
   dict on every call, so it does not need the copy. A version of the step
   without the copy gives the same velocity (difference 0).
2. Rotary embedding. transformers runs `dynamic_frequency_update` and checks
   `if seq_len > self.max_seq_len_cached`, which depends on the data, so
   `torch.export` stops (GuardOnDataDependentSymNode). Our shapes are fixed,
   so compute cos and sin once for the 281 positions and skip the dynamic
   update during export.
3. The old exporter (`dynamo=False`) traces the graph and then fails in the
   ONNX conversion with an empty AssertionError. We use `dynamo=True` from now.

## Expected next blockers

- `mask_apply` in `modeling_dual_tower.py` sends image and text tokens to
  different weights with boolean indexing. With fixed shapes the positions
  are known, so it can use fixed slices instead.
- Casts: the attention output is cast to bfloat16 (line 502) and attention
  runs in float32 (lines 590 to 592). An FP16 graph needs these as one dtype.
  The Orin GPU (Ampere) also supports BF16 in TensorRT 10, so a BF16 graph
  is an option to test.
- The memory encoder uses `flash_attn` directly. The backbone export needs
  a plain attention path there.
