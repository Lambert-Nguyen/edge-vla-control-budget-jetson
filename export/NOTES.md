# Hy-VLA ONNX export: status and blockers

Last update: Nov 20, 2026. Workstation, Hy-VLA uv env with torch 2.7.1,
onnx 1.19.1, onnxruntime-gpu 1.23.2 and onnxscript. Jetson: TensorRT 10.3.

## Done

- `wrappers.py` splits the model into `BackboneWrapper` and `ExpertStep`.
  With the same noise, the split gives the same actions as `sample_actions`
  (max difference 0).
- `export_onnx.py` exports the action expert (one Euler step) to ONNX in FP32.
  L0: ONNX Runtime vs PyTorch 0.044% of the largest velocity (limit 1%).
- On the Jetson, TensorRT builds the expert from that ONNX:
  - FP32 engine: within 0.025% of ONNX Runtime.
  - FP16 engine (`--fp16`): within 0.575% (limit 2%), about 750 MB, 16.2 ms
    of GPU time per step (trtexec), 21.3 ms from Python with input copies.
- Shapes with 3 cameras, 6 history frames at 224 px and 64 task tokens:
  K and V are `[32, 1, 240, 4, 128]` each (layer, batch, token, kv head,
  head dim). The prefix is 240 tokens. The expert adds 41 tokens.
- The exported expert step has about 373 M parameters (1.49 GB in FP32). The
  862 M "expert tower" count included parts the step does not use, such as
  the token embedding.

## Export patches (export_onnx.py, only while exporting)

The Hy-VLA files stay as they are. Each patch is the same math.

1. No deepcopy of the K/V cache in the expert step (it broke torch.export).
2. Rotary embedding without `@dynamic_rope_update` (data-dependent check).
3. `mask_apply` with fixed index lists. The code that makes a modality mask
   attaches its pattern, because the values are not known while exporting.
4. `embed_suffix` in the model dtype, not a hard-coded bfloat16, and its time
   embedding in float32, not float64 (TensorRT has no float64).
5. `HyDualTower.forward` without the bfloat16 cast of the attention output,
   and no q/k Split when there is only one part.
6. Angles of sin and cos in [0, 2 pi) first.

Two more were needed because TensorRT got wrong results. polygraphy showed
the same wrong output in TensorRT 10.3 and 11.5, and a right one when every
node was marked as an output:

7. `rotate_half` as `x @ R` with a fixed 128 x 128 matrix. TensorRT fused the
   slice, neg and concat wrongly: the rotated q and k of layer 0 were the
   first wrong values (FP32 engine 23% off before this fix).
8. RMSNorm divides each row by its largest |x| before squaring. With `--fp16`
   the squares (up to 280^2) overflowed FP16 and the output collapsed.

## Next

- Export the backbone. Expected problems: the memory encoder calls
  `flash_attn` directly (needs a plain attention path), the prefix modality
  mask needs its pattern, and the backbone is 3.8 B parameters, so the FP16
  engine (about 7.6 GB) does not fit the board. It needs INT4 weights (#15
  showed INT4 builds on TensorRT 10.3).
- Write the handoff package (manifest, both graphs, reference.npz) for the
  engine builder in #19.
