import numpy as np, onnxruntime as ort, sys
# L2: the full chunk (10 Euler steps of the expert) from each backbone's K and V,
# with the same state and noise, compared with the FP32 backbone.
ref = dict(np.load("hyvla-bb/backbone_reference.npz"))
expert = ort.InferenceSession("hyvla-fp32/expert.onnx", providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
state = (rng.random((1, 32), dtype=np.float32) - 0.5)
noise = rng.standard_normal((1, 40, 32)).astype(np.float32)
def chunk(k, v, pad):
    x, t, dt = noise.copy(), 1.0, -0.1
    for _ in range(10):
        vel = expert.run(None, {"state": state, "prefix_pad_mask": pad, "k": k, "v": v, "x_t": x, "t": np.array([t], np.float32)})[0]
        x = x + dt * vel; t += dt
    return x[0, :, :14]  # 14 real action dims for the dual-arm robot
base = {c: chunk(ref[f"{c}/k"], ref[f"{c}/v"], ref[f"{c}/prefix_pad_mask"]) for c in ("case00", "case01")}
for d in sys.argv[1:]:
    bb = ort.InferenceSession(f"{d}/backbone.onnx", providers=["CPUExecutionProvider"])
    names = [i.name for i in bb.get_inputs()]
    errs = []
    for c in ("case00", "case01"):
        k, v, pad = bb.run(None, {n: ref[f"{c}/{n}"] for n in names})
        a = chunk(k, v, pad); b = base[c]
        errs.append((np.abs(a - b).mean(), np.percentile(np.abs(a - b), 99), np.abs(a - b).max() / np.abs(b).max()))
    mae, p99, rel = (max(e[i] for e in errs) for i in range(3))
    print(f"  {d:22s} actions vs FP32 backbone: mae {mae:.4f}, p99 {p99:.4f}, max {100 * rel:.2f}% of the largest action")
print("  (export spec limits: L2 mae 0.05, p99 0.15, in normalized units)")
