import numpy as np, onnxruntime as ort, sys
d = sys.argv[1]
ref = dict(np.load("hyvla-bb/backbone_reference.npz"))
sess = ort.InferenceSession(f"{d}/backbone.onnx", providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
worst = {}
for c in ("case00", "case01"):
    outs = dict(zip(["k", "v"], sess.run(None, {n: ref[f"{c}/{n}"] for n in names})[:2]))
    for key in ("k", "v"):
        o = outs[key].astype(np.float64); w = ref[f"{c}/{key}"].astype(np.float64)
        cos = [float((o[i].ravel() @ w[i].ravel()) / (np.linalg.norm(o[i]) * np.linalg.norm(w[i]))) for i in range(32)]
        rel = [float(np.abs(o[i] - w[i]).max() / np.abs(w[i]).max()) for i in range(32)]
        a = worst.setdefault(key, [1.0, 0.0, 0.0])
        a[0] = min(a[0], min(cos)); a[1] = max(a[1], max(rel)); a[2] = max(a[2], rel[0])
for key, (c, r, r0) in worst.items():
    print(f"  {d} {key}: min cosine {c:.5f}, max diff {100 * r:.2f}%, layer 0 {100 * r0:.2f}%")
