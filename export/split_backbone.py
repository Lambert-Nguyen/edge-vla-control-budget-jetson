#!/usr/bin/env python3
"""Split an exported Hy-VLA backbone graph into two graphs at one language layer.

TensorRT 10.3 on the Jetson cannot build the INT4 backbone as one engine. It
fuses the 32 language layers into one block, and the only kernel for that block
asks for 5.3 GB of GPU memory while the board has about 5.8 GB free in all
(trtexec log of Nov 20). Two engines need about half of that each.

Part 1 runs the vision tower, the embeddings and layers 0 to N-1. Part 2 runs
layers N to 31 and returns the same outputs as the full graph (k, v and
prefix_pad_mask). Every tensor that part 2 needs from part 1 becomes an output
of part 1 and an input of part 2, with the same name. So the runtime feeds the
outputs of engine 1 into engine 2 by name.

Nodes that only read weights (DequantizeLinear, Cast, Constant) go into each part
that uses them, so no weight crosses between the parts. Both parts point to the
weights file of the input graph, so nothing large is copied.

    python export/split_backbone.py --onnx build/onnx/hyvla-bb-w4/backbone.onnx --layer 16

This writes backbone_part1.onnx and backbone_part2.onnx next to the input.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import onnx
from onnx import helper

LAYER = re.compile(r"language_model\.model\.layers\.(\d+)\.")


def namespace(n) -> str:
    return {p.key: p.value for p in n.metadata_props}.get("namespace", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=16, help="first language layer of part 2 (default: %(default)s)")
    args = ap.parse_args()

    m = onnx.load(str(args.onnx), load_external_data=False)
    g = m.graph
    nodes = list(g.node)
    inits = {t.name: t for t in g.initializer}
    graph_inputs = {i.name for i in g.input}

    # Nodes whose inputs are only weights (or other such nodes) do not depend on
    # the camera frames or the tokens. They can go into any part.
    const_out: set[str] = set(inits)
    const_node = []
    for n in nodes:
        c = all(x == "" or x in const_out for x in n.input)
        const_node.append(c)
        if c:
            const_out.update(n.output)

    cut = next((i for i, n in enumerate(nodes) if not const_node[i]
                and (mt := LAYER.search(namespace(n))) and int(mt.group(1)) >= args.layer), None)
    if cut is None:
        print(f"no node of language layer {args.layer} found")
        return 1
    late = [i for i in range(cut) if not const_node[i] and (mt := LAYER.search(namespace(nodes[i])))
            and int(mt.group(1)) >= args.layer]
    assert not late, f"nodes of layer >= {args.layer} before the cut: {late[:5]}"

    part = [None if const_node[i] else (1 if i < cut else 2) for i in range(len(nodes))]
    producer = {o: i for i, n in enumerate(nodes) for o in n.output}

    def needed_consts(node_ids):
        """All weight-only nodes that the given nodes read, directly or not."""
        out, todo = set(), [x for i in node_ids for x in nodes[i].input]
        while todo:
            x = todo.pop()
            i = producer.get(x)
            if i is not None and const_node[i] and i not in out:
                out.add(i)
                todo.extend(nodes[i].input)
        return out

    p1 = [i for i in range(len(nodes)) if part[i] == 1]
    p2 = [i for i in range(len(nodes)) if part[i] == 2]
    outputs = [o.name for o in g.output]
    # Tensors made by part 1 and read by part 2 (or graph outputs made by part 1).
    p1_out = {o for i in p1 for o in nodes[i].output}
    cross = sorted({x for i in p2 for x in nodes[i].input if x in p1_out} | {o for o in outputs if o in p1_out})

    # Shapes and types of the crossing tensors, for the new inputs and outputs.
    info = {v.name: v for v in onnx.shape_inference.infer_shapes(m).graph.value_info}
    info.update({v.name: v for v in g.output})

    def value_info(name):
        if name in info:
            return info[name]
        raise SystemExit(f"no shape for crossing tensor {name}")

    def build(ids, inputs, outs, tag):
        ids = sorted(set(ids) | needed_consts(ids))
        used = {x for i in ids for x in nodes[i].input}
        gi = [i for i in g.input if i.name in used] + [value_info(x) for x in inputs]
        go = [value_info(x) for x in outs]
        init = [inits[x] for x in sorted(used) if x in inits]
        sub = helper.make_graph([nodes[i] for i in ids], f"{g.name}_{tag}", gi, go, init)
        mm = helper.make_model(sub, opset_imports=m.opset_import, ir_version=m.ir_version,
                               producer_name=m.producer_name)
        path = args.onnx.with_name(args.onnx.stem + f"_{tag}.onnx")
        onnx.save(mm, str(path))  # weights stay in the input's .onnx.data file (same folder)
        wbytes = sum(int(t.external_data[[e.key for e in t.external_data].index("length")].value)
                     if t.data_location == onnx.TensorProto.EXTERNAL else len(t.raw_data) for t in init)
        print(f"{path.name}: {len(ids)} nodes, {len(gi)} inputs, {len(go)} outputs, weights {wbytes / 1e9:.2f} GB")
        return mm

    p1_outs = [x for x in cross]
    p2_ins = [x for x in cross if x not in outputs or any(x in nodes[i].input for i in p2)]
    p2_outs = [o for o in outputs if o not in p1_out]
    build(p1, [], p1_outs, "part1")
    build(p2, p2_ins, p2_outs, "part2")
    passthrough = [o for o in outputs if o in p1_out]
    print(f"cut before language layer {args.layer}: {len(cross)} tensors go from part 1 to part 2")
    if passthrough:
        print(f"  outputs made by part 1: {passthrough}")
    unused = graph_inputs - {x for i in p1 + p2 for x in nodes[i].input}
    if unused:
        print(f"  graph inputs no node reads: {sorted(unused)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
