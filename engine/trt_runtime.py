"""TensorRT engine builder and runtime for Hy-VLA on the Jetson.

The classes follow the engine builder design spec (Table 3):

- HandoffPackage: reads Lam's manifest.json and checks every SHA-256 hash.
- EngineCache: the cache key and the engine files in build/engines/.
- EngineBuilder: parses one ONNX graph and builds an engine on the board.
- TrtEngine: loads one engine after a memory check and runs it on a stream.
- EngineChain: several engines that run as one graph (the backbone parts).
- HyVlaEnginePolicy: the backbone and the expert, and the Euler loop. The loop stays in
  Python, so the number of steps can change without a new build (TR-6).

Engines are always built on the board (TR-1) and never need the network (TR-8).
"""

from __future__ import annotations

import hashlib
import json
import mmap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "build" / "engines"
BUILDER_VERSION = 2  # bump when the build settings below change (2: int8 with --int8 --fp16)

NP_DTYPES = {"float16": np.float16, "float32": np.float32, "int64": np.int64, "int32": np.int32,
             "bool": np.bool_, "int8": np.int8}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def mem_available_mib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return float("nan")


class PackageError(RuntimeError):
    pass


@dataclass
class HandoffPackage:
    root: Path
    manifest: dict = field(default_factory=dict)

    @classmethod
    def open(cls, root: str | Path, verify: bool = True) -> "HandoffPackage":
        root = Path(root)
        path = root / "manifest.json"
        if not path.exists():
            raise PackageError(f"no manifest.json in {root}")
        m = json.loads(path.read_text())
        for key in ("name", "precision", "graphs", "flow", "reference", "sha256"):
            if key not in m:
                raise PackageError(f"manifest has no '{key}'")
        pkg = cls(root, m)
        if verify:
            pkg.verify()
        return pkg

    def verify(self) -> None:
        """Stop on a missing file or a wrong hash: a half-copied file or a mix of two exports."""
        for name, want in self.manifest["sha256"].items():
            f = self.root / name
            if not f.exists():
                raise PackageError(f"{name} is listed in the manifest but missing")
            got = sha256(f)
            if got != want:
                raise PackageError(f"{name}: sha256 {got[:12]}... does not match the manifest {want[:12]}...")
        for g in self.manifest["graphs"].values():
            for key in ("file", "data"):
                if g.get(key) and g[key] not in self.manifest["sha256"]:
                    raise PackageError(f"{g[key]} has no hash in the manifest")

    def graph(self, name: str) -> dict:
        return self.manifest["graphs"][name]

    def onnx_path(self, name: str) -> Path:
        return self.root / self.graph(name)["file"]

    def parts(self, name: str) -> list[str]:
        """The graphs that make up `name`. The real backbone has several parts, the test one has one."""
        return list(self.graph(name).get("parts", [name]))

    def precision(self, name: str) -> str:
        return self.manifest["precision"][name]

    def file_hashes(self, name: str) -> dict:
        g = self.graph(name)
        return {f: self.manifest["sha256"][f] for f in (g.get("file"), g.get("data")) if f}

    def reference(self) -> dict:
        return dict(np.load(self.root / self.manifest["reference"]["file"]))


class EngineCache:
    """An engine is reused only when nothing it depends on changed (TR-4)."""

    def __init__(self, root: Path = ENGINE_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(file_hashes: dict, precision: str, settings: dict) -> str:
        import tensorrt as trt

        blob = json.dumps({"files": file_hashes, "trt": trt.__version__, "precision": precision,
                           "settings": settings, "builder": BUILDER_VERSION}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.engine", self.root / f"{key}.json"

    def find(self, key: str) -> Path | None:
        engine, side = self.paths(key)
        return engine if engine.exists() and side.exists() else None


class EngineBuilder:
    def __init__(self, workspace_mib: int = 1024, verbose: bool = False):
        self.workspace_mib = workspace_mib
        self.verbose = verbose

    def settings(self) -> dict:
        return {"workspace_mib": self.workspace_mib}

    def build(self, onnx_path: Path, precision: str, engine_path: Path, sidecar_path: Path,
              extra: dict | None = None) -> dict:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.VERBOSE if self.verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        # INT4-weight Q/DQ graphs are built strongly typed: the graph itself says
        # the precision of every tensor. INT8 Q/DQ graphs use the int8 and fp16
        # flags, like trtexec --int8 --fp16. Then the layers between the Q/DQ
        # pairs (and the layers kept out of INT8) run in FP16, not FP32.
        strongly_typed = precision == "int4_weight"
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"[FAIL] TensorRT could not parse {onnx_path.name}:\n{errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_mib << 20)
        if not strongly_typed:
            config.set_flag(trt.BuilderFlag.FP16)
        if precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
        timing_path = engine_path.parent / "timing.cache"
        cache = config.create_timing_cache(timing_path.read_bytes() if timing_path.exists() else b"")
        config.set_timing_cache(cache, ignore_mismatch=True)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        t = time.perf_counter()
        plan = builder.build_serialized_network(network, config)
        build_s = time.perf_counter() - t
        if plan is None:
            raise RuntimeError(f"[FAIL] TensorRT build of {onnx_path.name} failed")
        engine_path.write_bytes(bytes(plan))
        timing_path.write_bytes(bytes(config.get_timing_cache().serialize()))
        side = {"onnx": onnx_path.name, "precision": precision, "trt": trt.__version__,
                "build_s": build_s, "engine_mib": engine_path.stat().st_size / 2**20,
                "settings": self.settings(), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}
        sidecar_path.write_text(json.dumps(side, indent=2) + "\n")
        return side


class TrtEngine:
    """One engine with static shapes, run on its own CUDA stream."""

    def __init__(self, engine_path: Path, min_free_mib: float = 300.0):
        import tensorrt as trt
        import torch

        self.torch = torch
        size_mib = engine_path.stat().st_size / 2**20
        free = mem_available_mib()
        # The file is memory-mapped, so its bytes stay in the page cache, which the
        # kernel can free. Only the engine itself needs new RAM.
        if free < size_mib + min_free_mib:
            raise MemoryError(f"[FAIL] {engine_path.name} needs about {size_mib + min_free_mib:.0f} MiB, "
                              f"only {free:.0f} MiB available")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            self.engine = self.runtime.deserialize_cuda_engine(mm)
        if self.engine is None:
            raise RuntimeError(f"[FAIL] TensorRT could not load {engine_path.name}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.inputs, self.outputs, self.buffers = [], [], {}
        to_torch = {trt.float16: torch.float16, trt.float32: torch.float32, trt.int64: torch.int64,
                    trt.int32: torch.int32, trt.bool: torch.bool, trt.int8: torch.int8}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = to_torch[self.engine.get_tensor_dtype(name)]
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            self.buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(name)
        self.device_mib = self.engine.device_memory_size / 2**20
        self.size_mib = size_mib

    def __call__(self, **feeds):
        """Copy the inputs in, run, and return the output buffers (on the GPU)."""
        torch = self.torch
        with torch.cuda.stream(self.stream):
            for name in self.inputs:
                src = feeds[name]
                if not isinstance(src, torch.Tensor):
                    src = torch.from_numpy(np.ascontiguousarray(src))
                self.buffers[name].copy_(src.to(self.buffers[name].dtype), non_blocking=True)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return {name: self.buffers[name] for name in self.outputs}


class EngineChain:
    """Engines that run one after the other as one graph.

    Every output goes into a pool of tensors by name. Each engine takes its
    inputs from the pool, so a later part can read a tensor from any earlier
    part (export/split_backbone.py keeps the names). It has the same call as
    TrtEngine.
    """

    def __init__(self, engines: list[TrtEngine], inputs: list[str], outputs: list[str]):
        self.engines, self.inputs, self.outputs = engines, list(inputs), list(outputs)
        self.torch = engines[0].torch
        self.size_mib = sum(e.size_mib for e in engines)
        self.device_mib = sum(e.device_mib for e in engines)

    def __call__(self, **feeds):
        pool = dict(feeds)
        for e in self.engines:
            pool.update(e(**{n: pool[n] for n in e.inputs}))
        return {n: pool[n] for n in self.outputs}


class HyVlaEnginePolicy:
    """The backbone engine once per chunk, then the expert engine `steps` times."""

    def __init__(self, backbone: TrtEngine, expert: TrtEngine, steps: int, ctx_map: dict):
        self.backbone, self.expert, self.steps = backbone, expert, steps
        self.ctx_map = ctx_map  # backbone output name -> expert input name

    def sample_chunk(self, backbone_feeds: dict, state, noise):
        torch = self.backbone.torch
        ctx = {self.ctx_map[k]: v for k, v in self.backbone(**backbone_feeds).items() if k in self.ctx_map}
        x = torch.as_tensor(noise, device="cuda", dtype=torch.float32).clone()
        dt = -1.0 / self.steps
        t = 1.0
        out_name = self.expert.outputs[0]
        for _ in range(self.steps):
            v = self.expert(state=state, x_t=x, t=np.array([t], np.float32), **ctx)[out_name]
            x = x + dt * v.float()
            t += dt
        return x
