#!/usr/bin/env python3
"""Camera bring-up check: does a USB camera deliver frames fast enough, and cheaply?

Opens one V4L2 camera with OpenCV, the same backend LeRobot's OpenCVCamera
uses, and measures what the policy loop will actually pay for:

- achieved frame rate against the requested one
- CPU load of capture plus decode, which competes with inference on this board
- preprocessing, from a raw frame to a 224x224 RGB tensor on the GPU, which
  guide 05 says belongs inside end-to-end latency

No frame is written to disk or displayed. The only image statistic reported is
mean brightness, to catch a camera that streams black frames.

    python3 hardware/hello_camera.py
    python3 hardware/hello_camera.py --device /dev/v4l/by-id/usb-...-video-index0
    python3 hardware/hello_camera.py --fourcc YUYV --width 640 --height 480
    python3 hardware/hello_camera.py --json results/camera_brio_mjpg_640.json

Use the /dev/v4l/by-id path in LeRobot configs. It is keyed on the camera's
serial number, so it survives reboots and replugging where /dev/videoN does not.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = math.ceil(p / 100.0 * len(ordered)) - 1
    return ordered[max(0, min(len(ordered) - 1, idx))]


def summarize(values: list[float]) -> dict:
    return {"mean_ms": statistics.fmean(values), "p50_ms": pct(values, 50),
            "p99_ms": pct(values, 99), "max_ms": max(values)}


def default_device() -> str:
    # The first capture node of the first camera, by serial number.
    hits = sorted(Path("/dev/v4l/by-id").glob("*-video-index0"))
    return str(hits[0]) if hits else "/dev/video0"


def fourcc_name(code: float) -> str:
    code = int(code)
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--fourcc", default="MJPG", help="MJPG or YUYV")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=150, help="frames to time")
    parser.add_argument("--warmup-s", type=float, default=1.0,
                        help="seconds of frames discarded while exposure settles")
    parser.add_argument("--size", type=int, default=224, help="model input resolution")
    parser.add_argument("--interp", choices=("area", "linear"), default="area",
                        help="resize filter. Match what the policy was trained with. "
                             "area is ~3x slower than linear at 640x480, ~14x at 1080p")
    parser.add_argument("--json", type=Path, help="write the measurement record here")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("[FAIL] cv2 not importable. Use an environment with "
              "opencv-python-headless, see guide 03", file=sys.stderr)
        return 1
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        torch, cuda = None, False

    interp = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}[args.interp]
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[FAIL] cannot open {args.device}", file=sys.stderr)
        return 1

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        actual = {
            "fourcc": fourcc_name(cap.get(cv2.CAP_PROP_FOURCC)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
        }
        print(f"[ok]   opened {args.device}")
        print(f"       asked {args.fourcc} {args.width}x{args.height}@{args.fps:g}, "
              f"got {actual['fourcc']} {actual['width']}x{actual['height']}@{actual['fps']:g}")
        if actual["fourcc"] != args.fourcc:
            print(f"[warn] the driver refused {args.fourcc}. Check the modes with "
                  "`gst-device-monitor-1.0 Video/Source` or `v4l2-ctl --list-formats-ext`")

        start = time.perf_counter()
        while time.perf_counter() - start < args.warmup_s:
            cap.read()

        reads, prep, stamps, brightness = [], [], [], []
        cpu0, wall0 = time.process_time(), time.perf_counter()
        for _ in range(args.frames):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()
            if not ok or frame is None:
                print("[FAIL] read returned no frame", file=sys.stderr)
                return 1
            reads.append((t1 - t0) * 1000.0)
            stamps.append(t1)
            brightness.append(float(frame.mean()))

            # What a policy wrapper does before the forward pass.
            small = cv2.resize(frame, (args.size, args.size), interpolation=interp)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            if torch is not None:
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
                if cuda:
                    tensor = tensor.cuda(non_blocking=False)
                    torch.cuda.synchronize()
            prep.append((time.perf_counter() - t1) * 1000.0)
        cpu_s, wall_s = time.process_time() - cpu0, time.perf_counter() - wall0
    finally:
        cap.release()

    achieved = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    cpu_pct = 100.0 * cpu_s / wall_s
    mean_level = statistics.fmean(brightness)

    status = "ok" if achieved >= 0.9 * actual["fps"] else "warn"
    tag = "[ok]  " if status == "ok" else "[warn]"
    print(f"{tag} achieved {achieved:.1f} fps over {args.frames} frames")
    print(f"[ok]   read() blocks p50 {pct(reads, 50):.1f} ms, p99 {pct(reads, 99):.1f} ms "
          "(waiting for the next frame, mostly)")
    print(f"[ok]   capture + decode + preprocess CPU: {cpu_pct:.0f}% of one core")
    target = "GPU tensor" if cuda else ("CPU tensor" if torch is not None else "RGB array")
    print(f"[ok]   preprocess ({args.interp}) to {args.size}x{args.size} {target}: "
          f"p50 {pct(prep, 50):.2f} ms, p99 {pct(prep, 99):.2f} ms")
    level_tag = "[ok]  " if mean_level > 8 else "[warn]"
    print(f"{level_tag} mean frame brightness {mean_level:.0f}/255"
          + ("" if mean_level > 8 else "  <- frames look black"))

    if args.json:
        record = {
            "schema": "camera/v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "requested": {"fourcc": args.fourcc, "width": args.width,
                          "height": args.height, "fps": args.fps},
            "actual": actual,
            "frames": args.frames,
            "achieved_fps": achieved,
            "read_block": summarize(reads),
            "preprocess": {"size": args.size, "interp": args.interp, "target": target,
                           **summarize(prep)},
            "process_cpu_pct_of_one_core": cpu_pct,
            "mean_brightness": mean_level,
            "opencv": cv2.__version__,
            "torch": getattr(torch, "__version__", None),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
