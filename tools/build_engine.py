"""
Portable TensorRT engine builder. Auto-detects GPU and builds optimal engines.

Usage:
  python tools/build_engine.py                    # FP16, auto-detect everything
  python tools/build_engine.py --int8             # INT8 with auto-found calib cache
  python tools/build_engine.py --shape 720x1280   # Different resolution
  python tools/build_engine.py --trtexec /path/to/trtexec
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def detect_gpu():
    """Return (name, free_mb, compute_cap) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        name = parts[0]
        free_mb = int(float(parts[1]))
        cc = parts[2]
        return name, free_mb, cc
    except Exception as e:
        print(f"WARNING: Could not detect GPU via nvidia-smi: {e}")
        return "Unknown GPU", 4096, "8.6"


def find_trtexec(override=None):
    """Find trtexec binary. Check project path, then PATH."""
    if override:
        p = Path(override)
        if p.is_file():
            return str(p)
        print(f"ERROR: --trtexec path not found: {override}")
        sys.exit(1)

    # Project-local trtexec
    project_path = PROJECT_ROOT / "tools" / "vs" / "vs-plugins" / "vsmlrt-cuda" / "trtexec.exe"
    if project_path.is_file():
        return str(project_path)

    # System PATH
    found = shutil.which("trtexec")
    if found:
        return found

    print("ERROR: trtexec not found. Locations checked:")
    print(f"  - {project_path}")
    print("  - System PATH")
    print("Use --trtexec PATH to specify location.")
    sys.exit(1)


def build_engine(args):
    gpu_name, free_mb, cc = detect_gpu()
    print(f"GPU: {gpu_name} (compute {cc}, {free_mb} MB free)")

    trtexec = find_trtexec(args.trtexec)
    print(f"trtexec: {trtexec}")

    # Resolve ONNX path
    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = PROJECT_ROOT / onnx_path
    if not onnx_path.is_file():
        print(f"ERROR: ONNX file not found: {onnx_path}")
        sys.exit(1)

    # Parse shape
    m = re.match(r"(\d+)x(\d+)", args.shape)
    if not m:
        print(f"ERROR: Invalid --shape format '{args.shape}', expected HxW (e.g. 1080x1920)")
        sys.exit(1)
    h, w = int(m.group(1)), int(m.group(2))
    shape_str = f"input:1x3x{h}x{w}"

    # Determine precision and output name
    precision = "int8" if args.int8 else "fp16"
    onnx_dir = onnx_path.parent
    stem = onnx_path.stem  # e.g. drunet_student

    if args.output:
        engine_path = Path(args.output)
    else:
        res_tag = f"{h}p" if w == 1920 else f"{h}x{w}"
        engine_path = onnx_dir / f"{stem}_{res_tag}_{precision}.engine"

    timing_cache = onnx_dir / f"{stem}_timing.cache"

    # Workspace: min(60% free VRAM, 4096 MB)
    workspace_mb = min(int(free_mb * 0.6), 4096)

    # Build command
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--shapes={shape_str}",
        f"--saveEngine={engine_path}",
        f"--timingCacheFile={timing_cache}",
        f"--memPoolSize=workspace:{workspace_mb}M",
        "--builderOptimizationLevel=5",
        "--tilingOptimizationLevel=3",
        "--tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN,+EDGE_MASK_CONVOLUTIONS,+JIT_CONVOLUTIONS",
        "--persistentCacheRatio=0.5",
        "--avgTiming=16",
        "--useCudaGraph",
    ]

    # Precision flags
    if args.int8:
        # INT8 mode: also enable FP16 as fallback for layers that don't quantize well
        cmd.extend(["--fp16", "--int8"])

        # Find calibration cache
        calib_path = None
        if args.calib:
            calib_path = Path(args.calib)
        else:
            # Auto-find next to ONNX
            for pattern in [f"{stem}*int8*calib*.cache", "*int8*calib*.cache", "*calibration*.cache"]:
                candidates = sorted(onnx_dir.glob(pattern))
                if candidates:
                    calib_path = candidates[0]
                    break

        if not calib_path or not calib_path.is_file():
            print("ERROR: INT8 requires calibration cache. Not found.")
            print(f"  Searched in: {onnx_dir}")
            print("  Use --calib PATH or run: python tools/build_int8_engine.py")
            sys.exit(1)

        cmd.append(f"--calib={calib_path}")
        print(f"INT8 calibration: {calib_path}")
    else:
        cmd.append("--fp16")

    print(f"\nBuilding {precision.upper()} engine:")
    print(f"  ONNX:      {onnx_path}")
    print(f"  Shape:     {shape_str}")
    print(f"  Output:    {engine_path}")
    print(f"  Workspace: {workspace_mb} MB")
    print(f"  Cache:     {timing_cache}")
    print()

    # Run trtexec
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\nERROR: trtexec failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Summary
    engine_size_mb = engine_path.stat().st_size / (1024 * 1024) if engine_path.is_file() else 0
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  GPU:        {gpu_name}")
    print(f"  Precision:  {precision.upper()}")
    print(f"  Engine:     {engine_path}")
    print(f"  Size:       {engine_size_mb:.1f} MB")
    print(f"  Timing:     {timing_cache}")

    # Rough speed estimates for DRUNet student on common GPUs
    speed_estimates = {
        "fp16": {"3060": "50-55", "3070": "60-70", "3080": "75-85", "4060": "60-70",
                 "4070": "80-90", "4080": "100-120", "4090": "140-160"},
        "int8": {"3060": "55-65", "3070": "70-85", "3080": "90-105", "4060": "70-85",
                 "4070": "95-110", "4080": "120-145", "4090": "170-200"},
    }
    gpu_short = gpu_name.lower()
    for key, speed in speed_estimates.get(precision, {}).items():
        if key in gpu_short:
            print(f"  Est speed:  ~{speed} fps raw inference (1080p)")
            break
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine with optimal settings for current GPU"
    )
    parser.add_argument("--onnx", default="checkpoints/drunet_student/drunet_student.onnx",
                        help="Path to ONNX model (default: student checkpoint)")
    parser.add_argument("--output", "-o", help="Output engine path (auto-named if omitted)")
    parser.add_argument("--shape", default="1080x1920",
                        help="Input resolution as HxW (default: 1080x1920)")
    parser.add_argument("--int8", action="store_true",
                        help="Build INT8 engine (default: FP16)")
    parser.add_argument("--calib", help="INT8 calibration cache path (auto-found if omitted)")
    parser.add_argument("--trtexec", help="Path to trtexec binary (auto-found if omitted)")

    args = parser.parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()
