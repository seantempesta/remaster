"""
Build TensorRT engines using modern strongly-typed API + ModelOpt.

Replaces deprecated IInt8Calibrator / BuilderFlag.FP16 / layer.precision
with ModelOpt AutoCast (FP16) and Q/DQ quantization (INT8). TRT 11-ready.

Usage:
  python tools/build_engine.py fp16          # FP16 engine from FP16 ONNX
  python tools/build_engine.py int8          # INT8 Q/DQ engine (ModelOpt quantization)
  python tools/build_engine.py int8 --pure   # Full INT8 (no FP16 fallback layers)

Outputs go to checkpoints/drunet_student/ by default.
"""
import argparse
import os
import sys
import random
import time
from pathlib import Path

import numpy as np
import cv2

# ModelOpt checks PATH for cuDNN DLLs on Windows. PyTorch bundles them in
# torch/lib/ but that dir isn't on PATH by default, so ModelOpt fails to
# find them and falls back to CPU-only calibration (extremely slow at 1080p).
# Fix: add torch's lib dir to PATH before importing modelopt.
if sys.platform == "win32":
    try:
        import torch as _torch
        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if _torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "drunet_student"
ORIGINALS_DIR = PROJECT_ROOT / "data" / "originals"


def collect_calibration_data(num_frames=200, target_h=1080, target_w=1920):
    """Load calibration frames as numpy array [N, 3, H, W] float32."""
    all_frames = sorted(ORIGINALS_DIR.glob("*.png"))
    print(f"Found {len(all_frames)} original frames")

    random.seed(42)
    selected = random.sample(all_frames, min(num_frames, len(all_frames)))
    print(f"Selected {len(selected)} for calibration")

    data = np.zeros((len(selected), 3, target_h, target_w), dtype=np.float32)
    for i, path in enumerate(selected):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        crop = img[:min(h, target_h), :min(w, target_w)]
        if crop.shape[0] < target_h or crop.shape[1] < target_w:
            crop = cv2.copyMakeBorder(
                crop,
                0, max(0, target_h - crop.shape[0]),
                0, max(0, target_w - crop.shape[1]),
                cv2.BORDER_REPLICATE,
            )
        data[i] = crop.astype(np.float32).transpose(2, 0, 1) / 255.0
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(selected)} frames")

    print(f"Calibration data: {data.shape}")
    return data


def build_fp16_engine(onnx_path, output_path):
    """Build FP16 engine using ModelOpt AutoCast + strongly typed API."""
    import tensorrt as trt
    import onnx
    from modelopt.onnx.autocast import convert_to_mixed_precision

    print(f"=== Building FP16 engine (strongly typed) ===")
    print(f"Input ONNX: {onnx_path}")

    # Step 1: AutoCast to mixed FP32/FP16
    # Save a sample frame as NPZ so AutoCast knows dynamic shape dims
    print("Running ModelOpt AutoCast (FP32 -> mixed FP16)...")
    mixed_onnx_path = str(onnx_path).replace(".onnx", "_autocast_fp16.onnx")
    import tempfile
    calib_npz = os.path.join(tempfile.gettempdir(), "autocast_calib.npz")
    sample = np.random.randn(1, 3, 1080, 1920).astype(np.float16)
    np.savez(calib_npz, input=sample)
    model = convert_to_mixed_precision(
        str(onnx_path),
        low_precision_type="fp16",
        calibration_data=calib_npz,
    )
    os.remove(calib_npz)
    onnx.save(model, mixed_onnx_path)
    size_mb = os.path.getsize(mixed_onnx_path) / 1024**2
    print(f"  AutoCast ONNX: {mixed_onnx_path} ({size_mb:.1f}MB)")

    # Step 2: Build engine with strongly typed API
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing AutoCast ONNX...")
    with open(mixed_onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    # No BuilderFlag.FP16 -- strongly typed mode reads precision from ONNX

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 1080, 1920), (1, 3, 1080, 1920), (1, 3, 1080, 1920))
    config.add_optimization_profile(profile)

    print("Building engine...")
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    dt = time.time() - t0

    if engine is None:
        raise RuntimeError("Engine build failed")

    with open(output_path, "wb") as f:
        f.write(engine)

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Engine built in {dt:.0f}s: {output_path} ({size_mb:.1f}MB)")

    # Clean up intermediate
    os.remove(mixed_onnx_path)
    return output_path


def build_int8_engine(onnx_path, output_path, num_frames=20, pure=False):
    """Build INT8 engine using ModelOpt Q/DQ quantization + strongly typed API.

    By default, excludes Add ops from INT8 (verified: pure INT8 = 26 dB, excluding
    Adds = 67 dB). Use --pure to quantize everything including Adds.
    """
    import tensorrt as trt
    import onnx
    import modelopt.onnx.quantization as moq

    print(f"=== Building INT8 engine (strongly typed, ModelOpt Q/DQ) ===")
    print(f"Input ONNX: {onnx_path}")
    print(f"Calibration frames: {num_frames}")

    # Step 1: Collect calibration data
    calib_data = collect_calibration_data(num_frames)

    # Step 2: Quantize with ModelOpt
    print("Running ModelOpt INT8 quantization...")
    qdq_onnx_path = str(onnx_path).replace(".onnx", "_int8_qdq.onnx")

    # Skip-connection Add ops crush fine detail when quantized to INT8.
    # Verified: pure INT8 = 26 dB, excluding Adds = 67 dB. This is architectural,
    # not a calibration issue — the encoder/decoder paths have mismatched distributions
    # at the addition points.
    op_types_to_exclude = None if pure else ["Add"]

    moq.quantize(
        onnx_path=str(onnx_path),
        calibration_data={"input": calib_data},
        output_path=qdq_onnx_path,
        quantize_mode="int8",
        calibration_method="entropy",
        calibration_eps=["cuda:0"],  # Don't try CPU + TRT EPs
        op_types_to_exclude=op_types_to_exclude,
    )

    size_mb = os.path.getsize(qdq_onnx_path) / 1024**2
    print(f"  Q/DQ ONNX: {qdq_onnx_path} ({size_mb:.1f}MB)")

    # Step 3: Build engine with strongly typed API
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing Q/DQ ONNX...")
    with open(qdq_onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    # No BuilderFlag.INT8, no calibrator -- Q/DQ nodes handle everything

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 1080, 1920), (1, 3, 1080, 1920), (1, 3, 1080, 1920))
    config.add_optimization_profile(profile)

    print("Building engine...")
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    dt = time.time() - t0

    if engine is None:
        raise RuntimeError("Engine build failed")

    with open(output_path, "wb") as f:
        f.write(engine)

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Engine built in {dt:.0f}s: {output_path} ({size_mb:.1f}MB)")

    # Keep the Q/DQ ONNX for inspection
    print(f"Q/DQ ONNX saved: {qdq_onnx_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines (modern strongly-typed API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["fp16", "int8"],
                        help="Engine precision mode")
    parser.add_argument("--onnx", default=None,
                        help="Input ONNX path (default: auto-detect from checkpoints)")
    parser.add_argument("--output", default=None,
                        help="Output engine path (default: auto from mode)")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Number of calibration frames for INT8 (default: 20)")
    parser.add_argument("--pure", action="store_true",
                        help="INT8: quantize ALL layers including Adds (lower quality, faster)")
    args = parser.parse_args()

    if args.mode == "fp16":
        onnx_path = args.onnx or str(CHECKPOINT_DIR / "drunet_student.onnx")
        output = args.output or str(CHECKPOINT_DIR / "drunet_student_1080p_fp16.engine")
        build_fp16_engine(onnx_path, output)

    elif args.mode == "int8":
        onnx_path = args.onnx or str(CHECKPOINT_DIR / "drunet_student_fp32.onnx")
        output = args.output or str(CHECKPOINT_DIR / "drunet_student_1080p_int8.engine")
        build_int8_engine(onnx_path, output, args.num_frames, args.pure)


if __name__ == "__main__":
    main()
