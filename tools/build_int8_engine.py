"""
Build a properly calibrated INT8 TensorRT engine using real training frames.

Uses TensorRT Python API with a custom IInt8EntropyCalibrator2 that feeds
actual source frames (from data/originals/) through the network to determine
optimal quantization ranges per tensor.

Usage:
  python tools/build_int8_engine.py
  python tools/build_int8_engine.py --num-frames 200
"""
import argparse
import os
import sys
import random
import time
from pathlib import Path

import numpy as np
import cv2

# TensorRT Python API
import tensorrt as trt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class FrameCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator that feeds real source frames via PyTorch CUDA."""

    def __init__(self, frame_paths, batch_size=1, target_h=1080, target_w=1920,
                 cache_file="calibration.cache"):
        super().__init__()
        self.frame_paths = frame_paths
        self.batch_size = batch_size
        self.target_h = target_h
        self.target_w = target_w
        self.cache_file = cache_file
        self.current_idx = 0

        import torch
        # Allocate persistent GPU buffer via PyTorch
        self.device_buffer = torch.zeros(
            (batch_size, 3, target_h, target_w),
            dtype=torch.float32, device="cuda"
        )
        print(f"Calibrator: {len(frame_paths)} frames, batch_size={batch_size}")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Feed next batch of real frames."""
        if self.current_idx >= len(self.frame_paths):
            return None

        import torch
        host_data = np.zeros(
            (self.batch_size, 3, self.target_h, self.target_w), dtype=np.float32
        )

        for i in range(self.batch_size):
            idx = self.current_idx + i
            if idx >= len(self.frame_paths):
                break

            img = cv2.imread(str(self.frame_paths[idx]))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            padded = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
            padded[:min(h, self.target_h), :min(w, self.target_w)] = \
                img[:self.target_h, :self.target_w]
            host_data[i] = padded.astype(np.float32).transpose(2, 0, 1) / 255.0

        self.current_idx += self.batch_size

        # Copy to GPU via PyTorch, return raw CUDA pointer
        self.device_buffer.copy_(torch.from_numpy(host_data))
        if (self.current_idx % 50) < self.batch_size:
            print(f"  Calibrating: {self.current_idx}/{len(self.frame_paths)}")
        return [self.device_buffer.data_ptr()]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"  Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"  Writing calibration cache: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine(onnx_path, output_engine, calibrator):
    """Build INT8 TRT engine with proper calibration."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Enable FP16 + INT8
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    # Set optimization profile for static shape
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 1080, 1920), (1, 3, 1080, 1920), (1, 3, 1080, 1920))
    config.add_optimization_profile(profile)

    print("Building INT8 engine (this takes 3-10 minutes)...")
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    dt = time.time() - t0

    if engine is None:
        raise RuntimeError("Engine build failed")

    print(f"Engine built in {dt:.0f}s")

    # Save
    with open(output_engine, "wb") as f:
        f.write(engine)

    size_mb = os.path.getsize(output_engine) / 1024**2
    print(f"Saved: {output_engine} ({size_mb:.1f} MB)")

    return output_engine


def main():
    parser = argparse.ArgumentParser(description="Build properly calibrated INT8 TRT engine")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--onnx", default=str(PROJECT_ROOT / "checkpoints" / "drunet_student" / "drunet_student.onnx"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "checkpoints" / "drunet_student" / "drunet_student_1080p_int8.engine"))
    args = parser.parse_args()

    cache_file = args.output.replace(".engine", "_calib.cache")

    # Collect calibration frames from originals
    orig_dir = PROJECT_ROOT / "data" / "originals"
    all_frames = sorted(orig_dir.glob("*.png"))
    print(f"Found {len(all_frames)} original frames")

    random.seed(42)
    selected = random.sample(all_frames, min(args.num_frames, len(all_frames)))
    print(f"Selected {len(selected)} for calibration")

    calibrator = FrameCalibrator(selected, cache_file=cache_file)
    build_int8_engine(args.onnx, args.output, calibrator)

    print(f"\nCalibration cache saved: {cache_file}")
    print("This cache can be reused by trtexec or the C++ pipeline.")


if __name__ == "__main__":
    main()
