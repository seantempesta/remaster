"""
NAFNet local inference with optional INT8 quantization.

Designed for RTX 3060 (6GB VRAM). Separate from the cloud pipeline
(denoise_nafnet.py / modal_denoise.py) — this is a standalone local path.

Supports:
  - fp16 baseline inference
  - TorchAO INT8 weight-only quantization (with fallback to PyTorch native)
  - TensorRT INT8 inference (ONNX export -> TRT engine with INT8 calibration)
  - torch.compile acceleration
  - Eval mode for quality comparison on validation pairs
  - Video processing with PyAV decode + ffmpeg encode

Usage:
    # Eval mode — compare fp16 vs INT8 quality
    python pipelines/denoise_local.py --eval --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize none
    python pipelines/denoise_local.py --eval --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize int8

    # TensorRT INT8 (builds engine on first run, cached for subsequent runs)
    python pipelines/denoise_local.py --eval --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize tensorrt
    python pipelines/denoise_local.py --input video.mkv --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize tensorrt

    # Process a video
    python pipelines/denoise_local.py --input video.mkv --output denoised.mkv --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize int8

    # With torch.compile
    python pipelines/denoise_local.py --input video.mkv --checkpoint checkpoints/nafnet_distill/safe/nafnet_best.pth --quantize int8 --compile
"""
import sys
import os
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch._inductor.config
torch._inductor.config.conv_1x1_as_mm = True
from lib.nafnet_arch import NAFNet, swap_layernorm_for_compile, swap_layernorm_for_export
from lib.ffmpeg_utils import get_ffmpeg, get_video_info

DEVICE = "cuda"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_PAIRS_DIR = PROJECT_ROOT / "data" / "val_pairs"
TRAIN_PAIRS_DIR = PROJECT_ROOT / "data" / "train_pairs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "nafnet_distill"

# TensorRT fixed input shape: 1080p padded to multiple of 16
TRT_INPUT_H = 1088
TRT_INPUT_W = 1920


def load_nafnet(checkpoint_path, device="cuda", fp16=True):
    """Load NAFNet-width64 from checkpoint (no compile or quantize yet)."""
    model = NAFNet(
        img_channel=3, width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("model", ckpt.get("state_dict", ckpt))))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device)
    if fp16:
        model = model.half()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  NAFNet-width64: {params_m:.1f}M params, VRAM after load: {vram_mb:.0f}MB")

    return model


###############################################################################
# TensorRT INT8 support
###############################################################################

def _get_gpu_tag():
    """Return a short GPU identifier for engine filenames (e.g. 'RTX3060')."""
    name = torch.cuda.get_device_name(0)
    # Strip common prefixes and spaces for a compact filename-safe tag
    tag = name.replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "")
    return tag


def _onnx_path():
    return CHECKPOINTS_DIR / "nafnet_w64_fp16.onnx"


def _engine_path():
    tag = _get_gpu_tag()
    return CHECKPOINTS_DIR / f"nafnet_trt_int8_{tag}.engine"


def _calib_cache_path():
    return CHECKPOINTS_DIR / f"nafnet_int8_calib.cache"


def export_onnx(checkpoint_path):
    """Export NAFNet to ONNX with fixed 1088x1920 input shape.

    Uses swap_layernorm_for_export() to replace custom autograd LayerNorm2d
    with standard-op LayerNorm2dExport that ONNX/TRT can handle.

    Returns the path to the exported ONNX file.
    """
    onnx_path = _onnx_path()
    if onnx_path.exists():
        print(f"  ONNX model already exists: {onnx_path}")
        return onnx_path

    print(f"  Exporting NAFNet to ONNX...")

    model = NAFNet(
        img_channel=3, width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("model", ckpt.get("state_dict", ckpt))))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Critical: swap custom LayerNorm2d for export-safe version
    model = swap_layernorm_for_export(model)

    # Export on CPU in float32 to avoid VRAM OOM. JIT tracing keeps ALL
    # intermediate tensors alive simultaneously, which exceeds 6GB VRAM
    # even in fp16 for a 116M-param model at 1088x1920, and exceeds 8GB
    # system RAM in fp32 at 1088x1920.
    #
    # Solution: trace at smaller resolution (544x960) with dynamic spatial
    # axes. TRT will optimize for our target 1088x1920 at engine build time
    # using optimization profiles with fixed shapes.
    model.float().cpu()

    # Half-resolution for tracing — convolutions are resolution-independent
    trace_h, trace_w = TRT_INPUT_H // 2, TRT_INPUT_W // 2
    dummy = torch.randn(1, 3, trace_h, trace_w, dtype=torch.float32)

    print(f"  Tracing model on CPU at {trace_h}x{trace_w} (this may take a minute)...")
    torch.onnx.export(
        model, dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "out_height", 3: "out_width"},
        },
    )

    del model, dummy
    import gc
    gc.collect()

    file_mb = onnx_path.stat().st_size / 1024**2
    print(f"  ONNX exported: {onnx_path} ({file_mb:.1f} MB)")

    # Optional: validate with onnxruntime
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model, full_check=True)
        print(f"  ONNX model validation passed (opset {onnx_model.opset_import[0].version})")
        del onnx_model
    except ImportError:
        print(f"  onnx package not installed, skipping validation (pip install onnx)")
    except Exception as e:
        print(f"  ONNX validation warning: {e}")

    return onnx_path


def _collect_calibration_frames(num_frames=200):
    """Collect calibration frames from val_pairs and train_pairs.

    Returns list of numpy arrays, each (3, TRT_INPUT_H, TRT_INPUT_W) float16.
    Uses all val frames + random subset of train frames to reach num_frames.
    """
    from PIL import Image

    frames = []

    # All val frames first
    val_input = VAL_PAIRS_DIR / "input"
    if val_input.exists():
        for p in sorted(val_input.glob("*.png")):
            img = np.array(Image.open(str(p)).convert("RGB"))
            h, w = img.shape[:2]
            padded = np.zeros((TRT_INPUT_H, TRT_INPUT_W, 3), dtype=np.uint8)
            padded[:min(h, TRT_INPUT_H), :min(w, TRT_INPUT_W), :] = img[:min(h, TRT_INPUT_H), :min(w, TRT_INPUT_W)]
            t = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames.append(t)
        print(f"  Calibration: loaded {len(frames)} val frames")

    # Fill remaining with train frames
    train_input = TRAIN_PAIRS_DIR / "input"
    remaining = num_frames - len(frames)
    if remaining > 0 and train_input.exists():
        train_files = sorted(train_input.glob("*.png"))
        # Deterministic subset — take evenly spaced frames for diversity
        if len(train_files) > remaining:
            step = len(train_files) / remaining
            indices = [int(i * step) for i in range(remaining)]
            train_files = [train_files[i] for i in indices]
        for p in train_files[:remaining]:
            img = np.array(Image.open(str(p)).convert("RGB"))
            h, w = img.shape[:2]
            padded = np.zeros((TRT_INPUT_H, TRT_INPUT_W, 3), dtype=np.uint8)
            padded[:min(h, TRT_INPUT_H), :min(w, TRT_INPUT_W), :] = img[:min(h, TRT_INPUT_H), :min(w, TRT_INPUT_W)]
            t = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames.append(t)
        print(f"  Calibration: loaded {len(frames)} total frames ({len(frames) - remaining} val + train)")

    if not frames:
        raise RuntimeError(
            f"No calibration frames found! Need PNG files in:\n"
            f"  {val_input}\n  {train_input}"
        )

    print(f"  Calibration: {len(frames)} frames ready")
    return frames


def build_tensorrt_engine(onnx_path, num_calib_frames=200):
    """Build a TensorRT INT8+FP16 engine from the ONNX model.

    Uses IInt8EntropyCalibrator2 with frames from val_pairs and train_pairs.
    The engine is cached to disk; subsequent calls load from cache if the
    engine file exists.

    Returns the path to the serialized engine file.
    """
    import tensorrt as trt

    engine_path = _engine_path()
    calib_cache = _calib_cache_path()

    if engine_path.exists():
        print(f"  TRT engine already cached: {engine_path}")
        return engine_path

    print(f"\n  Building TensorRT INT8 engine (this takes several minutes)...")
    print(f"  ONNX: {onnx_path}")
    print(f"  Engine will be saved to: {engine_path}")

    # Collect calibration data
    calib_frames = _collect_calibration_frames(num_calib_frames)

    # Define calibrator class (must be defined here since tensorrt import is conditional)
    class NAFNetCalibrator(trt.IInt8EntropyCalibrator2):
        """INT8 entropy calibrator using pre-loaded frames.

        Uses PyTorch CUDA tensors for device memory to avoid pycuda dependency.
        """
        def __init__(self, frames, cache_file):
            super().__init__()
            self.frames = frames
            self.cache_file = str(cache_file)
            self.current_index = 0
            self.batch_size = 1
            # Allocate device buffer using PyTorch (avoids pycuda)
            self.device_input = torch.zeros(
                1, 3, TRT_INPUT_H, TRT_INPUT_W,
                dtype=torch.float32, device="cuda"
            )

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.current_index >= len(self.frames):
                return None
            # Copy frame data into the device tensor
            frame_tensor = torch.from_numpy(self.frames[self.current_index]).unsqueeze(0)
            self.device_input.copy_(frame_tensor)
            self.current_index += 1
            if self.current_index % 50 == 0:
                print(f"    Calibrating... {self.current_index}/{len(self.frames)}")
            return [self.device_input.data_ptr()]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                print(f"  Using cached calibration data: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            print(f"  Calibration cache saved: {self.cache_file}")

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # Parse ONNX
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    print(f"  ONNX parsed: {network.num_layers} layers, "
          f"input={network.get_input(0).shape}, output={network.get_output(0).shape}")

    # Configure builder for INT8 + FP16 mixed precision
    config = builder.create_builder_config()
    # 1 GB workspace — RTX 3060 has 6GB, model+calibration uses ~4GB already
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    # TensorRT automatically uses mixed FP16/INT8 precision — it keeps layers
    # in FP16 where INT8 would cause too much quantization error (intro/ending
    # convs, LayerNorm ops, small channel attention convs). The combination of
    # FP16 + INT8 flags enables this automatic mixed-precision behavior.

    # Optimization profile: ONNX was exported with dynamic H/W axes (to avoid
    # OOM during tracing). Set fixed 1088x1920 for best TRT kernel selection.
    profile = builder.create_optimization_profile()
    input_shape = (1, 3, TRT_INPUT_H, TRT_INPUT_W)
    profile.set_shape("input", min=input_shape, opt=input_shape, max=input_shape)
    config.add_optimization_profile(profile)

    # Set up calibrator. The int8_calibrator property is deprecated in TRT 10.1+
    # in favor of "explicit quantization", but still works and is the simplest
    # path for PTQ INT8 calibration. Explicit quantization requires manually
    # inserting Q/DQ nodes in the ONNX graph — not worth the complexity here.
    calibrator = NAFNetCalibrator(calib_frames, calib_cache)
    config.int8_calibrator = calibrator

    # Build serialized engine
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed!")

    # Save engine
    with open(str(engine_path), "wb") as f:
        f.write(serialized_engine)

    engine_mb = engine_path.stat().st_size / 1024**2
    print(f"  Engine built in {build_time:.0f}s, saved: {engine_path} ({engine_mb:.1f} MB)")

    # Cleanup calibrator device memory
    del calibrator, calib_frames
    torch.cuda.empty_cache()

    return engine_path


class TRTModel:
    """TensorRT inference wrapper that mimics a PyTorch model interface.

    Takes a PyTorch CUDA tensor as input, returns a PyTorch CUDA tensor.
    Uses PyTorch tensors for device memory management (no pycuda needed).
    """

    def __init__(self, engine_path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        print(f"  Loading TRT engine: {engine_path}")
        with open(str(engine_path), "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Pre-allocate I/O tensors on GPU — match engine dtype
        # Query actual dtypes from engine (may be fp32 even with INT8 engine)
        in_dtype = trt.nptype(self.engine.get_tensor_dtype("input"))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype("output"))
        torch_in_dtype = torch.float32 if in_dtype == np.float32 else torch.float16
        torch_out_dtype = torch.float32 if out_dtype == np.float32 else torch.float16
        self.input_tensor = torch.zeros(
            1, 3, TRT_INPUT_H, TRT_INPUT_W,
            dtype=torch_in_dtype, device="cuda"
        )
        self.output_tensor = torch.zeros(
            1, 3, TRT_INPUT_H, TRT_INPUT_W,
            dtype=torch_out_dtype, device="cuda"
        )

        # CUDA stream for async execution
        self.stream = torch.cuda.Stream()

        engine_mb = os.path.getsize(str(engine_path)) / 1024**2
        vram_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"  TRT engine loaded ({engine_mb:.1f} MB on disk), VRAM: {vram_mb:.0f} MB")

    def __call__(self, x):
        """Run inference. Input: (1, 3, H, W) fp16 CUDA tensor (H <= TRT_INPUT_H, W <= TRT_INPUT_W).

        The input is expected to be pre-padded to (1, 3, TRT_INPUT_H, TRT_INPUT_W).
        Returns output of the same padded shape; caller is responsible for cropping.
        """
        assert x.shape[0] == 1, f"TRT engine only supports batch_size=1, got {x.shape[0]}"
        assert x.shape[2] == TRT_INPUT_H and x.shape[3] == TRT_INPUT_W, (
            f"Input must be ({TRT_INPUT_H}, {TRT_INPUT_W}), got ({x.shape[2]}, {x.shape[3]}). "
            f"Pre-pad to the fixed TRT input shape."
        )

        # Copy input data, casting to engine's expected dtype
        self.input_tensor.copy_(x.to(dtype=self.input_tensor.dtype).contiguous())

        with torch.cuda.stream(self.stream):
            # Set tensor addresses (TensorRT 10.x API)
            self.context.set_tensor_address("input", self.input_tensor.data_ptr())
            self.context.set_tensor_address("output", self.output_tensor.data_ptr())
            # Execute
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

        self.stream.synchronize()
        return self.output_tensor.clone()

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        # No-op: TRT model is already on GPU
        return self

    def parameters(self):
        # Empty iterator — no PyTorch parameters
        return iter([])

    def __del__(self):
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine


def prepare_tensorrt(checkpoint_path, num_calib_frames=200):
    """Full TensorRT preparation: ONNX export -> engine build -> load.

    Returns a TRTModel instance ready for inference.
    """
    # Check if tensorrt is available
    try:
        import tensorrt as trt
        print(f"  TensorRT version: {trt.__version__}")
    except ImportError:
        print("\n" + "=" * 60)
        print("ERROR: TensorRT is not installed!")
        print()
        print("Install with:")
        print("  pip install tensorrt-cu12")
        print()
        print("This is compatible with our CUDA 12.1 environment.")
        print("Also install ONNX for model export:")
        print("  pip install onnx")
        print("=" * 60)
        sys.exit(1)

    # Step 1: Export ONNX
    onnx_path = export_onnx(checkpoint_path)

    # Step 2: Build TRT engine (or load from cache)
    engine_path = build_tensorrt_engine(onnx_path, num_calib_frames=num_calib_frames)

    # Step 3: Load engine for inference
    model = TRTModel(engine_path)
    return model


def apply_quantization(model, quantize_mode="int8"):
    """Apply INT8 quantization. Try TorchAO first, fall back to PyTorch native."""
    if quantize_mode == "none":
        print("  Quantization: none (fp16 weights)")
        return model

    vram_before = torch.cuda.memory_allocated() / 1024**2

    if quantize_mode == "int8":
        # Try TorchAO first — API varies by version
        try:
            from torchao.quantization import quantize_
            # Try newer API first (torchao >= 0.8)
            try:
                from torchao.quantization import Int8WeightOnlyConfig
                quantize_(model, Int8WeightOnlyConfig())
            except ImportError:
                # Older API (torchao 0.5-0.7)
                from torchao.quantization import int8_weight_only
                quantize_(model, int8_weight_only())
            vram_after = torch.cuda.memory_allocated() / 1024**2
            print(f"  Quantization: TorchAO INT8 weight-only")
            print(f"  VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB (saved {vram_before - vram_after:.0f}MB)")
            return model
        except ImportError:
            print("  TorchAO not installed, falling back to PyTorch native quantization")
        except Exception as e:
            print(f"  TorchAO quantization failed ({e}), falling back to PyTorch native")

        # Fallback: PyTorch native dynamic quantization (CPU only)
        try:
            model_cpu = model.float().cpu()
            model_q = torch.quantization.quantize_dynamic(
                model_cpu, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
            )
            vram_after = torch.cuda.memory_allocated() / 1024**2
            print(f"  Quantization: PyTorch native dynamic INT8 (CPU-only fallback)")
            print(f"  WARNING: Native dynamic quantization runs on CPU, will be very slow")
            # Tag model so callers know it's on CPU
            model_q._on_cpu = True
            return model_q
        except Exception as e:
            print(f"  PyTorch native quantization also failed: {e}")
            print(f"  Continuing with fp16 weights")
            return model

    print(f"  Unknown quantize mode '{quantize_mode}', using fp16")
    return model


def prepare_model(checkpoint_path, quantize="none", use_compile=False, fp16=True,
                   num_calib_frames=200):
    """Load, quantize, compile, and warmup the model.

    For quantize="tensorrt", returns a TRTModel instead of a PyTorch model.
    The TRTModel has the same __call__ interface but ignores compile/channels_last.
    """
    # TensorRT path — completely separate from PyTorch model
    if quantize == "tensorrt":
        model = prepare_tensorrt(checkpoint_path, num_calib_frames=num_calib_frames)
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak VRAM after setup: {peak_vram:.0f}MB")
        return model

    model = load_nafnet(checkpoint_path, device=DEVICE, fp16=fp16)

    # Apply quantization before compile (TorchAO recommends this order)
    model = apply_quantization(model, quantize)

    # channels_last for better cuDNN performance
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    # Swap LayerNorm for compile-friendly version
    if use_compile:
        model = swap_layernorm_for_compile(model)
        print("  Swapped LayerNorm2d -> LayerNorm2dCompile")

    # torch.compile — try reduce-overhead, fall back to default, then to eager
    if use_compile:
        compiled = False
        for mode in ["reduce-overhead", "default"]:
            try:
                model_compiled = torch.compile(model, mode=mode)
                # Test with a small dummy to catch Triton/backend errors early
                dummy = torch.randn(1, 3, 64, 64, dtype=torch.float16 if fp16 else torch.float32,
                                    device=DEVICE).to(memory_format=torch.channels_last)
                with torch.no_grad():
                    _ = model_compiled(dummy)
                del dummy
                torch.cuda.empty_cache()
                model = model_compiled
                print(f"  torch.compile enabled ({mode} mode)")
                compiled = True
                break
            except Exception as e:
                print(f"  torch.compile mode='{mode}' failed: {e}")
        if not compiled:
            print("  torch.compile unavailable (Triton not installed on Windows), using eager mode")

    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak VRAM after setup: {peak_vram:.0f}MB")
    return model


def run_eval(model, max_frames=-1, fp16=True):
    """Run model on validation pairs and report metrics.

    Handles both PyTorch models and TRTModel. For TRTModel, input is padded
    to the fixed TRT_INPUT_H x TRT_INPUT_W shape.
    """
    input_dir = VAL_PAIRS_DIR / "input"
    target_dir = VAL_PAIRS_DIR / "target"

    if not input_dir.exists():
        print(f"ERROR: Validation pairs not found at {VAL_PAIRS_DIR}")
        return

    from PIL import Image

    is_trt = isinstance(model, TRTModel)

    input_files = sorted(input_dir.glob("*.png"))
    if max_frames > 0:
        input_files = input_files[:max_frames]

    print(f"\nEval: {len(input_files)} frames from {VAL_PAIRS_DIR}")
    if is_trt:
        print(f"  Mode: TensorRT INT8 (fixed {TRT_INPUT_H}x{TRT_INPUT_W})")

    torch.cuda.reset_peak_memory_stats()
    total_psnr = 0.0
    total_l1 = 0.0
    total_time = 0.0
    count = 0

    dtype = torch.float16 if fp16 else torch.float32

    for img_path in input_files:
        target_path = target_dir / img_path.name
        if not target_path.exists():
            print(f"  Skipping {img_path.name} (no target)")
            continue

        # Load input
        inp_np = np.array(Image.open(str(img_path)).convert("RGB"))
        inp_t = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0
        inp_t = inp_t.to(DEVICE)

        _, _, h, w = inp_t.shape

        if is_trt:
            # TRT needs exact fixed shape — pad to TRT_INPUT_H x TRT_INPUT_W
            inp_padded = torch.zeros(1, 3, TRT_INPUT_H, TRT_INPUT_W, dtype=dtype, device=DEVICE)
            inp_padded[:, :, :h, :w] = inp_t
            inp_t = inp_padded
        else:
            inp_t = inp_t.to(memory_format=torch.channels_last)
            # Pad to multiple of 16
            h_pad = h + (16 - h % 16) % 16
            w_pad = w + (16 - w % 16) % 16
            if h_pad != h or w_pad != w:
                inp_padded = torch.zeros(1, 3, h_pad, w_pad, dtype=dtype, device=DEVICE)
                inp_padded = inp_padded.to(memory_format=torch.channels_last)
                inp_padded[:, :, :h, :w] = inp_t
                inp_t = inp_padded

        # Inference
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_t = model(inp_t)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Crop back to original size
        out_t = out_t[:, :, :h, :w].clamp(0, 1)
        total_time += t1 - t0

        # Load target and compute metrics
        tgt_np = np.array(Image.open(str(target_path)).convert("RGB"))
        tgt_t = torch.from_numpy(tgt_np).permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0
        tgt_t = tgt_t.to(DEVICE)

        # PSNR
        mse = torch.mean((out_t - tgt_t) ** 2).item()
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = 100.0
        total_psnr += psnr

        # L1
        l1 = torch.mean(torch.abs(out_t - tgt_t)).item()
        total_l1 += l1

        count += 1
        print(f"  [{count}/{len(input_files)}] {img_path.name}: PSNR={psnr:.2f}dB, L1={l1:.4f}, "
              f"time={t1 - t0:.3f}s")

        # Free GPU memory
        del inp_t, out_t, tgt_t

    if count == 0:
        print("No frames processed!")
        return

    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    avg_psnr = total_psnr / count
    avg_l1 = total_l1 / count
    avg_fps = count / total_time

    print(f"\n{'=' * 60}")
    print(f"EVAL RESULTS ({count} frames)")
    print(f"  Avg PSNR:   {avg_psnr:.2f} dB")
    print(f"  Avg L1:     {avg_l1:.4f}")
    print(f"  Avg FPS:    {avg_fps:.1f}")
    print(f"  Peak VRAM:  {peak_vram:.0f} MB")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

    return {
        "avg_psnr": avg_psnr,
        "avg_l1": avg_l1,
        "avg_fps": avg_fps,
        "peak_vram_mb": peak_vram,
        "frames": count,
    }


def denoise_video(
    input_path, output_path, model,
    batch_size=1, crf=18, encoder="libx265",
    max_frames=-1, fp16=True,
):
    """Stream video through NAFNet with threaded IO (local version)."""
    import av

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    w, h, fps, total_frames, duration = get_video_info(input_path)

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    print(f"\nInput: {input_path}")
    print(f"  {w}x{h} @ {fps:.3f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Output: {output_path}")

    # FFmpeg encoder setup — local only uses libx265 or libx264
    ffmpeg_bin = get_ffmpeg()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_bytes = w * h * 3
    fps_str = f"{fps:.6f}"

    if encoder == "libx264":
        write_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str, "-threads", "0", "-i", "pipe:0",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
            "-pix_fmt", "yuv420p10le",
            "-movflags", "+faststart", output_path,
        ]
    else:
        write_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps_str, "-threads", "0", "-i", "pipe:0",
            "-c:v", "libx265", "-crf", str(crf), "-preset", "fast",
            "-pix_fmt", "yuv420p10le",
            "-x265-params", "aq-mode=3:aq-strength=0.8:deblock=-1,-1:no-sao=1:rc-lookahead=20:pools=4",
            "-movflags", "+faststart", output_path,
        ]

    # Threaded IO
    frame_queue = Queue(maxsize=batch_size * 8)
    result_queue = Queue(maxsize=batch_size * 8)
    SENTINEL = object()
    WRITER_DONE = object()

    def reader_thread():
        container = av.open(input_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 0
        count = 0
        for frame in container.decode(stream):
            frame_queue.put(frame.to_ndarray(format='rgb24'))
            count += 1
            if max_frames > 0 and count >= max_frames:
                break
        frame_queue.put(SENTINEL)
        container.close()

    writer_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=frame_bytes * 8)

    def writer_thread():
        while True:
            item = result_queue.get()
            if item is WRITER_DONE:
                break
            writer_proc.stdin.write(item.tobytes())
        writer_proc.stdin.close()
        writer_proc.wait()

    reader_t = Thread(target=reader_thread, daemon=True)
    writer_t = Thread(target=writer_thread, daemon=True)
    reader_t.start()
    writer_t.start()

    is_trt = isinstance(model, TRTModel)
    if is_trt and batch_size > 1:
        print(f"  WARNING: TRT engine only supports batch_size=1, overriding (was {batch_size})")
        batch_size = 1

    print(f"\nProcessing: batch_size={batch_size}, {'TRT INT8' if is_trt else 'fp16' if fp16 else 'fp32'}")
    print(f"Encoding: {encoder} CRF {crf}")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    processed = 0
    dtype = torch.float16 if fp16 else torch.float32

    # Pad dimensions
    if is_trt:
        h_pad = TRT_INPUT_H
        w_pad = TRT_INPUT_W
    else:
        h_pad = h + (16 - h % 16) % 16
        w_pad = w + (16 - w % 16) % 16

    while True:
        # Collect batch
        frames = []
        done = False
        for _ in range(batch_size):
            frame = frame_queue.get()
            if frame is SENTINEL:
                done = True
                break
            frames.append(frame)

        if not frames:
            break

        # Prepare tensor
        batch_t = torch.zeros(len(frames), 3, h_pad, w_pad, dtype=dtype, device=DEVICE)
        if not is_trt:
            batch_t = batch_t.to(memory_format=torch.channels_last)
        for i, f in enumerate(frames):
            t = torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).to(dtype) / 255.0
            batch_t[i, :, :h, :w] = t

        # Inference
        with torch.no_grad():
            out_t = model(batch_t)

        # Crop and convert
        out_cropped = out_t[:, :, :h, :w]
        out_np = (out_cropped.clamp(0, 1) * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
        result_queue.put(out_np[:len(frames)])

        processed += len(frames)
        if processed % max(batch_size * 10, 50) == 0 or processed == len(frames):
            elapsed = time.time() - start
            fps_actual = processed / elapsed
            eta_min = (total_frames - processed) / max(fps_actual, 0.01) / 60
            vram = torch.cuda.memory_allocated() / 1024**2
            print(f"  [{processed}/{total_frames}] {fps_actual:.1f} fps, "
                  f"VRAM: {vram:.0f}MB, ETA: {eta_min:.1f}min")

        del batch_t, out_t
        if done:
            break

    result_queue.put(WRITER_DONE)
    writer_t.join()
    reader_t.join()

    elapsed = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    out_size = os.path.getsize(output_path) / 1024**2

    print(f"\n{'=' * 60}")
    print(f"DONE: {processed} frames in {elapsed / 60:.1f} min ({processed / elapsed:.1f} fps)")
    print(f"  Peak VRAM: {peak_vram:.0f} MB")
    print(f"  Output: {output_path} ({out_size:.1f} MB)")
    print(f"{'=' * 60}")

    return {
        "frames": processed,
        "elapsed_min": elapsed / 60,
        "fps": processed / elapsed,
        "peak_vram_mb": peak_vram,
        "output_size_mb": out_size,
    }


def main():
    parser = argparse.ArgumentParser(description="NAFNet local inference with INT8 quantization")
    parser.add_argument("--input", default=None, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--checkpoint", required=True, help="NAFNet checkpoint path")
    parser.add_argument("--quantize", default="none", choices=["none", "int8", "tensorrt"],
                        help="Quantization mode: none=fp16, int8=TorchAO, tensorrt=TRT INT8")
    parser.add_argument("--batch-size", type=int, default=1, help="Frames per batch (default: 1)")
    parser.add_argument("--crf", type=int, default=18, help="CRF for encoding")
    parser.add_argument("--encoder", default="libx265", choices=["libx265", "libx264"],
                        help="Video encoder (local only, no NVENC)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to process")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--calib-frames", type=int, default=200,
                        help="Number of calibration frames for TensorRT INT8 (default: 200)")
    parser.add_argument("--eval", action="store_true", help="Run on val_pairs and report metrics")
    args = parser.parse_args()

    if not args.eval and not args.input:
        parser.error("Either --eval or --input is required")

    fp16 = True
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"Quantize: {args.quantize}, Compile: {args.compile}, fp16: {fp16}")
    print()

    model = prepare_model(
        args.checkpoint,
        quantize=args.quantize,
        use_compile=args.compile,
        fp16=fp16,
        num_calib_frames=args.calib_frames,
    )

    if args.eval:
        run_eval(model, max_frames=args.max_frames, fp16=fp16)
    else:
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_local.mkv"

        denoise_video(
            input_path=args.input,
            output_path=args.output,
            model=model,
            batch_size=args.batch_size,
            crf=args.crf,
            encoder=args.encoder,
            max_frames=args.max_frames,
            fp16=fp16,
        )


if __name__ == "__main__":
    main()
