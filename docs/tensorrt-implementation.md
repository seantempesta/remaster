# TensorRT INT8 Implementation Plan for NAFNet

Implementation guide for accelerating NAFNet-width64 denoiser inference on RTX 3060 (6GB VRAM, Ampere SM 8.6, Windows) using TensorRT INT8 quantization.

**Current baseline:** fp16 eager PyTorch on RTX 3060 = 0.59 fps at 1080p, ~3.3 GB VRAM.

---

## 1. Recommended Path: ONNX → TensorRT (Native Python API)

After evaluating three options, the **ONNX → TensorRT native Python API** path is the clear winner for our setup:

| Path | Windows Support | Ease of Use | Performance | Verdict |
|------|----------------|-------------|-------------|---------|
| **ONNX → TensorRT** (trtexec / Python API) | Full pip support via `tensorrt-cu12` | Medium | Best (pure TRT engine) | **Use this** |
| **Torch-TensorRT** | No pre-built Windows wheels; must build from source with Bazel | Hard | Good (but overhead vs pure TRT) | Skip on Windows |
| **torch2trt** (NVIDIA-AI-IOT) | Community project, works on Windows | Easy | Good | Backup option |

### Why not Torch-TensorRT?

Torch-TensorRT (`torch-tensorrt` on PyPI) does not ship pre-built Windows wheels. Installation requires cloning the repo and building from source using Bazel with Windows-specific build configuration. This is fragile and not worth the effort when the native TensorRT Python API works well on Windows via pip.

### Why not torch2trt?

torch2trt is a community project from NVIDIA-AI-IOT. It works but is less actively maintained than the official TensorRT SDK and may lag behind on TensorRT 10.x API changes. It is a reasonable backup if the ONNX path hits issues.

---

## 2. Installation & Version Compatibility

### Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| TensorRT | 10.16.0 (latest, March 2026) | `pip install tensorrt-cu12` |
| CUDA (local) | 12.1 | Compatible with tensorrt-cu12 |
| Python | 3.10 | Windows wheels available (cp310-win_amd64) |
| PyTorch | 2.5.1+cu121 | For ONNX export only; not used at inference |
| ONNX | 1.15+ | `pip install onnx` |
| ONNX Runtime (optional) | 1.17+ | For validating ONNX model before TRT conversion |
| RTX 3060 | SM 8.6 (Ampere) | Supports INT8 tensor cores (3rd gen) |

### Installation Commands

```bash
# In the upscale conda env
pip install tensorrt-cu12
pip install onnx onnxruntime-gpu

# Verify TensorRT
python -c "import tensorrt as trt; print(trt.__version__)"

# trtexec ships with the tensorrt pip package
# Find it at: <env>/Lib/site-packages/tensorrt_cu12/bin/trtexec.exe
```

### CUDA 12.1 Compatibility Note

TensorRT built with CUDA 12.x is compatible across CUDA 12.x minor versions. Our CUDA 12.1 will work with `tensorrt-cu12`. The RTX 3060 (SM 8.6) is fully supported by all TensorRT versions >= 7.2.1.

---

## 3. Step 1: ONNX Export

### Export Script

```python
import torch
from lib.nafnet_arch import NAFNet, swap_layernorm_for_export

# Build model
model = NAFNet(
    img_channel=3, width=64,
    middle_blk_num=12,
    enc_blk_nums=[2, 2, 4, 8],
    dec_blk_nums=[2, 2, 2, 2],
)

# Load checkpoint
ckpt = torch.load("checkpoints/nafnet_distill/safe/nafnet_best.pth",
                   map_location="cpu", weights_only=True)
state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt))
model.load_state_dict(state_dict, strict=True)
model.eval()

# CRITICAL: swap custom LayerNorm2d for export-safe version
model = swap_layernorm_for_export(model)
model.half().cuda()

# Fixed 1080p input (padded to multiple of 16 = 1088x1920)
dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)

torch.onnx.export(
    model, dummy,
    "nafnet_w64_fp16.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,               # opset 17 for native LayerNormalization support
    do_constant_folding=True,
    dynamic_axes=None,              # FIXED shape for best TRT optimization
)
```

### Key Export Decisions

**Opset 17:** Required for native ONNX `LayerNormalization` op. Our `LayerNorm2dExport` computes mean/var/normalize manually so it works at lower opsets too, but opset 17 is recommended by NVIDIA for best TRT layer fusion of normalization ops.

**Fixed vs Dynamic shapes:** Use **fixed shape** `(1, 3, 1088, 1920)`. TensorRT generates significantly better kernels for fixed shapes — no dynamic shape overhead, better memory planning, more aggressive kernel fusion. Since we always process 1080p video (padded to 1088), there's no reason for dynamic shapes. If we later want batch 2, we build a separate engine.

**PixelShuffle:** ONNX supports `DepthToSpace` (the ONNX equivalent of PixelShuffle) since opset 1. PyTorch exports `nn.PixelShuffle` as a reshape+transpose sequence that TensorRT handles natively. No issues expected. The official PyTorch ONNX super-resolution tutorial uses PixelShuffle as a key example.

**LayerNorm2dExport:** Our `swap_layernorm_for_export()` replaces the custom `LayerNormFunction` (autograd.Function) with standard ops: mean → subtract → pow → mean → sqrt → divide → scale → bias. This is critical because custom autograd Functions cannot be traced for ONNX export. The export version is numerically identical to the fp16 path of the original.

### Potential Export Gotchas

1. **`check_image_size` dynamic padding:** The `NAFNet.forward()` method calls `self.check_image_size(inp)` which uses `F.pad` conditionally. With fixed 1088x1920 input (already a multiple of 16), the padding is zero and should be constant-folded away. Verify with `onnxruntime` after export.

2. **Output slicing:** `x[:, :, :H, :W]` at the end of forward — with fixed input, H=1088 and the model pads internally. The output slice `[:, :, :1080, :1920]` will be baked as a constant Slice op. This is fine.

3. **SimpleGate's `chunk`:** `x.chunk(2, dim=1)` exports as a `Split` op in ONNX, which TensorRT supports natively.

---

## 4. Step 2: TensorRT Engine Build with INT8 Calibration

### Option A: Quick Test with trtexec (FP16 only first)

```bash
# Find trtexec in your pip install
# Usually at: C:\Users\sean\miniconda3\envs\upscale\Lib\site-packages\tensorrt_cu12\bin\trtexec.exe

trtexec.exe --onnx=nafnet_w64_fp16.onnx ^
    --fp16 ^
    --saveEngine=nafnet_w64_fp16.engine ^
    --workspace=4096 ^
    --verbose
```

This builds an FP16 TRT engine without INT8 calibration. Use this to verify the export works before adding INT8.

### Option B: INT8 Engine with Python API Calibration

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class NAFNetCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator using our validation/training frames."""

    def __init__(self, frame_dir, num_frames=200, batch_size=1,
                 cache_file="nafnet_int8_calib.cache"):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Load calibration frames (1088x1920 padded)
        from pathlib import Path
        from PIL import Image

        frame_paths = sorted(Path(frame_dir).glob("*.png"))[:num_frames]
        self.frames = []
        for p in frame_paths:
            img = np.array(Image.open(str(p)).convert("RGB"))
            h, w = img.shape[:2]
            # Pad to 1088x1920
            padded = np.zeros((1088, 1920, 3), dtype=np.uint8)
            padded[:h, :w, :] = img
            t = padded.transpose(2, 0, 1).astype(np.float16) / 255.0
            self.frames.append(t)

        self.num_frames = len(self.frames)
        self.device_input = cuda.mem_alloc(
            batch_size * 3 * 1088 * 1920 * 2  # fp16 = 2 bytes
        )
        print(f"Calibrator: {self.num_frames} frames loaded")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_frames:
            return None

        batch = self.frames[self.current_index]
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.current_index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """Read cached calibration data to skip re-calibration."""
        import os
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration cache for reuse."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Calibration cache written to {self.cache_file}")


def build_int8_engine(onnx_path, engine_path, calib_frame_dir):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(i)}")
            return None

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    # Set up INT8 calibrator
    calibrator = NAFNetCalibrator(calib_frame_dir, num_frames=200)
    config.int8_calibrator = calibrator

    # Build engine
    print("Building INT8 engine (this takes several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Engine saved to {engine_path}")
    return serialized_engine
```

### Calibration Strategy

**How many frames?** NVIDIA recommends 500-1000 calibration samples for best results. Minimum is ~100. We have:
- 33 val frames (val_pairs/input/)
- 1224 train frames

**Recommendation:** Use all 33 val frames + a random subset of ~170-470 training frames = **200-500 total**. The frames should cover diverse scene content (dark scenes, bright scenes, fast motion, static shots). For video denoising, temporal diversity matters more than quantity — 200 frames from different scenes is better than 500 frames from one scene.

**Calibration cache:** TensorRT writes a calibration cache file (a few KB) that stores per-layer scale factors. This cache can be reused to skip recalibration when rebuilding the engine (e.g., after TRT version update). The cache is model-specific and data-specific — regenerate if the model weights change.

**IInt8EntropyCalibrator2 vs IInt8MinMaxCalibrator:** Use **EntropyCalibrator2** (the default). Entropy calibration finds optimal thresholds by minimizing KL divergence between the original and quantized activation distributions. MinMax simply uses the absolute max value, which is sensitive to outliers. For image restoration where activation distributions can have long tails (bright highlights, dark shadows), entropy calibration preserves more precision in the important mid-range.

---

## 5. Step 3: TensorRT Inference

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInferencer:
    def __init__(self, engine_path):
        # Load engine
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device memory for input/output
        # Input: (1, 3, 1088, 1920) fp16
        # Output: (1, 3, 1088, 1920) fp16
        input_size = 1 * 3 * 1088 * 1920 * 2  # fp16
        output_size = 1 * 3 * 1088 * 1920 * 2

        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.stream = cuda.Stream()

        # Pre-allocate host output buffer
        self.h_output = np.empty((1, 3, 1088, 1920), dtype=np.float16)

    def infer(self, input_np):
        """Run inference on a single frame.

        Args:
            input_np: numpy array (1, 3, 1088, 1920) float16

        Returns:
            output_np: numpy array (1, 3, 1088, 1920) float16
        """
        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, input_np, self.stream)

        # Set tensor addresses (TRT 10.x API)
        self.context.set_tensor_address("input", int(self.d_input))
        self.context.set_tensor_address("output", int(self.d_output))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output

    def __del__(self):
        del self.context
        del self.engine
```

### VRAM Usage Estimate

| Component | Size |
|-----------|------|
| TRT engine weights (INT8) | ~120 MB |
| Input buffer (1, 3, 1088, 1920) fp16 | ~12 MB |
| Output buffer (1, 3, 1088, 1920) fp16 | ~12 MB |
| TRT activation workspace | ~200-400 MB |
| CUDA/driver overhead | ~300-500 MB |
| **Total (estimated)** | **~650 MB - 1.1 GB** |

This is dramatically less than PyTorch fp16 eager (3.3 GB) because TensorRT:
- Uses INT8 weights (half the size of fp16)
- Pre-plans memory allocation (no dynamic allocation overhead)
- Fuses layers to reduce intermediate activation storage
- Does not carry PyTorch framework overhead

With ~1 GB VRAM for the engine, we could potentially run **batch 2 or even batch 4** within the 6 GB budget.

---

## 6. Op Compatibility Analysis

All NAFNet ops are well-supported in TensorRT:

| NAFNet Op | ONNX Op | TensorRT Support | INT8 Support | Notes |
|-----------|---------|-------------------|--------------|-------|
| `nn.Conv2d` | Conv | Native | Full INT8 | Core op, excellent INT8 tensor core utilization |
| `LayerNorm2dExport` | Reducemean, Sub, Pow, Sqrt, Div, Mul, Add | Native (fused) | Keep FP16 | With opset 17, TRT can fuse into LayerNorm kernel |
| `SimpleGate` (chunk+mul) | Split + Mul | Native | Full INT8 | Split is a zero-cost reshape; Mul has INT8 path |
| `nn.AdaptiveAvgPool2d(1)` | GlobalAveragePool | Native | Full INT8 | Output (1,1) is a special case TRT optimizes well |
| `nn.PixelShuffle(2)` | Reshape + Transpose (DepthToSpace) | Native | Full INT8 | Exports as reshape+transpose, TRT fuses to shuffle |
| `F.pad` (check_image_size) | Pad | Native | Full INT8 | With fixed input, this is constant-folded away |
| Residual add (`inp + x * beta`) | Add, Mul | Native | Full INT8 | Element-wise ops |
| `x[:, :, :H, :W]` (output crop) | Slice | Native | N/A | Data movement only |

### Sensitive Layers — Keep in FP16

TensorRT's mixed-precision engine will automatically keep certain layers in higher precision when INT8 would cause too much error. However, we can explicitly mark sensitive layers:

1. **`intro` conv (3 → 64 channels):** Only 3 input channels — limited quantization bins. Keep FP16.
2. **`ending` conv (64 → 3 channels):** Directly produces output pixels. Keep FP16.
3. **LayerNorm2d ops:** Already computed in FP32 internally in our export version. TRT should handle this.
4. **SCA (channel attention) convs:** Small 1x1 convs after global avg pool. Low compute cost, keep FP16.

The `--fp16` flag alongside `--int8` in trtexec enables this mixed-precision mode automatically — TRT chooses the best precision per-layer based on calibration data quality.

---

## 7. Expected Performance

### Speedup Estimates

| Configuration | Expected FPS | Speedup vs Baseline | Source |
|---------------|-------------|---------------------|--------|
| PyTorch fp16 eager (current) | 0.59 | 1.0x | Measured |
| TRT FP16 engine | 1.0 - 1.5 | 1.7-2.5x | TRT kernel fusion + memory optimization |
| TRT INT8 engine | 1.5 - 2.5 | 2.5-4.2x | INT8 tensor cores (101.9 TOPS vs 12.7 TFLOPS fp16) |
| TRT INT8 batch 2 | 2.5 - 4.0 | 4.2-6.8x | Better GPU utilization with batching |

### Rationale

- **FP16 TRT vs PyTorch eager:** TensorRT's layer fusion, memory planning, and optimized kernel selection typically give 1.5-2.5x over eager PyTorch for CNN models. This is well-documented in NVIDIA benchmarks (ResNet-50: ~1.5x, larger models: up to 3x).

- **INT8 vs FP16 TRT:** NVIDIA's own benchmarks show ~60% speedup going from FP16 to INT8 on ResNet-50 (507 to 812 qps). For our larger model at 1080p, the speedup may be closer to 40-60% because we're more memory-bandwidth-bound than compute-bound at this resolution.

- **RTX 3060 INT8 theoretical:** 101.9 INT8 TOPS vs 12.7 FP16 TFLOPS = 8x theoretical peak advantage. Real-world is 1.5-2x due to memory bandwidth limits, but still significant.

- **Batch 2:** With INT8 engine using ~1 GB VRAM, batch 2 adds only ~12 MB for I/O buffers plus some activation overhead. GPU utilization improves significantly with batch 2 on the 3584-core RTX 3060.

### Quality Expectations

| Metric | Expected Impact | Notes |
|--------|----------------|-------|
| PSNR drop (INT8 PTQ) | 0.1-0.3 dB | CNN denoisers are robust to INT8 |
| PSNR drop (mixed FP16/INT8) | < 0.1 dB | Keep intro/ending/LN in FP16 |
| Visual difference | Imperceptible | For denoising, 0.3 dB is invisible |

---

## 8. Alternative: torch.compile on Windows (Triton)

### Current Status (April 2026)

**Triton now works natively on Windows.** As of PyTorch 2.7 (released 2025) and Triton 3.3, Windows is officially supported. The separate `triton-windows` fork was archived in February 2026 because support was merged into mainline Triton.

### Requirements

- PyTorch >= 2.7
- Triton >= 3.3
- CUDA >= 12.8
- TinyCC is bundled in Triton wheels (no separate C compiler needed)

### Problem: CUDA Version

Our local environment uses **CUDA 12.1**, but Triton on Windows requires **CUDA >= 12.8**. Upgrading CUDA locally would require:
1. Installing CUDA 12.8+ toolkit
2. Reinstalling PyTorch with cu128: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
3. Verifying all other deps still work

This is doable but adds complexity and risk to the local environment.

### Expected Performance with torch.compile

On the cloud (H100), torch.compile gave us 27.9 fps (vs ~15 fps eager). That's roughly a 1.8x speedup. On RTX 3060:

| Configuration | Expected FPS | Notes |
|---------------|-------------|-------|
| PyTorch fp16 eager | 0.59 | Current baseline |
| torch.compile fp16 | 0.9 - 1.1 | ~1.5-1.8x (Inductor kernel fusion) |
| torch.compile + INT8 (TorchAO) | 1.0 - 1.3 | TorchAO INT8 weight-only + compile |

### torch.compile vs TensorRT Verdict

| Factor | torch.compile | TensorRT INT8 |
|--------|---------------|---------------|
| Expected speedup | 1.5-1.8x | 2.5-4.2x |
| Setup difficulty | Medium (CUDA upgrade) | Medium (ONNX export + calibration) |
| Code changes | Minimal (`torch.compile(model)`) | Separate inference wrapper |
| Quality impact | None (same numerics) | < 0.3 dB PSNR |
| Maintenance | Easy (stays in PyTorch ecosystem) | Engine must be rebuilt per TRT version |
| Batch flexibility | Dynamic | Fixed per engine |

**Recommendation:** TensorRT INT8 offers substantially better performance (2-4x more than torch.compile). However, torch.compile is a reasonable quick win if you upgrade to CUDA 12.8+. They are not mutually exclusive — you could use torch.compile as an intermediate step.

### WSL2 Alternative

If upgrading CUDA on native Windows is undesirable, WSL2 is a viable path:
- WSL2 uses the Windows GPU driver directly (no separate driver install needed in Ubuntu)
- PyTorch + CUDA + Triton work well on WSL2 Ubuntu
- torch.compile works fully on Linux/WSL2
- TensorRT also works on WSL2 (Linux wheels)

However, WSL2 adds filesystem performance overhead for video I/O (reading/writing large files across the Windows/Linux boundary is slow). Video files should be stored on a Linux-native filesystem within WSL2, not accessed via `/mnt/c/` or `/mnt/e/`.

---

## 9. Step-by-Step Implementation Plan

### Phase 1: ONNX Export & Validation (30 min)

1. Install deps: `pip install onnx onnxruntime-gpu`
2. Write export script using the code in Section 3
3. Export NAFNet to ONNX with `swap_layernorm_for_export()`
4. Validate ONNX model: run a few frames through ONNX Runtime and compare outputs to PyTorch (should be numerically identical for fp16)
5. Inspect ONNX graph with Netron (`pip install netron`) to verify all ops look correct

### Phase 2: TensorRT FP16 Engine (1 hour)

1. Install TensorRT: `pip install tensorrt-cu12`
2. Build FP16 engine with trtexec (no INT8 yet)
3. Write TRT inference wrapper (Section 5)
4. Benchmark FP16 TRT vs PyTorch eager — expect 1.5-2.5x speedup
5. Verify output quality matches PyTorch (should be identical for FP16)

### Phase 3: INT8 Calibration & Engine (1-2 hours)

1. Extract calibration frames: 200 frames from val_pairs + training data
2. Implement `NAFNetCalibrator` class (Section 4)
3. Build INT8+FP16 mixed-precision engine
4. Benchmark INT8 TRT — expect additional 40-60% over FP16 TRT
5. Compare PSNR on val_pairs: fp16 PyTorch vs INT8 TRT (target < 0.3 dB drop)
6. If PSNR drop > 0.3 dB, try keeping intro/ending convs in FP16 by adjusting layer precisions

### Phase 4: Integration into Pipeline (1-2 hours)

1. Create `pipelines/denoise_trt.py` — TensorRT inference pipeline
2. Reuse the threaded I/O pattern from `denoise_local.py` (PyAV decode + ffmpeg encode)
3. Add batch 2 support if VRAM allows
4. End-to-end test on a short clip

### Phase 5: Optimization (optional, 1-2 hours)

1. Try batch 2 and batch 4 engines — build separate engines for each batch size
2. Profile with `trtexec --dumpProfile` to find bottleneck layers
3. Experiment with `--sparsity=enable` if model has sparse weights
4. Try CUDA graphs wrapping the TRT execution for reduced launch overhead

---

## 10. Potential Issues & Mitigations

### Issue 1: pycuda dependency

TensorRT's Python samples use `pycuda` for device memory management. On Windows, pycuda can be tricky to install.

**Mitigation:** Use `pip install pycuda`. If that fails, try the unofficial wheel from Christoph Gohlke's site or use `cupy` instead (`pip install cupy-cuda12x`). Alternatively, TensorRT 10.x supports using raw `ctypes` for memory management, or you can use `torch.cuda` tensors as device memory pointers:

```python
# Using PyTorch tensors instead of pycuda
input_tensor = torch.zeros(1, 3, 1088, 1920, dtype=torch.float16, device="cuda")
output_tensor = torch.zeros(1, 3, 1088, 1920, dtype=torch.float16, device="cuda")
context.set_tensor_address("input", input_tensor.data_ptr())
context.set_tensor_address("output", output_tensor.data_ptr())
```

This avoids the pycuda dependency entirely.

### Issue 2: Engine build time

Building a TensorRT engine for a 116M parameter model at 1080p can take 5-30 minutes depending on the GPU and optimization level.

**Mitigation:** The engine is serialized to disk and loaded instantly on subsequent runs. The calibration cache also persists, so recalibration is only needed when the model weights change.

### Issue 3: Engine not portable

TensorRT engines are specific to the exact GPU model, TensorRT version, and CUDA version. An engine built on RTX 3060 won't work on RTX 4060.

**Mitigation:** Keep the ONNX model and calibration cache. Rebuild the engine when the environment changes. This takes minutes, not hours.

### Issue 4: check_image_size dynamic behavior

NAFNet's `check_image_size` method uses `F.pad` with amounts computed from the input shape. With ONNX tracing, this should be constant-folded for our fixed input size.

**Mitigation:** Pre-pad inputs to 1088x1920 before feeding to the model, and handle the output crop (to 1080x1920) outside the model. This makes the model's internal padding a no-op.

### Issue 5: LayerNorm2dExport hardcodes `.half()` return

The `LayerNorm2dExport.forward()` ends with `return x_f.half()`. In an INT8 engine, this cast may be unnecessary or suboptimal.

**Mitigation:** For ONNX export, this cast will appear as a `Cast` op. TensorRT will optimize it away or keep it as appropriate for the precision it chooses for surrounding layers. No action needed.

---

## 11. TensorRT for RTX (Alternative)

NVIDIA released **TensorRT for RTX** — a lightweight (~200 MB) inference library optimized specifically for desktop RTX GPUs on Windows 11. It claims >50% improvement over DirectML and includes a JIT optimizer that builds engines in 15-30 seconds.

### Differences from Standard TensorRT

| Feature | TensorRT (Standard) | TensorRT for RTX |
|---------|---------------------|------------------|
| Target | Data center + desktop | Desktop RTX only |
| Size | ~1-2 GB | ~200 MB |
| Engine build | Ahead-of-time | JIT (AOT + JIT hybrid) |
| API | C++/Python, ONNX parser | C++ only, ONNX format |
| INT8 | Full calibration API | Supported |
| Python API | Yes | Not currently |

### Verdict

TensorRT for RTX lacks a Python API as of early 2026, making it impractical for our Python-based pipeline. Standard TensorRT via `pip install tensorrt-cu12` is the right choice. TensorRT for RTX may become relevant if/when they add Python bindings.

---

## 12. Summary & Decision

**Primary path:** ONNX → TensorRT INT8 via native Python API

- Install: `pip install tensorrt-cu12 onnx`
- Export: `swap_layernorm_for_export()` → `torch.onnx.export()` with opset 17, fixed shape
- Calibrate: `IInt8EntropyCalibrator2` with 200-500 frames from val_pairs + training data
- Build: Mixed FP16/INT8 engine (intro/ending convs stay FP16)
- Run: TRT Python runtime with PyTorch tensor memory (no pycuda needed)
- Expected result: **1.5-2.5 fps** (vs 0.59 fps current), **< 1.5 GB VRAM** (vs 3.3 GB current)

**Fallback:** If TensorRT hits unexpected issues, upgrade to CUDA 12.8 and use torch.compile with TorchAO INT8 weight-only quantization for a simpler 1.5-1.8x speedup.

---

## Sources

### TensorRT Installation & Compatibility
- [TensorRT PyPI (tensorrt-cu12)](https://pypi.org/project/tensorrt-cu12/) — pip install, version history
- [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) — official install docs
- [TensorRT Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) — GPU/CUDA/OS compatibility
- [TensorRT Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html) — ONNX workflow

### Torch-TensorRT (Windows)
- [Torch-TensorRT Installation](https://docs.pytorch.org/TensorRT/getting_started/installation.html) — no pre-built Windows wheels
- [Building Torch-TensorRT on Windows](https://docs.pytorch.org/TensorRT/getting_started/getting_started_with_windows.html) — requires Bazel build
- [torch-tensorrt PyPI](https://pypi.org/project/torch-tensorrt/) — Linux-only wheels
- [Torch-TensorRT PTQ Guide](https://docs.pytorch.org/TensorRT/user_guide/ptq.html) — INT8 calibration via Torch-TRT

### INT8 Calibration
- [IInt8EntropyCalibrator2 API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Int8/EntropyCalibrator.html)
- [Calibration Methods Comparison (Medium)](https://medium.com/@yangwq177/which-quantization-calibration-methods-does-nvidia-tensorrt-support-e5085dfc9021) — Entropy vs MinMax vs Percentile
- [TensorRT INT8 Calibration Examples (GitHub)](https://github.com/rmccorm4/tensorrt-utils/blob/master/int8/calibration/ImagenetCalibrator.py)
- [NVIDIA TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)

### ONNX Export
- [PyTorch ONNX Export — Super Resolution Tutorial](https://docs.pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) — PixelShuffle export example
- [PyTorch ONNX Documentation](https://docs.pytorch.org/docs/stable/onnx.html)
- [PixelShuffle ONNX Issue #51181](https://github.com/pytorch/pytorch/issues/51181) — pixel_unshuffle support
- [LayerNorm ONNX Opset 17 Issue #126160](https://github.com/pytorch/pytorch/issues/126160)

### TensorRT ONNX Samples
- [NVIDIA TensorRT ONNX ResNet50 Sample](https://github.com/NVIDIA/TensorRT/blob/main/samples/python/introductory_parser_samples/onnx_resnet50.py)
- [Windows TensorRT Python Guide (GitHub)](https://github.com/chansoopark98/Windows-TensorRT-Python)

### Performance References
- [torch.compile vs TensorRT (Collabora)](https://www.collabora.com/news-and-blog/blog/2024/12/19/faster-inference-torch.compile-vs-tensorrt/)
- [TensorRT for RTX Blog](https://developer.nvidia.com/blog/nvidia-tensorrt-for-rtx-introduces-an-optimized-inference-ai-library-on-windows/)
- [NVIDIA TensorRT Speed Up Inference Blog](https://developer.nvidia.com/blog/speed-up-inference-tensorrt/)

### Triton / torch.compile on Windows
- [Triton Windows (official, merged into mainline)](https://github.com/triton-lang/triton-windows) — archived Feb 2026
- [torch.compile Windows Support Timeline (PyTorch Forums)](https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268)
- [PyTorch 2.7 Release — Triton 3.3 + Windows](https://pytorch.org/blog/pytorch-2-7/)

### TensorRT Memory Management
- [TensorRT Resource Management (Torch-TRT)](https://docs.pytorch.org/TensorRT/tutorials/resource_memory/resource_management.html)
- [TensorRT Memory Discussion (GitHub)](https://github.com/NVIDIA/TensorRT/issues/2297)
