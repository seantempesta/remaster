# Speed Optimization Research Log

Agents: append your findings here so future agents don't repeat work or make the same mistakes.

---

## 2026-04-01: torch.compile results

- **reduce-overhead** mode (CUDA graphs): 3.3x speedup on L4 (0.78 → 2.56 fps), 12.4 fps on A100
- **max-autotune** mode (Triton autotuning): only 3% better than reduce-overhead (12.78 vs 12.43 fps on A100), but takes 5+ min warmup. Not worth it.
- **channels_last**: hurts without compile (0.67 vs 0.78 fps), helps ~15% with compile (2.56 vs 2.21 fps). The custom LayerNorm2d fp16→fp32 casting likely causes extra format conversions without compile fusion.
- **cudnn.benchmark**: negligible effect either way
- **Batching**: bs=8 gives ~8% over bs=1 on A100 (13.4 vs 12.4 fps steady-state). Model is memory-bandwidth-bound.
- **Compile cache**: `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` env vars point to Modal volume to persist compiled kernels across runs. Eliminates warmup on subsequent runs.

## 2026-04-01: torch.compile cache persistence

- Set `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a path on the Modal volume before calling `torch.compile()`
- Cache is automatic — no code changes beyond the env vars
- AOTInductor (`aoti_compile_and_package`) gives zero-warmup export but requires PyTorch >= 2.6 (we're on 2.5.1)

## 2026-04-01: TensorRT investigation (in progress)

- **LayerNorm2d issue**: Original `LayerNorm2d` uses `torch.autograd.Function` with manual fp32 casting — not compatible with ONNX/TRT export. Created `LayerNorm2dExport` using standard ops only.
- **swap_layernorm_for_export()**: Replaces all `LayerNorm2d` modules in the model with `LayerNorm2dExport`, preserving weights. Must be called before TRT compilation.
- **Modal image issue**: `add_local_dir` must come LAST in Modal image builds. `trt_image` needed restructuring to put `pip_install("torch-tensorrt")` before `add_local_dir`.
- **torch-tensorrt==2.5.0**: Matches our PyTorch 2.5.1. Prints warning about missing `modelopt` library for quantized models — this is harmless for fp16 inference.
- **RESOLVED**: `torch.compile(backend="torch_tensorrt")` is preferred over `torch_tensorrt.compile()` — see engine caching section below.

## 2026-04-01: TensorRT engine caching & serialization

Sources: [Saving models](https://docs.pytorch.org/TensorRT/user_guide/saving_models.html), [Engine caching](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/engine_caching_example.html), [torch.compile backend](https://docs.pytorch.org/TensorRT/dynamo/torch_compile.html)

- **Preferred API**: `torch.compile(backend="torch_tensorrt", cache_built_engines=True, reuse_cached_engines=True)` — engines cached automatically, reused across runs.
- **AOT alternative**: `torch_tensorrt.compile(ir="dynamo")` → `torch_tensorrt.save(model, "trt.ep")` → `torch_tensorrt.load("trt.ep")`. Manual save/load but gives a portable `.ep` file.
- **`torch_tensorrt.compile()` (current code)**: JIT path, engines live in memory only, CANNOT be serialized. This is what we're using now — **should switch**.
- **ONNX path**: Export to ONNX → `trtexec --saveEngine=model.engine` → load with TRT runtime. Works but more complex than native torch_tensorrt.
- **For Modal**: Point cache env vars to Modal volume (same as TORCHINDUCTOR_CACHE_DIR). First run builds engine (~5min), subsequent runs load from cache instantly.

**Action item**: Switch from `torch_tensorrt.compile()` to `torch.compile(backend="torch_tensorrt", cache_built_engines=True, reuse_cached_engines=True)` in modal_profile.py. This caches engines automatically.

## 2026-04-01: TensorRT attempts — BLOCKED on version mismatch

**Problem**: We're on PyTorch 2.5.1 + torch-tensorrt 2.5.0. The APIs have changed significantly since then. Latest is torch-tensorrt 2.10.0 + PyTorch 2.10.

**What we tried and failed:**
1. `torch_tensorrt.compile()` (old API) — engine built in 467s, then crashed on input shape mismatch (1080 vs 1088 padded). Fixed the shape issue but this API doesn't support engine caching.
2. `torch.compile(backend="torch_tensorrt")` with `immutable_weights=False` — fails with "Engine caching requires make_refittable to be set to True". The `immutable_weights` option is from v2.12 docs; v2.5.0 uses `make_refittable`.
3. `torch.compile(backend="torch_tensorrt")` with `make_refittable=True` — NOT TESTED YET.

**Key findings from the failed TRT run (old API, no caching):**
- TRT engine built successfully in **467.8 seconds** on A100
- Engine warmup worked — inference ran but crashed on shape mismatch
- The model IS TRT-compatible after swapping LayerNorm2d → LayerNorm2dExport
- Shape fix: must pad input to multiple of 16 BEFORE passing to TRT model (NAFNet's internal padding is baked into the TRT graph)

**Decision needed: UPGRADE PyTorch + torch-tensorrt**
- Current: PyTorch 2.5.1 + torch-tensorrt 2.5.0 + CUDA 12.1
- Target: PyTorch 2.7+ + torch-tensorrt 2.7+ (or latest stable)
- This unblocks: proper engine caching, newer Inductor, AOTInductor exports, better TRT op coverage
- Risk: might break existing torch.compile results (need to re-baseline)
- The Modal image pip_install lines need updating in both modal_profile.py and modal_denoise.py

**Correct v2.5.0 API for engine caching (untested):**
```python
model = torch.compile(
    model,
    backend="torch_tensorrt",
    dynamic=False,
    options={
        "enabled_precisions": {torch.float, torch.half},
        "use_python_runtime": True,
        "make_refittable": True,  # v2.5.0 name (NOT immutable_weights)
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "engine_cache_dir": "/mnt/data/trt_engine_cache",
        "min_block_size": 1,
    },
)
```

## 2026-04-01: INT8 quantization research for NAFNet

### Summary

INT8 quantization is viable for image denoising CNNs with ~0.5-1.0 dB PSNR loss. The recommended path is **NVIDIA ModelOpt + TensorRT** (not torchao). Three approaches were evaluated:

| Approach | Maturity for CNNs | Expected Speedup | Complexity |
|----------|-------------------|-------------------|------------|
| ModelOpt + TensorRT INT8 | Best — designed for this | ~1.5-2x over fp16 TRT | Medium |
| torch_tensorrt PTQ (legacy) | Works but being superseded | Same | Medium |
| torchao (PyTorch native) | Poor for CNNs — targets linear layers | Minimal | Low |

**Recommendation: ModelOpt PTQ → TensorRT INT8 compilation.**

### Research findings by topic

#### 1. Does INT8 PTQ work for image denoising CNNs?

Yes, with caveats. Published results:
- A DnCNN denoising model quantized to INT8 lost only **~0.6 dB PSNR** vs fp32 ([ETASR 2024](https://mail.etasr.com/index.php/ETASR/article/download/15428/6155))
- CVPR 2025 Mobile AI challenge ran INT8-quantized super-resolution models with competitive fidelity ([CVPR 2025W paper](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Ignatov_Quantized_Image_Super-Resolution_on_Mobile_NPUs_Mobile_AI_2025_Challenge_CVPRW_2025_paper.pdf))
- IEEE paper "Image Denoising Meets Quantization" confirms PTQ maintains competitive denoising quality ([IEEE Xplore](https://ieeexplore.ieee.org/document/11137609/))
- Key risk: **activation outliers** correlate with image color info. Naive quantization can clip these, causing color shifts. Addressed by entropy calibration (TRT default) or outlier-aware methods ([ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Outlier-Aware_Post-Training_Quantization_for_Image_Super-Resolution_ICCV_2025_paper.pdf))

**Expected quality impact for NAFNet denoising: 0.5-1.0 dB PSNR loss (39.6 → ~38.6-39.1 dB).** This is within the 1-2 dB tolerance.

#### 2. PyTorch quantization landscape (2025-2026)

PyTorch is consolidating quantization into **torchao** ([github.com/pytorch/ao](https://github.com/pytorch/ao)). However:
- torchao's `Int8DynamicActivationInt8WeightConfig` targets **nn.Linear layers only** — it does not quantize nn.Conv2d
- torchao works best for "compute-bound" models (transformers/LLMs), not memory-bandwidth-bound CNNs
- For CNN quantization on NVIDIA GPUs, the recommended path remains TensorRT-based

**torchao is NOT suitable for NAFNet** (pure Conv2d architecture).

#### 3. TensorRT INT8: two paths

**Path A — Legacy torch_tensorrt PTQ (TorchScript IR):**
Uses `torch_tensorrt.ptq.DataLoaderCalibrator` to calibrate, then `torch_tensorrt.compile()` with `enabled_precisions={torch.float, torch.half, torch.int8}`. This is the older TorchScript workflow.

```python
# Legacy approach — works but being superseded
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    calib_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=torch.device("cuda:0"),
)
trt_mod = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 1080, 1920))],
    enabled_precisions={torch.float, torch.half, torch.int8},
    calibrator=calibrator,
)
```

**Path B — ModelOpt + Dynamo (recommended):**
Uses `modelopt.torch.quantization` (mtq) to insert QDQ nodes, then `torch_tensorrt.compile(ir="dynamo")` or `torch.compile(backend="torch_tensorrt")`. This is the modern workflow that supports engine caching.

```python
import modelopt.torch.quantization as mtq
import torch_tensorrt

# Step 1: Calibrate with ModelOpt
def calibrate_loop(model):
    for batch in calib_dataloader:
        model(batch.cuda())

quant_cfg = mtq.INT8_DEFAULT_CFG
mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

# Step 2: Export and compile with TensorRT
with torch.no_grad():
    exported = torch.export.export(model, (dummy_input,))
trt_model = torch_tensorrt.compile(
    exported,
    ir="dynamo",
    arg_inputs=[dummy_input],
    enabled_precisions={torch.int8},
    min_block_size=1,
)
```

#### 4. Mixed precision: keeping LayerNorm in fp16/fp32

TensorRT handles this automatically. When you set `enabled_precisions={torch.float, torch.half, torch.int8}`, TRT will:
- Quantize Conv2d weights per-channel (INT8)
- Quantize activations per-tensor (INT8)
- Keep layers that don't benefit from INT8 (like normalization) in fp16/fp32

ModelOpt also allows per-layer control:
```python
import copy
config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
# Disable quantization for specific layers by name pattern
config["quant_cfg"]["*norm*"] = {"enable": False}
config["quant_cfg"]["*output_quantizer"] = {"enable": False}
```

This is important for NAFNet's `LayerNorm2d` which already does fp32 casting internally.

#### 5. NVIDIA ModelOpt details

- **Package**: `pip install nvidia-modelopt` (or `nvidia-modelopt[torch]`)
- **GitHub**: [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- **Calibration data**: NVIDIA recommends **at least 500 samples** for CNN/ViT models. Our 1000 calibration images are sufficient.
- **Calibration algorithms**: entropy (default, best for most), max (simpler, works for NLP)
- **Key feature**: inserts quantizers into nn.Conv2d and nn.Linear automatically
- **Warning**: ModelOpt has a `disable_conv_quantization` option — some docs suggest channel quantization "doesn't work well with convolutions." This may need testing. If conv quantization hurts quality, we can quantize only the 1x1 pointwise convs and leave 3x3 depthwise convs in fp16.
- **Compatibility**: torch-tensorrt 2.5.0 already prints a warning about missing modelopt — installing it will resolve that and unlock INT8

#### 6. Gotchas and version requirements

- **ModelOpt requires torch_tensorrt >= 2.4**: We have 2.5.0, so compatible
- **torch.export.export()**: Required for the dynamo IR path. NAFNet should export cleanly (pure CNN, no dynamic control flow)
- **LayerNorm2dExport**: Already created in `lib/nafnet_arch.py` — must use `swap_layernorm_for_export()` before quantization since the custom autograd Function won't trace
- **Calibration images must match inference preprocessing**: Same normalization, same resolution (1080p), same dtype pipeline
- **A100 INT8 Tensor Cores**: A100 has INT8 tensor cores that deliver 2x the TOPS of fp16 (624 vs 312 TOPS). This is the theoretical ceiling — memory bandwidth may still be the bottleneck, but INT8 halves the bandwidth per element
- **Depthwise convolutions**: NAFNet uses depthwise separable convs (groups=channels). TensorRT INT8 supports grouped convolutions, but quantization accuracy may vary. Test quality carefully on these layers.

### Action plan

1. **Install modelopt** in the Modal image: `pip_install("nvidia-modelopt[torch]")`
2. **Prepare calibration dataloader**: Load ~500-1000 frames from `data/train_pairs/` input images, resize/normalize to match inference pipeline
3. **Quantize with ModelOpt**: `mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calib_loop)` after calling `swap_layernorm_for_export()`
4. **Compile with TensorRT dynamo IR**: Use `torch_tensorrt.compile(ir="dynamo", enabled_precisions={torch.int8})`
5. **Benchmark**: Compare fps and PSNR vs fp16 TRT baseline
6. **If quality drops > 1 dB**: Try disabling quantization on depthwise conv layers or LayerNorm, or use entropy calibration with more samples
7. **If quality is acceptable**: Cache the INT8 TRT engine on Modal volume for production use

### Sources

- [Torch-TensorRT PTQ docs](https://docs.pytorch.org/TensorRT/user_guide/ptq.html)
- [Torch-TensorRT Quantization guide](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html)
- [VGG16 PTQ example (ModelOpt + TRT)](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/vgg16_ptq.html)
- [NVIDIA Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [ModelOpt PyTorch quantization guide](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)
- [torchao quantization docs](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- [TensorRT working with quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [NVIDIA INT8 quantization whitepaper (2020)](https://arxiv.org/pdf/2004.09602)
- [Outlier-Aware PTQ for Image SR (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Outlier-Aware_Post-Training_Quantization_for_Image_Super-Resolution_ICCV_2025_paper.pdf)
- [Image Denoising Meets Quantization (IEEE 2024)](https://ieeexplore.ieee.org/document/11137609/)

## 2026-04-01: Full optimization audit

### PyTorch upgrade: 2.5.1 → 2.7.1 + torch-tensorrt 2.7.0

- **Target**: torch==2.7.1, torchvision==0.22.1, torch-tensorrt==2.7.0, cu124
- **Engine caching API**: `immutable_weights=False` is correct for 2.7.0 (our existing code was right)
- **15x faster TRT engine builds** in 2.7.0 vs 2.5.0
- **Python 3.10** still supported (range: 3.9-3.13)
- **CUDA 12.4 forward-compatible** with A100 (sm_80) and H100 (sm_90)
- **Breaking changes**: None for pure CNNs. Custom autograd.Function still not TRT-traceable (swap_layernorm_for_export still needed)
- **New feature**: `tiling_optimization_level` option ("none", "fast", "moderate", "full") — worth testing

### Inductor optimizations we missed

- **TORCHINDUCTOR_FREEZING=1**: Inlines model weights as constants, enables constant folding. Reports show 15-30% latency reduction for UNet models. Just an env var, works on 2.5.1+.
- **conv_1x1_as_mm=True**: Converts 1x1 convolutions to GEMM (better Tensor Core utilization). NAFNet uses 1x1 convs extensively (conv1, conv3, conv4, conv5 in every NAFBlock). Expected 5-15% gain.
- **coordinate_descent_tuning=True**: More thorough Triton kernel tuning than default, less expensive than max-autotune.

### LayerNorm2d → GroupNorm(1) swap

Custom `LayerNormFunction` (autograd.Function) is opaque to torch.compile — Inductor cannot fuse it. Replace with `nn.GroupNorm(num_groups=1, num_channels=C)`:
- Mathematically identical (GroupNorm with 1 group = LayerNorm over C,H,W)
- Same weights, no retraining needed
- Inductor can fuse GroupNorm with adjacent convolutions
- PyTorch's native GroupNorm handles fp16→fp32 accumulation internally
- Eliminates explicit `.float()` / `.to(dtype)` round-trips that double memory traffic per norm op
- 72 LayerNorm ops in the model (2 per NAFBlock × 36 blocks)

### AOTInductor (PyTorch 2.6+, stable in 2.7+)

- Compile once → save `.pt2` artifact → zero-warmup loads
- Same steady-state fps as torch.compile (same Inductor backend)
- Eliminates 30-60s warmup per Modal container start
- For production: compile offline, upload `.pt2` to Modal volume, load instantly

### Modal GPU pricing update

A100 dropped from $2.78/hr to $2.10/hr. At 13.4 fps, cost is now ~$2.65/episode (not $3.51).

| GPU | VRAM | $/hr | BW (GB/s) | Projected FPS | $/episode |
|-----|------|------|-----------|---------------|-----------|
| L4 | 24 GB | $0.80 | 300 | 2.6 | $5.24 |
| A10 | 24 GB | $1.10 | 600 | 5.2 | $3.59 |
| A100-40GB | 40 GB | $2.10 | 1,555 | 13.4 (measured) | $2.65 |
| A100-80GB | 80 GB | $2.50 | 2,039 | ~17.6 | $2.41 |
| H100 | 80 GB | $3.95 | 3,350 | ~29 | $2.32 |
| H200 | 141 GB | $4.54 | 4,800 | ~41 | $1.86 |

### INT8 quantization — realistic expectations

- **Realistic speedup: 1.2-1.4x** (not 2x) due to depthwise conv limitations in TensorRT
- TensorRT historically has suboptimal INT8 kernels for depthwise convs (groups=channels)
- TRT 10.x: "no optimized FP8 Convolutions for Group/Depthwise Convolutions" — INT8 recommended but still has limitations
- NAFNet dw_channel=128 (power of 2) avoids the worst regression (60% perf drop for non-power-of-2 groups)
- **Quality risk**: Depthwise convs have "fluctuant dynamic data range across filters" — MobileNet-V1/V2 don't reach baseline with INT8 PTQ alone
- **Mitigation**: Disable quantization on depthwise conv layers if quality drops > 1 dB: `config["quant_cfg"]["*conv2*"] = {"enable": False}`
- **FP8 not viable**: No FP8 kernels for depthwise convs, even on H100
- **ModelOpt 0.42.0** is current, requires PyTorch upgrade first
- **Calibration**: 1000 images sufficient, use entropy calibration (handles activation outliers better than minmax for denoising)

## 2026-04-01: GroupNorm(1) swap — WRONG MATH

The research agent incorrectly stated GroupNorm(1, C) is equivalent to LayerNorm2d. It is NOT:
- **LayerNorm2d**: normalizes over channel dim (dim=1) — mean/var computed across C channels at each (h,w) position
- **GroupNorm(1, C)**: normalizes over C,H,W together — mean/var computed across all channels AND spatial positions
- **GroupNorm(C, C)**: normalizes over H,W per channel — this is InstanceNorm, also not the same

**Correct replacement**: `F.layer_norm(x.permute(0,2,3,1), [C], weight, bias, eps).permute(0,3,1,2)` — this applies LayerNorm over the channel dimension at each spatial position, matching the original. Implemented as `LayerNorm2dCompile` in `lib/nafnet_arch.py`.

The fp32 cast is still needed for fp16 stability (same as original LayerNorm2d).

## 2026-04-01: PyTorch 2.7.1 + torch.compile results

Upgraded from PyTorch 2.5.1 → 2.7.1, cu121 → cu124. All optimizations combined:
- `TORCHINDUCTOR_FREEZING=1`
- `conv_1x1_as_mm=True`
- `LayerNorm2dCompile` (F.layer_norm, Inductor-fusable)
- `channels_last` + `cudnn.benchmark`
- `compile(reduce-overhead)`

| GPU | FPS | Cost/ep | vs previous best |
|-----|-----|---------|-----------------|
| A100-80GB bs=8 | **15.0** | **$2.37** | +12% fps, -11% cost (was 13.4 fps, $2.65) |
| H100 bs=8 | **30.82** | **$2.17** | NEW — 2x A100, cheapest yet |

## 2026-04-01: TensorRT 2.7.0 — BROKEN

torch-tensorrt 2.7.0 with `immutable_weights=False` (engine caching) crashes with `DataDependentOutputException` in `_save_weight_mapping`. The conversion falls back to eager PyTorch on failed subgraphs, giving 3.0 fps (5x slower than torch.compile).

The error is in `_TRTInterpreter.check_weight_equal()` which tries to compare tensor values inside FakeTensor mode — `aten._local_scalar_dense` is not supported in FakeTensor. This is a torch-tensorrt bug.

With `immutable_weights=True` (no caching), TRT might work but the engine would need rebuilding every run (~8 min). Not practical.

**Conclusion**: TensorRT via torch.compile backend is not viable for NAFNet with torch-tensorrt 2.7.0. torch.compile with Inductor backend is faster anyway (15 fps vs 3 fps fallback). INT8 via ModelOpt+TRT is blocked by the same issue.

## 2026-04-01: Batch size scaling with CUDA graphs (reduce-overhead)

On H100 with compile(reduce-overhead):
- bs=8: **30.82 fps** (0.5 GB peak) — best
- bs=16: **11.2 fps** (0.6 GB peak) — 2.75x SLOWER
- bs=32: **OOM** — CUDA graph allocation exhausts 80 GB

CUDA graphs pre-allocate the full execution memory upfront. Larger batch sizes mean more activation memory per graph recording, and the graph replay overhead increases. For this bandwidth-bound model, bs=8 is the sweet spot — it amortizes weight reads without excessive graph overhead.

This confirms the model is bandwidth-bound, not compute-bound. Batching beyond bs=8 provides no benefit.
