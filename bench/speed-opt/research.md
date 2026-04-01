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
