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
- **TODO**: Verify if torch_tensorrt.compile() is the right API vs torch.compile(backend="torch_tensorrt"). Check if upgrading to PyTorch 2.7 + torch-tensorrt latest gives better results.
