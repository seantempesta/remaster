# NAFNet Speed Investigation

## Problem

NAFNet-width64 (distilled from SCUNet) runs at **0.7 fps** on Modal L4 at 1080p — the same speed as SCUNet. The point of distillation was speed. A 42-min Firefly episode (61K frames) costs ~$19 at this rate. Target: ≤$3/episode (~5 fps or better).

## What We Know

### Model Details
- **NAFNet-width64**: 115.98M parameters, 442MB checkpoint
- Architecture: U-Net with 4 encoder levels [2,2,4,8 blocks], 12 middle blocks, 4 decoder levels [2,2,2,2 blocks]
- Channel widths: 64 → 128 → 256 → 512 → 1024 (middle) → back down
- Each NAFBlock: LayerNorm → 1×1 conv (expand 2×) → 3×3 depthwise conv → SimpleGate → channel attention → 1×1 conv → residual → FFN (same pattern)
- Pure CNN — no attention, no dynamic control flow. Should be torch.compile and TensorRT friendly.
- fp16 inference works (LayerNorm casts to fp32 internally, returns fp16)

### Measured Performance on Modal L4 (24GB VRAM)
| Config | FPS | Notes |
|--------|-----|-------|
| NAFNet batch_size=8 eager | OOM | Falls back to batch_size=1, gets 0.7 fps |
| NAFNet batch_size=2 eager | 0.7 | No OOM, steady rate |
| NAFNet batch_size=2 compiled | untested | torch.compile warmup started but run was stopped before steady-state speed was measured |
| SCUNet batch_size=6 eager | 0.7 | For comparison — same speed |

### OOM at batch_size=8
NAFNet OOMed at batch_size=8 on L4 (24GB) despite model being only 221MB in VRAM. The pipeline's OOM handler catches this and falls back to processing one frame at a time, but it does this for every batch (tries 8, OOMs, processes 1-by-1, repeats).

Likely cause: NAFNet-width64 at 1080p has large intermediate feature maps. At the first encoder level (1920×1088×64 channels), each activation tensor is ~253MB in fp16. With DW_Expand=2, the expanded conv output is 128 channels = ~506MB. The LayerNorm2d fp16→fp32 cast temporarily doubles a feature map. Skip connections (`encs` list) accumulate ~4GB across all levels. Peak memory for batch_size=8 likely exceeds 24GB when accounting for intermediates, CUDA context, and cuDNN workspace.

This has **not been profiled** — the numbers above are analytical estimates. `torch.cuda.max_memory_allocated()` should be measured.

### torch.compile Status
torch.compile was enabled with `mode="reduce-overhead"` but the run was stopped during JIT warmup. The warmup dummy used 256×256 input, but the real 1080p input triggers a separate compilation (different shape). No steady-state compiled speed data exists.

The denoise_nafnet.py pipeline warms up with a 256×256 dummy, which doesn't help — it should warm up at the actual input resolution (1920×1088 padded).

## What Hasn't Been Tried

1. **Profiling actual memory usage** — `torch.cuda.max_memory_allocated()` at different batch sizes to find the true maximum that fits in 24GB
2. **torch.compile at 1080p** — measure actual compiled throughput. CNNs typically see 1.5-3x speedup from compile
3. **Batch size 3-4** — might fit in 24GB and give better GPU utilization than batch_size=2
4. **A10G or A100 GPU** — A10G is ~40% faster compute than L4, A100 has 40GB+ VRAM for larger batches
5. **Smaller NAFNet (width=32 or width=16)** — 4x or 16x fewer FLOPs. Would need retraining from SIDD-width32 or from scratch
6. **TensorRT export** — maximum optimization for CNN inference. torch.compile with `mode="max-autotune"` is a lighter alternative
7. **ONNX Runtime** — another inference optimization path
8. **Tiling** — process 1080p in tiles to reduce peak memory, allowing larger batches. Adds overhead from tile overlap but might enable batch_size=8+
9. **Reducing the model after distillation** — prune or quantize the trained model
10. **Different architecture entirely** — if NAFNet-width64 can't be made fast enough, a smaller model (e.g., RRDB, lightweight U-Net) trained the same way might be better

## Suggested Investigation Plan

### Phase 1: Profile and tune (cheap, ~$0.20 on Modal)
Write a small Modal script that:
1. Loads NAFNet at 1080p on L4
2. Measures `max_memory_allocated()` at batch_size=1,2,4,6,8
3. Measures fps with and without torch.compile (after proper 1080p warmup)
4. Tests `torch.compile(mode="max-autotune")` vs `mode="reduce-overhead"`
5. Reports results

This tells us: (a) what batch size actually fits, (b) whether compile gives a real speedup, (c) what the realistic achievable fps is.

### Phase 2: Based on results
- If compiled NAFNet at optimal batch size gives ≥3 fps: update the pipeline and process Firefly
- If not: consider width=32 NAFNet (retrain, ~$2 on Modal) or TensorRT export
- If nothing gets NAFNet fast enough: evaluate whether SCUNet on a faster GPU (A100) or with TensorRT is more cost-effective

## Approach

Study `reference-code/autoresearch/` (Karpathy's autoresearch) for methodology on rapid iteration loops for ML optimization. The key idea: write small, fast experiments that test one thing at a time, measure results, and iterate. Don't try to optimize everything at once.

For this investigation, that means:
1. Write a small Modal profiling script (not a full pipeline run)
2. Measure one variable at a time (batch size, compile mode, GPU type)
3. Decide based on data, not estimates
4. Keep experiments under $0.50 each

## Files

- `lib/nafnet_arch.py` — NAFNet architecture (read this to understand the model)
- `pipelines/denoise_nafnet.py` — inference pipeline (the OOM handler is in the batch processing loop)
- `cloud/modal_denoise.py` — Modal wrapper (where GPU type and batch_size defaults are set)
- `checkpoints/nafnet_distill/nafnet_best.pth` — current best checkpoint (iter 1000, 56.82 dB)
- `reference-code/autoresearch/` — Karpathy's autoresearch framework (study for iteration methodology)
