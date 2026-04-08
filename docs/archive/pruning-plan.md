# Gradual Structured Pruning — Transfer Prompt

## Objective

Take the fine-tuned DRUNet teacher (32.6M params, 53.37 dB PSNR, ~5 fps) and gradually prune it during training to produce a fast model that hits **30+ fps at 1080p on RTX 3060** while maintaining quality close to the teacher.

The final pruned model will be a standard dense network (no special sparse runtime needed) that works with ONNX export, TensorRT INT8, and torch.compile.

## What We Have

### Fine-tuned DRUNet teacher
- **Architecture:** UNetRes from `reference-code/KAIR/models/network_unet.py` (class `UNetRes`)
- **Config:** `in_nc=3, out_nc=3, nc=[64,128,256,512], nb=4, act_mode='R', bias=False`
- **Params:** 32.6M, Conv+ReLU only (no BN, no attention, no LayerNorm)
- **Checkpoint:** `checkpoints/drunet_teacher/final.pth` (124.5 MB, key='params')
- **Quality:** 53.37 dB PSNR vs SCUNet targets (better than NAFNet w32_mid4 at 49.5 dB)
- **Speed:** ~5 fps on RTX 3060 FP16 (too slow for real-time)

### Speed targets (benchmarked on RTX 3060)
| Config | Params | FP16 fps | TRT INT8 fps |
|--------|--------|----------|-------------|
| nc=[64,128,256,512] nb=4 | 32.6M | 5.1 | — |
| nc=[64,128,256,512] nb=1 | 9.19M | 20.4 | — |
| nc=[32,64,128,256] nb=2 | 4.25M | 11.6 | — |
| nc=[32,64,128,256] nb=1 | 2.30M | 20.4 | — |
| nc=[16,32,64,128] nb=2 | 1.06M | 30.1 | 54.8 |
| nc=[16,32,64,128] nb=4 | 2.04M | 16.3 | — |

To hit 30+ fps FP16 we need roughly nc=[16,32,64,128] nb=2 equivalent (~1M params). With TRT INT8 this gives ~55 fps.

### Training infrastructure
- **Script:** `training/train.py` — unified training with EMA, CUDA profiling, graceful stop, sample images, loss curves
- **Data:** 3,382 training pairs + 362 val pairs in `data/mixed_pairs/` and `data/mixed_val/` (6 sources, see `docs/training-data-plan.md`)
- **Cloud:** Modal with L40S ($1.95/hr). Script: `cloud/modal_train.py`
- **Optimizer options:** AdamW or Prodigy (parameter-free LR)

### INT8 validation
The target architecture (DRUNet nc=[16,32,64,128] nb=2) has been validated end-to-end:
- ONNX export: clean graph (Conv, Relu, Add, ConvTranspose only)
- TRT FP16: 42.5 fps
- TRT INT8: 54.8 fps
- See `bench/validate_int8_pipeline.py`

## Pruning Strategy Options

### Option A: Gradual structured channel pruning (recommended)
Use `torch-pruning` (pip install torch-pruning) which has DepGraph for handling UNet skip connections.
1. Load the full 32.6M param teacher
2. Set up a MetaPruner or GrowingRegPruner with `pruning_ratio=0.75, iterative_steps=15`
3. Train for ~50K iters: 35K with gradual pruning, 15K fine-tuning at final size
4. The pruner physically removes channels — model actually shrinks
5. DepGraph ensures skip connections stay consistent (this is critical for UNet)
6. Export the final pruned model as a standard dense network

Key: the importance criterion should be GroupNormImportance or TaylorImportance (not random). GrowingRegPruner adds a regularization term that pushes unimportant channels toward zero before pruning, making the removal smoother.

### Option B: Progressive depth reduction
Start with nb=4 (full depth), gradually reduce to nb=1 by removing least-important ResBlocks. Simpler than channel pruning but less flexible — you can only remove in block-sized increments.

### Option C: Combined width + depth pruning
Prune both channels AND remove blocks. Most flexible but requires careful dependency tracking.

### Option D: Knowledge distillation into fixed small architecture
Skip pruning entirely. Train the small DRUNet (nc=[16,32,64,128] nb=2) from scratch using the teacher for online distillation. We tried this and it converged slowly — but with Prodigy optimizer and better LR, it might work now that we have a much better teacher (53.4 dB vs the old 49.5 dB).

## Key Considerations

1. **Structured pruning gives real speedup** — the resulting model is physically smaller, works on any hardware. Unstructured sparsity (including 2:4) requires special hardware support and gives less speedup.

2. **UNet skip connections** — you CANNOT independently prune encoder and decoder channels. When you prune encoder output channels, the corresponding decoder input (via skip connection) must match. torch-pruning's DepGraph handles this automatically.

3. **The teacher is Conv+ReLU only** — no BN, so BNScaleImportance won't work. Use GroupNormImportance (L2 norm across coupled channel groups) or TaylorImportance (gradient × activation).

4. **INT8 quantization comes AFTER pruning** — prune first to get the right size, then INT8 quantize the pruned model for an additional ~2x speedup.

5. **Quality monitoring** — validate PSNR at each pruning step. If quality drops too fast, slow down the pruning schedule or use GrowingRegPruner for smoother transitions.

## Key Files

- `training/train.py` — unified training script (modify to add pruning hooks)
- `reference-code/KAIR/models/network_unet.py` — UNetRes architecture
- `reference-code/KAIR/models/basicblock.py` — ResBlock, conv helpers
- `checkpoints/drunet_teacher/best.pth` — fine-tuned teacher checkpoint
- `bench/validate_int8_pipeline.py` — INT8 validation script (adapt for pruned model)
- `cloud/modal_train_plainnet.py` — Modal training wrapper
- `data/mixed_pairs/` — 3,382 training pairs (input/ + target/)
- `data/mixed_val/` — 362 validation pairs

## Reference Research

- **torch-pruning:** https://github.com/VainF/Torch-Pruning — structured pruning with DepGraph
- **NVIDIA sparsity blog:** https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/
- **Gradual magnitude pruning (Zhu & Gupta 2018):** cubic sparsity schedule
- **GrowingReg:** adds growing L1 regularization to push channels toward zero before pruning
- **Prodigy optimizer:** `pip install prodigyopt` — parameter-free LR, set lr=1.0

## Success Criteria

1. Pruned model hits **30+ fps FP16** at 1080p on RTX 3060 (or 55+ fps TRT INT8)
2. PSNR remains **above 45 dB** vs SCUNet targets (acceptable quality loss from 53.4 dB teacher)
3. Model exports cleanly to ONNX with only Conv, Relu, Add ops
4. Visual quality on test clips is noticeably better than the compressed input
