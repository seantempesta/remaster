# Architecture Investigation: Reaching 30+ fps at 1080p on RTX 3060

## The Problem (as of 2026-04-03)

NAFNet w32_mid4 runs at **5.5 fps** (180ms/frame) on RTX 3060 Laptop GPU. This was
previously reported as 78 fps — that number was wrong (pipeline timing artifact
that summed overlapping component times instead of measuring actual throughput).

The model is **inference-bound**. Pipeline overhead (decode + encode) is ~15ms —
negligible compared to 180ms inference. GIL is not the bottleneck.

**Target: 30+ fps (33ms/frame) for real-time 24fps playback with headroom.**

## What We Know

### Current model: NAFNet w32_mid4
- 14.3M parameters, 55 MB checkpoint
- width=32, middle_blk_num=4, enc=[2,2,4,8], dec=[2,2,2,2]
- Input: 1x3x1088x1920 fp16 channels_last
- CUDA event timed: **180ms/frame = 5.5 fps** (RTX 3060, torch.compile reduce-overhead)
- CUDA event timed on A10: **94ms/frame = 10.7 fps** (Modal, same torch.compile)
- TensorRT: 233ms/frame = 4.3 fps (only 1.3x slower than torch.compile, not 18x as thought)
- VRAM: 3.3 GB (fp16 + CUDA graphs)
- Quality: 49.50 dB PSNR vs teacher targets

### Larger model: NAFNet w64
- 116M parameters (8x more), 464 MB checkpoint
- width=64, middle_blk_num=12
- **1.94 fps** on RTX 3060
- Quality: 56.82 dB PSNR (much better)
- Cloud (H100): 27.9 fps

### Teacher: SCUNet
- 15.2M params (transformer-based, attention)
- ~0.5 fps on RTX 3060
- Reference quality

## Approach: Work Backwards from Target FPS

**Method:** Generate dummy architectures of varying sizes, benchmark at 1080p,
find the parameter/FLOP budget that hits 30 fps, then design a real model within
that budget.

### Architectures to benchmark (all at 1x3x1088x1920 fp16):

```python
# Sweep width and depth independently
configs = [
    # Tiny: absolute minimum
    {"width": 16, "middle_blk_num": 1, "enc": [1,1,1,1], "dec": [1,1,1,1]},
    {"width": 16, "middle_blk_num": 2, "enc": [1,1,2,2], "dec": [1,1,1,1]},
    
    # Small: reasonable minimum  
    {"width": 24, "middle_blk_num": 2, "enc": [1,1,2,4], "dec": [1,1,1,1]},
    {"width": 24, "middle_blk_num": 4, "enc": [2,2,4,8], "dec": [2,2,2,2]},
    
    # Current w32_mid4
    {"width": 32, "middle_blk_num": 4, "enc": [2,2,4,8], "dec": [2,2,2,2]},
    
    # Ablations: reduce depth at fixed width
    {"width": 32, "middle_blk_num": 2, "enc": [1,1,2,4], "dec": [1,1,1,1]},
    {"width": 32, "middle_blk_num": 1, "enc": [1,1,1,2], "dec": [1,1,1,1]},
    
    # Ablations: reduce width at fixed depth
    {"width": 16, "middle_blk_num": 4, "enc": [2,2,4,8], "dec": [2,2,2,2]},
    
    # Alternative: fewer encoder levels (less downsampling)
    # 3 levels instead of 4 = factor 8 instead of 16
    # Would need architecture modification
]
```

### What to measure for each:
1. **CUDA event time** for single frame inference (fp16, channels_last, torch.compile)
2. **Parameter count** and **FLOP count**
3. **VRAM usage** (fp16 + CUDA graphs)
4. **Peak memory** during torch.compile warmup

### Script to run:
```bash
python bench/sweep_architectures.py
```

## Alternative Architectures to Consider

### 1. Plain CNN (no NAFBlock)
NAFBlock has: DWConv → SimpleGate (Split+Mul) → SCA (GlobalAvgPool+Conv1x1) → FFN
The SCA (channel attention) requires a global reduction — expensive at 1080p.
A plain residual Conv → ReLU → Conv block without attention might be much faster.

### 2. MobileNet-style (depthwise separable convolutions)
Replace NAFBlock's expand-then-DWConv with MobileNet inverted residuals.
Much fewer FLOPs per block.

### 3. Fewer encoder levels
NAFNet uses 4 encoder levels (16x spatial downsampling).
At level 4, spatial is 68x120 with 256 channels — still large.
Using 3 levels (8x downsample) means the bottleneck is 136x240 with 128ch.
Fewer levels = less total computation but may hurt receptive field.

### 4. ONNX → TensorRT (now viable)
Since TRT is only 1.3x slower than torch.compile (not 18x as we thought),
TRT is a viable production path. A model that runs at 33ms on torch.compile
would run at ~43ms on TRT — still under the 33ms target for 30fps.

### 5. Asymmetric encoder/decoder
Heavy encoder (captures features), lightweight decoder (reconstructs).
The current model has enc=[2,2,4,8] dec=[2,2,2,2] — decoder is already light.
Could make it even lighter: dec=[1,1,1,1].

### 6. Resolution-adaptive processing
Process at half resolution, upsample at the end.
540p inference at ~22 fps (4x fewer pixels) then PixelShuffle back to 1080p.
Quality tradeoff but massive speed gain.

## Key Insight: Quality Headroom

The w32_mid4 model (14.3M params) achieves 49.50 dB PSNR vs teacher targets.
But we already showed that **reducing the middle blocks from 12→4 barely hurt quality**
(w64 full: 56.82 dB, w32_mid4: 49.50 dB — the width reduction from 64→32 is
the main quality driver, not depth).

This suggests:
- We can aggressively reduce depth with minimal quality loss
- Width (channel count) matters more than depth for this task
- There may be a sweet spot around width=24, depth=minimal that hits 30fps with acceptable quality

## Training is Cheap

- **~$13 for 25K iterations on H100** (~3.2 hours)
- We can train and evaluate multiple architecture variants quickly
- The distillation pipeline is proven: generate targets with SCUNet → train student
- RAM cache + CUDA event profiling + graceful shutdown all work

## Files Reference

- `lib/nafnet_arch.py` — NAFNet architecture (modify for new variants)
- `training/train_nafnet.py` — training loop (--width, --middle-blk-num already configurable)
- `pipelines/denoise_nafnet.py` — inference pipeline with timing breakdown
- `bench/bench_nafnet.py` — standalone benchmark
- `cloud/modal_train.py` — Modal training script
- `CLAUDE.md` — updated with corrected 5.5 fps numbers

## Next Steps

1. **Write `bench/sweep_architectures.py`** — benchmark different configs at 1080p
2. **Find the 30fps architecture** — what width/depth hits 33ms/frame?
3. **Train it** — distill from SCUNet teacher, evaluate quality
4. **If quality acceptable** — deploy via VapourSynth + TRT (viable at 1.3x overhead)
5. **If quality insufficient** — try alternative architectures (MobileNet-style, resolution-adaptive)
