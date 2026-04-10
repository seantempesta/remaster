# NeRV Modal Scaling Plan

## Objective

Scale the exp37 HNeRV denoising configuration (42.9 dB on 16 local frames) to Modal cloud GPUs with more frames, bigger batch size, and optionally a larger model to push quality further.

## Current Best Config (exp37, local RTX 3060)

| Parameter | Value |
|-----------|-------|
| val_psnr | 42.9 dB (peak epoch 109/145) |
| hf_ratio | 0.954 |
| sharpness_ratio | 0.961 |
| params | 4.44M |
| VRAM | 5.4 GB |
| GPU | RTX 3060 6GB |
| batch_size | 1 |
| frames | 16 (12 train + 4 holdout) |
| epochs | 150 (completed 145 in 20 min timeout) |
| enc_strides | 5,3,2,2,2 (total 120x) |
| dec_strides | 5,3,2,2,2 (total 120x) |
| dec_blks | 1,1,1,1,1 |
| enc_dim | 16 |
| fc_dim | 120 |
| reduce | 1.2 |
| lower_width | 12 |
| enc_blocks | 1 |
| skip_connections | True (scale_init=0.1) |
| skip_dropout | 0.0 |
| loss | l1_freq (pixel_weight=10) |
| patch color loss | weight 5.0, 32x32 patches |
| edge_weight | 0.5 (ramped 0 -> 0.5) |
| asym_edge_weight | 0.5 (ramped 0 -> 0.5) |
| optimizer | Prodigy (d_coef=1.0, wd=0.01) |
| weight_decay | 0.01 |

## Key Differences: NeRV vs DRUNet Training

NeRV training is fundamentally different from the existing DRUNet `modal_train.py`:

1. **NeRV fits one video** -- the model IS the video representation. Each model encodes a specific clip. DRUNet trains on thousands of independent image pairs.
2. **No input/target pairs** -- the loss is self-supervised (reconstruct noisy input + auxiliary losses). DRUNet needs paired degraded/clean images.
3. **Epoch = pass through all frames** -- with 16 frames, one epoch is 16 forward passes. With 128 frames, one epoch is 128 passes.
4. **Batch size means different frames** -- batch_size=4 processes 4 DIFFERENT frames simultaneously. This helps BatchNorm statistics and gradient smoothness.
5. **No data augmentation needed** -- the network memorizes specific frames. Augmentation would teach the wrong thing.
6. **Training is short** -- 150-300 epochs (minutes to hours), not 25K iterations (hours to days).

**Conclusion: We need a new Modal wrapper (`cloud/modal_train_nerv.py`), not a modification of `modal_train.py`.** The wrapper is straightforward -- upload frames to Modal Volume, call `train()` from `tools/train_nerv.py`.

## Plan

### Phase 1: Direct Scaling Test (T4, $0.59/hr)

**Goal**: Verify the exp37 config works on Modal with identical settings, then scale frames.

**GPU: T4 (16GB, $0.59/hr)**
- 3x local VRAM. Cheapest option. Perfect for validation.
- T4 FP16: 65 TFLOPS (vs RTX 3060's ~13 TFLOPS) -- roughly 3-5x faster per epoch.

**Experiments:**

| Run | Frames | Batch Size | Epochs | Est. Time | Est. Cost | Purpose |
|-----|--------|------------|--------|-----------|-----------|---------|
| 1a | 16 | 1 | 150 | ~5 min | ~$0.05 | Reproduce exp37 on Modal (sanity check) |
| 1b | 16 | 4 | 150 | ~5 min | ~$0.05 | Test batch_size>1 effect on quality |
| 1c | 64 | 4 | 200 | ~20 min | ~$0.20 | 4x more temporal context |
| 1d | 128 | 4 | 250 | ~45 min | ~$0.45 | 8x more temporal context |
| 1e | 128 | 8 | 250 | ~35 min | ~$0.35 | Larger batch with more frames |

**Total Phase 1 budget: ~$1.10**

**What to watch:**
- Does val_psnr match local (~42.9) for run 1a?
- Does batch_size>1 help or hurt? (BatchNorm with bs=1 is statistically unreliable -- this was flagged in the code audit)
- Does more frames improve quality? (exp30 showed 8 frames < 16 frames; does 64 > 16?)
- Epoch time scaling: T4 should be 3-5x faster than RTX 3060 per forward pass

**Key concern: BatchNorm**
The model uses `nn.BatchNorm2d` in `UpConvBlock`. With batch_size=1, BN computes per-sample statistics (variance=0 during training, only running stats during eval). batch_size>=4 gives meaningful batch statistics. This could be a significant quality improvement. If batch_size>1 helps a lot, consider replacing BN with GroupNorm for consistency across batch sizes.

### Phase 2: Model Scaling (L4, $0.80/hr)

**Goal**: Once we know the optimal frame count, test larger models.

**GPU: L4 (24GB, $0.80/hr)**
- 4x local VRAM. Good cost/VRAM ratio for model scaling.
- L4 FP16: 121 TFLOPS -- roughly 6-9x faster than RTX 3060.

**Model scaling options:**

| Config | fc_dim | enc_dim | dec_blks | Params (est) | VRAM (est, bs=4) |
|--------|--------|---------|----------|--------------|-------------------|
| Baseline (exp37) | 120 | 16 | 1,1,1,1,1 | 4.44M | ~7-8 GB |
| Medium | 170 | 24 | 1,1,1,1,1 | ~8M | ~12-14 GB |
| Large | 200 | 32 | 1,1,2,2,2 | ~14M | ~18-22 GB |
| XL | 240 | 48 | 1,1,2,2,2 | ~22M | ~22-24 GB |

Parameter estimates are rough. The dominant cost is in the decoder (PixelShuffle convolutions):
- fc_dim controls the base decoder width -- scales roughly as fc_dim^2
- dec_blks add extra convolutions per stage -- ~linear cost
- enc_dim is cheap (only affects bottleneck)

**Recommended scaling strategy:**
1. Start with fc_dim=170 (the default in the script) + enc_dim=24. This roughly doubles params.
2. If quality improves, try fc_dim=200 + enc_dim=32 + dec_blks=1,1,2,2,2.
3. Don't go above ~15M params initially -- the model must fit the video, and too much capacity means noise memorization.

**Critical insight from research log**: More params = more noise memorization risk. The spectral bias denoising works BECAUSE the model is capacity-limited. Scaling must be paired with stronger regularization:
- Higher weight_decay (0.02 or 0.03)
- Skip dropout (0.1-0.3)
- More training frames (dilutes per-frame capacity)
- The edge/asym loss ramp is essential to prevent overfitting

| Run | fc_dim | enc_dim | dec_blks | Frames | BS | Epochs | Est. Time | Est. Cost |
|-----|--------|---------|----------|--------|----|--------|-----------|-----------|
| 2a | 170 | 24 | 1,1,1,1,1 | 128 | 4 | 300 | ~30 min | ~$0.40 |
| 2b | 170 | 24 | 1,1,1,1,1 | 128 | 8 | 300 | ~25 min | ~$0.33 |
| 2c | 200 | 32 | 1,1,2,2,2 | 128 | 4 | 300 | ~45 min | ~$0.60 |

**Total Phase 2 budget: ~$1.33**

### Phase 3: Production Scale (L40S, $1.95/hr)

**Goal**: Run the best config from Phase 1+2 at full scale.

**GPU: L40S (48GB, $1.95/hr)**
- Our standard training GPU. 48GB fits any reasonable model + batch size.
- L40S FP16: 366 TFLOPS -- roughly 20x faster than RTX 3060.

**Production run:**
- Best model size from Phase 2
- 300+ frames from a full Firefly scene (5-10 min of video at 1 frame/sec)
- batch_size=8-16
- 500-1000 epochs (let it fully converge)
- Full W&B logging with all visualizations

| Run | Frames | BS | Epochs | Est. Time | Est. Cost |
|-----|--------|----|--------|-----------|-----------|
| 3a | 300 | 16 | 500 | ~60 min | ~$1.95 |
| 3b | 300 | 16 | 1000 | ~120 min | ~$3.90 |

**Total Phase 3 budget: ~$5.85**

## Data Pipeline

### Frame Extraction

Currently: 16 PNGs at `E:/upscale-data/nerv-test/micro_gop_01/` (manually extracted).

For scaling, extract more frames from Firefly source using ffmpeg:

```bash
# Extract 128 frames starting at 20:00 (matching existing test clip location)
ffmpeg -ss 00:20:00 -i "path/to/firefly_s01e01.mkv" \
    -vframes 128 -q:v 1 "E:/upscale-data/nerv-test/clip_128/frame_%04d.png"

# Extract 300 frames for production run
ffmpeg -ss 00:20:00 -i "path/to/firefly_s01e01.mkv" \
    -vframes 300 -q:v 1 "E:/upscale-data/nerv-test/clip_300/frame_%04d.png"
```

### Upload to Modal

Upload PNGs to the existing `upscale-data` Modal Volume:

```python
# In modal_train_nerv.py local_entrypoint
vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
with vol.batch_upload(force=True) as batch:
    for f in sorted(glob.glob("E:/upscale-data/nerv-test/clip_128/*.png")):
        batch.put_file(f, f"/nerv-data/clip_128/{os.path.basename(f)}")
```

Data size estimate:
- 1080p PNG: ~5-6 MB each
- 16 frames: ~90 MB (trivial upload)
- 128 frames: ~700 MB (~2 min upload)
- 300 frames: ~1.6 GB (~5 min upload)

**Use `--skip-upload` after first upload** (same pattern as `modal_train.py`).

### RAM Caching

On Modal, all frames should be cached in RAM (64GB system RAM on Modal containers). The dataset is tiny compared to available RAM. Set `num_workers=0` since the dataset fits entirely in memory.

## Modal Wrapper Design (`cloud/modal_train_nerv.py`)

The wrapper needs to:

1. **Image definition**: Same base as `modal_train.py` (PyTorch 2.11+cu130, prodigyopt, wandb) plus PIL/torchvision (already in the base).
2. **Upload `tools/train_nerv.py`** via `.add_local_file()`.
3. **Volume mount**: Mount `upscale-data` at `/mnt/data` (same as existing).
4. **Function signature**: Expose all exp37 CLI args as function parameters.
5. **Local entrypoint**: Handle data upload, GPU selection, checkpoint download.
6. **W&B**: Use `wandb-api-key` Modal Secret (same as existing).
7. **Graceful stop**: Use `train-signals` Modal Dict (same as existing).

**Key differences from modal_train.py:**
- No teacher/student distillation (NeRV is self-supervised)
- No crop_size (NeRV processes full frames)
- Frame count instead of dataset size
- Epoch-based training instead of iteration-based
- Checkpoint format: `{epoch, model, optimizer, metrics}` (not `{params, iteration, psnr}`)

**Files to include in image:**
```python
.add_local_file("tools/train_nerv.py", remote_path="/root/project/tools/train_nerv.py")
```

No other project files needed -- `train_nerv.py` is fully self-contained (model, dataset, loss, metrics, visualization all in one file).

## GPU Selection Guidance

| GPU | VRAM | Best For | When to Use |
|-----|------|----------|-------------|
| T4 ($0.59) | 16GB | Phase 1: reproduce + frame scaling | bs=1-4, up to 128 frames, model <= 5M params |
| L4 ($0.80) | 24GB | Phase 2: model scaling | bs=4-8, model up to ~15M params |
| A10G ($1.10) | 24GB | Same as L4 but 2x bandwidth | Only if L4 is memory-bandwidth limited |
| L40S ($1.95) | 48GB | Phase 3: production scale | bs=16+, any model size, long runs |

**Do not use H100/B200** -- NeRV training is too short to justify the cost. A T4 run costs $0.05-0.45; the same run on H100 would be ~$0.50-3.00 for marginally faster epochs.

## VRAM Estimation

At 1080p (1920x1080), approximate VRAM usage:

| Component | bs=1 | bs=4 | bs=8 | bs=16 |
|-----------|------|------|------|-------|
| Model params (4.44M, FP16) | 9 MB | 9 MB | 9 MB | 9 MB |
| Optimizer state (Prodigy) | ~100 MB | ~100 MB | ~100 MB | ~100 MB |
| Activations (FP16) | ~1.5 GB | ~6 GB | ~12 GB | ~24 GB |
| Gradients | ~9 MB | ~9 MB | ~9 MB | ~9 MB |
| FFT buffers (loss) | ~0.5 GB | ~2 GB | ~4 GB | ~8 GB |
| Skip connection features | ~0.3 GB | ~1.2 GB | ~2.4 GB | ~4.8 GB |
| **Total (est.)** | **~2.5 GB** | **~9.5 GB** | **~18.5 GB** | **~37 GB** |

**Note**: Local RTX 3060 uses 5.4 GB at bs=1 -- the extra ~3 GB beyond the estimate is PyTorch CUDA context, cuDNN workspace, and AMP scaler overhead. Add ~2-3 GB overhead to all estimates above.

**Practical limits per GPU:**

| GPU | Max batch_size (est.) |
|-----|----------------------|
| T4 (16GB) | 4 (maybe 6 with grad checkpoint) |
| L4 (24GB) | 6-8 |
| L40S (48GB) | 12-16 |

With gradient checkpointing (`--grad-checkpoint`), VRAM drops ~40% but training is ~2x slower. Only use if needed to fit larger batch.

## Training Time Estimation

Epoch time depends on: frames * (forward + backward) / GPU_TFLOPS.

Local RTX 3060 (13 TFLOPS FP16): ~8.3s/epoch for 16 frames at bs=1.

| GPU | TFLOPS | Speedup vs 3060 | 16f epoch | 128f epoch | 300f epoch |
|-----|--------|------------------|-----------|------------|------------|
| RTX 3060 | 13 | 1x | ~8.3s | ~67s | ~156s |
| T4 | 65 | ~3-5x | ~2-3s | ~15-25s | ~35-55s |
| L4 | 121 | ~5-7x | ~1.5-2s | ~10-15s | ~25-35s |
| L40S | 366 | ~15-20x | ~0.5-1s | ~4-7s | ~10-16s |

**Larger batch sizes provide additional speedup** because GPU utilization is higher (more work per kernel launch). bs=4 is roughly 2-3x faster per-frame than bs=1 on these GPUs.

With 128 frames, bs=4, 300 epochs on T4: ~300 * 20s = 100 min. On L4: ~300 * 12s = 60 min. On L40S: ~300 * 5s = 25 min.

## Loss Function Preservation

The exp37 loss is complex and must be exactly preserved. All components are in `compute_loss()` in `train_nerv.py`:

1. **L1 pixel loss** (weight 10.0)
2. **FFT frequency loss** (weight 1.0, implicit)
3. **Patch-level color loss** (weight 5.0, 32x32 avg_pool2d)
4. **Edge preservation loss** (ramped 0 -> 0.5 over epochs)
5. **Asymmetric residual structure loss** (ramped 0 -> 0.5 over epochs)

These are all in the existing `tools/train_nerv.py`. No code changes needed for the loss.

## W&B Integration

Already implemented in `train_nerv.py`. The `--wandb` flag enables it. On Modal, set:
- `--wandb --wandb-project remaster --wandb-entity seantempesta`
- Run name auto-generated as `hnerv-{params}M-enc{enc_dim}-fc{fc_dim}`
- Add `--run-name` override for modal runs (e.g., `modal-hnerv-128f-bs4`)

## Risks and Mitigations

### Risk 1: More frames may not help
- **Evidence**: Exp30 showed 8 frames < 16 frames. But the relationship may not be monotonic -- too many frames could dilute the model's capacity.
- **Mitigation**: Phase 1 tests 16/64/128 frames at constant model size. If 128 is worse than 64, stop there.

### Risk 2: Larger model overfits
- **Evidence**: Exp02 showed wider encoder bottleneck made quality WORSE. Extra decoder blocks (exp10/11) overfitted.
- **Mitigation**: Scale frames first (Phase 1) before model (Phase 2). More frames = more training signal to justify larger model. Pair model scaling with stronger weight_decay and skip_dropout.

### Risk 3: BatchNorm with small batch size
- **Evidence**: Code audit flagged BN with bs=1 as unreliable. The current best result was achieved with bs=1.
- **Mitigation**: Phase 1 run 1b directly tests bs=4 vs bs=1. If bs>1 helps significantly, consider replacing BN with GroupNorm for robustness.

### Risk 4: Training script not portable to Linux
- **Evidence**: `train_nerv.py` uses `os.path` which is cross-platform, and has no Windows-specific code (no `os.add_dll_directory()`, no Win32 APIs). Data paths are passed as arguments.
- **Mitigation**: The script should work as-is on Linux. Test in Phase 1 run 1a.

### Risk 5: Diminishing returns from longer training
- **Evidence**: Exp20 showed 30 min training on the old config overfitted past 20 min. But exp37's loss ramp may allow longer training without overfitting.
- **Mitigation**: Set high epoch count but use `--max-time` as a safety valve. Monitor val_psnr curves in W&B.

## Total Budget Summary

| Phase | GPU | Est. Cost |
|-------|-----|-----------|
| Phase 1 (frame scaling) | T4 | ~$1.10 |
| Phase 2 (model scaling) | L4 | ~$1.33 |
| Phase 3 (production) | L40S | ~$5.85 |
| **Total** | | **~$8.28** |

This is very budget-friendly. Even doubling all estimates puts us under $20.

## Implementation Steps

1. **Extract frames** locally to `E:/upscale-data/nerv-test/clip_128/` and `clip_300/`
2. **Write `cloud/modal_train_nerv.py`** -- Modal wrapper for `tools/train_nerv.py`
3. **Upload 16-frame data** to Modal Volume (sanity check)
4. **Run Phase 1a** -- reproduce exp37 on T4
5. **Compare W&B** -- verify metrics match local
6. **Run Phase 1b-1e** -- frame and batch scaling
7. **Analyze** -- determine optimal frame count and batch size
8. **Run Phase 2** -- model scaling on L4
9. **Run Phase 3** -- production scale on L40S

## Command Templates

```bash
# Phase 1a: Reproduce exp37 on Modal T4
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/remaster/python.exe -m modal run \
    cloud/modal_train_nerv.py \
    --data-dir E:/upscale-data/nerv-test/micro_gop_01 \
    --gpu T4 --frames 16 --batch-size 1 --epochs 150 \
    --fc-dim 120 --enc-dim 16 --dec-blks 1,1,1,1,1 \
    --skip-connections --skip-scale-init 0.1 \
    --edge-weight 0.5 --asym-edge-weight 0.5 \
    --weight-decay 0.01 --wandb \
    --run-name modal-exp37-repro

# Phase 1c: 64 frames, batch_size=4
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/remaster/python.exe -m modal run \
    cloud/modal_train_nerv.py \
    --data-dir E:/upscale-data/nerv-test/clip_64 \
    --gpu T4 --frames 64 --batch-size 4 --epochs 200 \
    --fc-dim 120 --enc-dim 16 --dec-blks 1,1,1,1,1 \
    --skip-connections --skip-scale-init 0.1 \
    --edge-weight 0.5 --asym-edge-weight 0.5 \
    --weight-decay 0.01 --wandb \
    --run-name modal-hnerv-64f-bs4

# Phase 2a: Larger model on L4
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/remaster/python.exe -m modal run \
    cloud/modal_train_nerv.py \
    --data-dir E:/upscale-data/nerv-test/clip_128 \
    --gpu L4 --frames 128 --batch-size 4 --epochs 300 \
    --fc-dim 170 --enc-dim 24 --dec-blks 1,1,1,1,1 \
    --skip-connections --skip-scale-init 0.1 \
    --edge-weight 0.5 --asym-edge-weight 0.5 \
    --weight-decay 0.02 --wandb \
    --run-name modal-hnerv-170fc-128f

# Phase 3a: Production scale on L40S
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/remaster/python.exe -m modal run \
    cloud/modal_train_nerv.py \
    --data-dir E:/upscale-data/nerv-test/clip_300 \
    --gpu L40S --frames 300 --batch-size 16 --epochs 500 \
    --fc-dim 170 --enc-dim 24 --dec-blks 1,1,1,1,1 \
    --skip-connections --skip-scale-init 0.1 \
    --edge-weight 0.5 --asym-edge-weight 0.5 \
    --weight-decay 0.02 --wandb \
    --run-name modal-hnerv-prod-300f
```

## Open Questions

1. **Should we try multiple clips?** Training separate NeRV models for different scenes (interior, exterior, dark, bright) would reveal whether the approach generalizes. But this multiplies the experiment count.

2. **GOP segmentation**: For a full episode, we'd need to train one NeRV model per ~300-600 frame segment. The plan above focuses on a single clip. Production deployment would need: scene detection -> segment -> train N models -> decode each segment.

3. **Deployment latency**: NeRV decoding is a forward pass through the decoder only (encoder not needed at inference). A 4.44M model at FP16 should decode at 50-100+ fps on RTX 3060. But we need one trained model per GOP -- is per-GOP training cost-effective vs DRUNet?

4. **Comparison with DRUNet teacher**: The DRUNet teacher achieves 53.27 dB on paired data (clean targets). NeRV achieves 42.9 dB without any clean targets. Is the ~10 dB gap acceptable for the benefit of not needing clean training data? Or should NeRV be used as a complementary tool (e.g., temporal consistency post-processing)?
