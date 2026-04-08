# Training Data Plan

## Objective

Build a diverse, high-quality training dataset that teaches the model to denoise, remove compression artifacts, and recover detail from degraded video content.

## Staged Pipeline

Data is built in three independent stages, each resumable and skippable:

### Stage 1: Extract Originals (`--extract-only`)

- Probe each source video for total frame count
- Sample proportionally: 1 frame per 500 source frames
- Extract with NVDEC hwaccel, save raw PNG to `data/originals/`
- Compute per-frame metrics (noise_level, laplacian_var, sobel_mean)
- Save DataFrame to `data/originals/meta.pkl`

### Stage 2: Denoise with SCUNet GAN (`--denoise`)

- Run every original through SCUNet GAN (pretrained perceptual/adversarial denoiser)
- SCUNet GAN denoises AND sharpens in one pass — no sigma tuning needed
- Apply light unsharp mask (strength=1.0, sigma=1.5) to push detail slightly further
- Split 90/10 train/val (stratified by source, fixed seed)
- Save targets to `data/training/{train,val}/target/`
- ~0.5 fps on RTX 3060 (one-time cost)

### Stage 3: Build Inputs (`--build-inputs`)

- For each frame, randomly choose degradation type:
  - **~33%**: raw original unchanged (model sees real artifacts)
  - **~33%**: original + light Gaussian noise (sigma 1-5)
  - **~33%**: original + edge-aware blur (from clean target edge map) + noise
- Parallel CPU processing (ThreadPoolExecutor, 8 workers)
- Save inputs to `data/training/{train,val}/input/`

## Key Design Decisions

### Why SCUNet GAN instead of DRUNet?

DRUNet is PSNR-optimized — it removes noise but softens detail. SCUNet GAN was trained with adversarial + VGG perceptual loss, so it denoises while preserving and enhancing texture. No sigma tuning needed — it adapts to the input content automatically. A light unsharp mask (1.0) pushes detail slightly further.

### Why keep originals?

Raw extracted frames are permanent. If the target generation strategy evolves, we regenerate targets and inputs from cached originals without re-extracting from video.

### Why proportional sampling?

1 frame per 500 source frames ensures each source contributes proportionally to its content length. No more hand-picking arbitrary counts.

### Why mixed inputs?

33% raw originals teach the model to handle real compression artifacts without synthetic degradation. 33% with added noise teaches robustness. 33% with edge-aware blur teaches detail recovery. The model must be stable on all three.

## Sources (Proportional Sampling)

| Source | Total Frames | Episodes | Samples (1/500) |
|--------|-------------|----------|-----------------|
| Firefly S01 | ~938K | 14 | ~1,876 |
| The Expanse S02 | ~818K | 13 | ~1,635 |
| One Piece S01 | ~653K | 8 | ~1,306 |
| Squid Game S02 | ~619K | 7 | ~1,237 |
| Dune Part Two | ~238K | 1 | ~476 |
| Foundation S03 | ~213K | 3 | ~426 |
| **Total** | **~3.48M** | | **~6,956** |

## Directory Structure

```
data/
  originals/              ~7K raw extracted frames (permanent cache)
    meta.pkl              DataFrame: noise_level, laplacian_var, sobel_mean per frame
  training/
    train/
      input/              ~6,260 degraded frames
      target/             ~6,260 clean denoised frames
    val/
      input/              ~696 degraded frames
      target/             ~696 clean denoised frames
    meta.pkl              DataFrame: sigma, split, degradation_type, all metrics
  calibration/
    grids/                Pre-rendered sigma comparison images
    samples.pkl           Sample frame info for labeling
    labels.pkl            Human-selected sigma labels
    sigma_model.pkl       Fitted noise->sigma mapping
    sigma_curve.png       Visualization of the calibration curve
  analysis/               Visualization outputs (sharpness plots, etc.)
  archive/                Old data, episode comparisons, demo clips
  output/                 Training artifacts from Modal
```

## Build Commands

```bash
# Stage 1: Extract originals (~55 min at ~2 fps)
python tools/build_training_data.py --extract-only

# Stage 2: Denoise with SCUNet GAN (~4 hrs at ~0.5 fps)
python tools/build_training_data.py --denoise

# Stage 3: Build inputs (~2 min, parallel)
python tools/build_training_data.py --build-inputs

# Verify + analyze
python tools/verify_data.py
python tools/measure_sharpness.py
python tools/visualize_sharpness.py
```

## Denoiser Details

- **Model:** SCUNet GAN (`scunet_color_real_gan.pth`) — Swin-Conv-UNet with adversarial training
- **Architecture:** SCUNet, in_nc=3, config=[4,4,4,4,4,4,4], dim=64 (~15M params)
- **Weights:** `reference-code/SCUNet/model_zoo/scunet_color_real_gan.pth` (69 MB)
- **Post-processing:** Light unsharp mask (strength=1.0, sigma=1.5) on GAN output
- **Why GAN:** Perceptual + adversarial loss preserves texture while denoising (vs PSNR models that soften)
