# Training Guide

How to train the remaster models from scratch or fine-tune existing checkpoints.

## Overview

Training uses a **teacher-student distillation** approach:

1. **Build training data** -- Extract frames, generate clean targets with SCUNet GAN, create degraded inputs
2. **Train teacher** -- Large DRUNet (32.6M params) learns enhancement with perceptual loss
3. **Distill student** -- Small DRUNet (1.06M params) learns to replicate teacher via feature matching
4. **Fine-tune on full frames** -- Fix artifacts at frame edges and in dark regions

All training runs on [Modal](https://modal.com) cloud GPUs. Local GPU is used for inference only.

## Data Pipeline

### 1. Extract source frames

```bash
python tools/build_training_data.py --extract-only
```

Extracts 1 frame per 500 source frames from each video, proportional to content length. Frames are saved to `data/originals/` with per-source prefixes and a `meta.pkl` manifest.

### 2. Generate clean targets

```bash
python tools/build_training_data.py --denoise
```

Runs SCUNet GAN + Unsharp Mask (strength=1.0) on each extracted frame. The combined denoise+sharpen target teaches the model to go beyond artifact removal into detail recovery.

### 3. Build degraded inputs

```bash
python tools/build_training_data.py --build-inputs
```

Creates a 3-way degradation mix:
- **33% raw originals** -- Real HEVC compression artifacts
- **33% + noise** -- Synthetic Gaussian noise on top of compression
- **33% + edge-aware blur + noise** -- Teaches sharpening and deblurring

### 4. Verify data

```bash
python tools/verify_data.py
```

Checks pair completeness, readability, and per-source counts.

### Data directory structure

```
data/
  originals/       Raw extracted frames + meta.pkl
  training/
    train/         Training pairs: input/ + target/
    val/           Validation pairs: input/ + target/
    meta.pkl       DataFrame with sigma, split, degradation_type
```

---

## Teacher Training

The teacher is a large DRUNet that establishes the quality ceiling.

```bash
modal run cloud/modal_train.py \
    --arch drunet --nc-list 64,128,256,512 --nb 4 \
    --checkpoint-dir checkpoints/drunet_teacher \
    --optimizer prodigy --perceptual-weight 0.05 --batch-size 64 \
    --ema --wandb --resume
```

### Key settings

| Setting | Value | Notes |
|---------|-------|-------|
| Architecture | DRUNet nc=[64,128,256,512] nb=4 | 32.6M params |
| Loss | Charbonnier + DISTS (0.05) | Perceptual prevents softness |
| Optimizer | Prodigy | Auto-tuned LR, no manual scheduling |
| Batch size | 64 | 256x256 crops, fits in 48GB (L40S) |
| EMA | Enabled | Exponential moving average of weights |
| Pretrained | drunet_deblocking_color.pth | Gaussian denoising weights from KAIR |

### Expected results

- 13K+ iterations to convergence
- PSNR: ~53 dB vs SCUNet GAN targets
- Sharpness: 107% of original source

---

## Student Distillation

The student learns to replicate the teacher at 10x the speed.

```bash
modal run cloud/modal_train.py \
    --arch drunet --nc-list 16,32,64,128 --nb 2 \
    --teacher checkpoints/drunet_teacher/final.pth --teacher-model drunet \
    --checkpoint-dir checkpoints/drunet_student \
    --feature-matching-weight 0.1 --optimizer prodigy --batch-size 192 \
    --ema --wandb --resume
```

### Key settings

| Setting | Value | Notes |
|---------|-------|-------|
| Architecture | DRUNet nc=[16,32,64,128] nb=2 | 1.06M params |
| Loss | Charbonnier + Feature Matching (0.1) | L1 on teacher encoder features |
| Feature adapters | 1x1 conv per encoder level | Aligns channel dimensions |
| Batch size | 192 | Smaller model = larger batch |

---

## Full-Frame Fine-Tuning

Fixes dark-area artifacts and edge effects that crop-based training misses.

```bash
modal run cloud/modal_train.py \
    --arch drunet --nc-list 16,32,64,128 --nb 2 \
    --teacher checkpoints/drunet_teacher/final.pth --teacher-model drunet \
    --checkpoint-dir checkpoints/drunet_student \
    --feature-matching-weight 0.1 --perceptual-weight 0.05 --optimizer prodigy \
    --batch-size 8 --crop-size 0 --max-iters 5000 \
    --ema --wandb --resume --fresh-optimizer
```

Key difference: `--crop-size 0` uses full 1920x1080 frames instead of 256x256 crops. Batch size drops to 8 to fit in VRAM.

---

## Modal Cloud Setup

Training runs on Modal's serverless GPU infrastructure.

### Prerequisites

```bash
pip install modal
modal token set
```

Requires a Modal account with billing. Create a W&B secret for experiment tracking:

```bash
modal secret create wandb-api-key WANDB_API_KEY=your_key_here
```

### GPU selection

Default is L40S ($1.95/hr, 48GB VRAM). Change with `--gpu`:

```bash
modal run cloud/modal_train.py --gpu H100 ...   # Faster, $3.95/hr
modal run cloud/modal_train.py --gpu B200 ...   # Fastest, $6.25/hr
```

### Data upload

Training data persists on the Modal volume between runs. `--skip-upload` is the default. Pass `--no-skip-upload` only after rebuilding training data.

### Stopping training

```bash
python tools/stop_training.py
```

Sends a graceful stop signal. The running training job will save a checkpoint and exit.

---

## Checkpoint Format

| File | Contents | Use |
|------|----------|-----|
| `final.pth` | `{"params": state_dict}` | Inference, teacher weights |
| `best.pth` | `{"params": state_dict, "iteration": int, "psnr": float}` | Best validation PSNR |
| `latest.pth` | Full training state (model + optimizer + scheduler + EMA) | Resume training |

Use `final.pth` for inference and as teacher weights during distillation.

---

## W&B Experiment Tracking

All Modal training runs log to Weights & Biases automatically.

- **Project:** `remaster`
- **Logged metrics:** train/val losses, PSNR, LR, training speed, VRAM usage
- **Logged artifacts:** side-by-side sample images, gradient histograms, best model checkpoint
- **Disable:** `--no-wandb`
