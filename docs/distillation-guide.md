# NAFNet Distillation Guide

Train a fast NAFNet model to match SCUNet quality on compression artifacts, then use it for full episode processing.

## Overview

SCUNet produces great quality (39.6 dB PSNR) but is slow (0.6 fps locally, 0.7 fps on Modal L4). It's memory-bandwidth-bound due to Swin Transformer windowed attention — ceiling ~2-3 fps even with TensorRT. NAFNet is a pure CNN that's torch.compile/TensorRT friendly, but the pretrained SIDD checkpoint only scores 37.3 dB on compression artifacts. Distillation closes the gap.

## Current State (2026-04-01)

### What's Done
- **Training pairs**: 1032 pairs in `data/train_pairs/` (720 from mid clip + 100 from E01 + 106 from E04 + 106 from E08). Pairs are {input=compressed frame, target=SCUNet denoised frame} at full 1080p.
- **Local training test**: 200 iters, batch_size=2, loss 0.258→0.0025, PSNR 45.62→48.21 dB
- **Modal training**: Ran to ~5350/50000 iters on A10G before stopping. Best validation PSNR 56.82 dB at iter 1000. Loss plateaued at ~0.0015 by iter 3000. Training was 1.8 it/s (IO-bound from Modal Volume), would have taken ~8 hours for 50K iters.
- **Best checkpoint**: `checkpoints/nafnet_distill/nafnet_best.pth` — downloaded from Modal volume, from iter 1000
- **End-to-end test**: Processed `data/clip_mid_1080p.mp4` (720 frames) on Modal L4. Output is `data/clip_mid_1080p_nafnet.mkv`. Video + mux pipeline works correctly. 0 errors.
- **modal_denoise.py**: Updated to support both `--model scunet` (default) and `--model nafnet --checkpoint path`. Uploads checkpoint to volume, runs NAFNet pipeline, muxes audio/subs, downloads result.

### What's NOT Done — Speed Problem
NAFNet on Modal L4 at 1080p runs at **0.7 fps** — the same speed as SCUNet. The distillation produced a model with matching quality but **no speed gain** in the current configuration. A full 42-min episode (61K frames) would take ~24 hours and cost ~$19 on L4.

The speed problem needs investigation before processing full episodes. See `docs/nafnet-speed-investigation.md` for details and next steps.

### Training Observations
- **NaN loss**: Occurred at iter 50 and iter 500 (both right at LR milestones — start and end of warmup). Model recovered both times. Gradient clipping (`--grad-clip`) was disabled. Future runs should enable it.
- **IO bottleneck on Modal**: Training was 1.8 it/s on A10G (batch_size=8, 256x256 crops). This is slower than expected — the DataLoader reads full 1080p PNGs from the Modal Volume, then random-crops. Pre-cropping or caching data locally inside the container would help.
- **Loss plateau**: Loss dropped quickly (0.26→0.002 in first 200 iters) then slowly improved (0.0019 at 1K, 0.0015 at 5K). Returns diminish fast.

## Key Files

| File | Purpose |
|------|---------|
| `training/generate_pairs.py` | SCUNet pair generation (crash-safe) |
| `training/train_nafnet.py` | Distillation training (Charbonnier + AdamW + cosine LR) |
| `cloud/modal_train.py` | Modal wrapper — uploads data, trains on A10G, downloads checkpoint |
| `cloud/modal_denoise.py` | Modal denoise — supports both SCUNet and NAFNet models |
| `pipelines/denoise_nafnet.py` | NAFNet streaming video pipeline (threaded IO) |
| `lib/nafnet_arch.py` | Standalone NAFNet architecture (fp16 LayerNorm fix) |
| `checkpoints/nafnet_distill/` | Trained weights (gitignored) |
| `data/train_pairs/` | 1032 training pairs (gitignored) |

## How to Run

### Generate training pairs
```bash
# Local (~25 min for 720 frames at 0.47 fps on RTX 3060)
python training/generate_pairs.py \
    --input-dir data/frames_mid_1080p \
    --output-dir data/train_pairs
```

### Train locally (quick test)
```bash
python training/train_nafnet.py \
    --data-dir data/train_pairs \
    --max-iters 200 \
    --batch-size 2 \
    --val-freq 100
```

### Train on Modal
```bash
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run \
    cloud/modal_train.py \
    --data-dir data/train_pairs \
    --max-iters 50000
```

### Denoise on Modal
```bash
# NAFNet
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run \
    cloud/modal_denoise.py \
    --input "path/to/episode.mkv" \
    --model nafnet \
    --checkpoint checkpoints/nafnet_distill/nafnet_best.pth

# SCUNet (original, for comparison)
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run \
    cloud/modal_denoise.py \
    --input "path/to/episode.mkv"
```

## Training Parameters

| Parameter | Quick test | Full training | Notes |
|-----------|-----------|---------------|-------|
| `--max-iters` | 200 | 50000 | Diminishing returns after ~5K based on loss curve |
| `--batch-size` | 2 | 8 | Limited by VRAM: 2 for 6GB, 8 for 24GB |
| `--lr` | 2e-4 | 2e-4 | Matches NAFNet original paper |
| `--val-freq` | 100 | 1000 | How often to compute validation PSNR |
| `--crop-size` | 256 | 256 | Random crop size for training patches |
| Training pairs | 5+ | 500-2000 | 1032 currently available |

## Known Issues
- **NAFNet fp16 LayerNorm**: Fixed in `lib/nafnet_arch.py`. Casts to fp32 for normalization, returns fp16.
- **Modal volume paths**: Upload uses volume-relative paths (`/input/file`), container uses mount prefix (`/mnt/data/input/file`). Must call `vol.reload()` before reading.
- **Modal x265 threading**: Add `pools=4` to x265-params or encoding is single-threaded.
- **Windows conda run**: Breaks with Unicode. Use `PYTHONUTF8=1 .../python.exe -m modal run` directly.
- **torch.compile on SCUNet**: Does NOT work (dynamic W/SW window branches). Works on NAFNet.
- **NaN during training**: Occurs at LR warmup boundaries. Enable `--grad-clip 1.0` for stability.
