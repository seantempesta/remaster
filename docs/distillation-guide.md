# NAFNet Distillation Guide

Train a fast NAFNet model to match SCUNet quality on compression artifacts, then use it for full episode processing.

## Overview

SCUNet produces great quality (39.6 dB PSNR) but is slow (0.6 fps locally, 0.7 fps on Modal L4). It's memory-bandwidth-bound due to Swin Transformer windowed attention — ceiling ~2-3 fps even with TensorRT. NAFNet is a pure CNN that's torch.compile/TensorRT friendly, but the pretrained SIDD checkpoint only scores 37.3 dB on compression artifacts. Distillation closes the gap: in testing, even 5 frames + 100 iterations matched SCUNet quality (39.5 dB).

## What's Done vs What's Left

### Done
- `training/generate_pairs.py` — crash-safe SCUNet pair generation (tested locally, 0.47 fps)
- `training/train_nafnet.py` — NAFNet fine-tuning with Charbonnier loss, AMP, cosine LR (tested locally: 100 iters, loss 0.021→0.0025, PSNR 49.56→50.66)
- `cloud/modal_train.py` — Modal wrapper for cloud training (NOT yet tested on Modal)
- `pipelines/denoise_nafnet.py` — NAFNet streaming video pipeline (tested locally)
- `lib/nafnet_arch.py` — standalone NAFNet with fp16 LayerNorm fix
- `bench/bench_nafnet.py` — quality comparison benchmark

### TODO (for next session)
1. **Generate full training pairs** — run generate_pairs.py on 720 frames (~25 min local)
2. **Train on Modal** — run modal_train.py with 50K iterations (~$2-3, 1-2 hours)
3. **Benchmark distilled model** — compare fps + quality vs SCUNet
4. **Add NAFNet to modal_denoise.py** — currently only supports SCUNet. Need `--model nafnet --checkpoint path` flag so full episodes can use the distilled model end-to-end on Modal
5. **Test torch.compile on L4** — NAFNet should compile cleanly (pure CNN, no dynamic control flow). Could be 2-5x speedup on top of baseline NAFNet speed.
6. **Process full Firefly S01E02** — the actual goal

## Step-by-Step

### Step 1: Generate training pairs

```bash
# Local (~25 min for 720 frames at 0.47 fps on RTX 3060)
# Crash-safe: restarts skip already-processed frames
python training/generate_pairs.py \
    --input-dir data/frames_mid_1080p \
    --output-dir data/train_pairs
```

Creates `data/train_pairs/{input,target}/frame_XXXXX.png` — original + SCUNet denoised pairs.

If you have existing SCUNet outputs, skip regeneration:
```bash
python training/generate_pairs.py \
    --input-dir data/frames_mid_1080p \
    --scunet-dir data/frames_mid_scunet \
    --output-dir data/train_pairs
```

For better generalization, add frames from multiple episodes:
```bash
# Extract frames from other episodes first
python tools/extract_clip.py --input "E:/plex/tv/Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)/Firefly (2002) - S01E01 - Serenity (1080p x265 Silence).mkv" --output data/frames_s01e01 --max-frames 200

# Then generate pairs (appends to existing, won't overwrite)
python training/generate_pairs.py --input-dir data/frames_s01e01 --output-dir data/train_pairs
```

### Step 2: Train

**Local quick test** (verify everything works before spending Modal credits):
```bash
python training/train_nafnet.py \
    --data-dir data/train_pairs \
    --max-iters 100 \
    --batch-size 2 \
    --val-freq 50
```

**Full training on Modal** (~$2-3, 1-2 hours on A10G):
```bash
# IMPORTANT: Use python.exe directly, NOT conda run (Unicode issues)
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run \
    cloud/modal_train.py \
    --data-dir data/train_pairs \
    --max-iters 50000
```

Checkpoints save to `checkpoints/nafnet_distill/`:
- `nafnet_best.pth` — best validation PSNR (standalone weights, use for inference)
- `nafnet_latest.pth` — full state dict (use for resuming training)

### Step 3: Verify quality

```bash
# Quick visual test
python pipelines/denoise_nafnet.py \
    --input data/clip_mid_1080p.mp4 \
    --checkpoint checkpoints/nafnet_distill/nafnet_best.pth \
    --max-frames 30

# Benchmark against SCUNet
python bench/bench_nafnet.py --frames 30
# (update bench_nafnet.py to include distilled checkpoint comparison)
```

### Step 4: Process full episodes

**Local:**
```bash
python pipelines/denoise_nafnet.py \
    --input "E:/plex/tv/Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)/Firefly (2002) - S01E02 - The Train Job (1080p BluRay x265 Silence).mkv" \
    --checkpoint checkpoints/nafnet_distill/nafnet_best.pth \
    --compile
```

**On Modal (faster, with NVENC):**
```bash
# NOTE: cloud/modal_denoise.py currently only supports SCUNet.
# It needs a --model nafnet --checkpoint flag added before this works.
# See TODO item #4 above.
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run \
    cloud/modal_denoise.py \
    --input "E:/plex/tv/Firefly.../S01E02.mkv"
```

## Architecture Notes

### Why distillation works
NAFNet-SIDD-width64 is already a strong denoiser (40.3 dB on SIDD). It just doesn't know about compression artifacts (JPEG blocking, banding, mosquito noise). Fine-tuning from the SIDD checkpoint only needs to adjust the model's understanding of the noise distribution — the denoising capability is already there. That's why even 5 frames + 100 iterations closes the quality gap.

### Why NAFNet is faster
SCUNet uses Swin Transformer blocks with 32,640 tiny 8x8 windowed attention ops per frame at 1080p. These are memory-bandwidth-bound — the L4's compute cores sit idle waiting for data. NAFNet uses only convolutions + channel attention (global avg pool), which have much better compute/memory ratios and parallelize trivially.

### Key files
| File | Purpose |
|------|---------|
| `training/generate_pairs.py` | SCUNet pair generation (crash-safe) |
| `training/train_nafnet.py` | Distillation training (Charbonnier + AdamW + cosine LR) |
| `cloud/modal_train.py` | Modal wrapper — uploads data, trains on A10G, downloads checkpoint |
| `pipelines/denoise_nafnet.py` | NAFNet streaming video pipeline (threaded IO, same as SCUNet pipeline) |
| `lib/nafnet_arch.py` | Standalone NAFNet architecture (fp16 LayerNorm fix included) |
| `checkpoints/nafnet_distill/` | Where trained weights go (gitignored) |

### Known issues
- **NAFNet fp16 LayerNorm**: Fixed in `lib/nafnet_arch.py`. The custom `LayerNormFunction` autograd loses precision in fp16. Fix: casts to fp32 for normalization, returns fp16.
- **Modal volume paths**: Upload uses volume-relative paths (`/input/file`), container uses mount prefix (`/mnt/data/input/file`). Must call `vol.reload()` before reading.
- **Modal x265 threading**: Add `pools=4` to x265-params or encoding is single-threaded.
- **Windows conda run**: Breaks with Unicode. Use `PYTHONUTF8=1 .../python.exe -m modal run` directly.
- **torch.compile on SCUNet**: Does NOT work (dynamic W/SW window branches). Works on NAFNet.

## Training Parameters

| Parameter | Quick test | Full training | Notes |
|-----------|-----------|---------------|-------|
| `--max-iters` | 100 | 50000 | More iters = better, diminishing returns after 50K |
| `--batch-size` | 2 | 8 | Limited by VRAM: 2 for 6GB, 8 for 24GB |
| `--lr` | 2e-4 | 2e-4 | Matches NAFNet original paper |
| `--val-freq` | 50 | 1000 | How often to compute validation PSNR |
| `--crop-size` | 256 | 256 | Random crop size for training patches |
| Training pairs | 5+ | 500-2000 | More diverse = better generalization |
