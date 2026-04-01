# NAFNet Distillation Guide

Train a fast NAFNet model to match SCUNet quality on your specific content.

## Why

SCUNet produces great quality (39.6 dB PSNR) but runs at 0.6 fps. NAFNet is a pure CNN that's faster and optimizable (torch.compile, TensorRT), but the pretrained SIDD checkpoint only scores 37.3 dB on compression artifacts. Distillation closes the gap: in testing, even 5 frames + 100 iterations matched SCUNet quality (39.5 dB).

## Quick Start (Local, ~1 hour total)

### Step 1: Generate training pairs (~25 min for 720 frames)

```bash
# Uses SCUNet to denoise frames as pseudo ground truth
# Crash-safe: restarts skip already-processed frames
python training/generate_pairs.py \
    --input-dir data/frames_mid_1080p \
    --output-dir data/train_pairs
```

This creates:
```
data/train_pairs/
  input/frame_00001.png    # original (compressed)
  input/frame_00002.png
  target/frame_00001.png   # SCUNet denoised (pseudo GT)
  target/frame_00002.png
```

### Step 2: Train locally (~40 min for 5K iterations)

```bash
python training/train_nafnet.py \
    --data-dir data/train_pairs \
    --max-iters 5000 \
    --batch-size 2 \
    --val-freq 500
```

Saves to `checkpoints/nafnet_distill/`:
- `nafnet_best.pth` — best validation PSNR (standalone weights)
- `nafnet_latest.pth` — latest checkpoint (full state for resume)

### Step 3: Test the result

```bash
python pipelines/denoise_nafnet.py \
    --input data/clip_mid_1080p.mp4 \
    --checkpoint checkpoints/nafnet_distill/nafnet_best.pth \
    --max-frames 30
```

## Full Training (Modal, ~$2-3)

### Step 1: Generate pairs (same as above, or on Modal)

```bash
# Local is fine — only ~25 min for 720 frames
python training/generate_pairs.py \
    --input-dir data/frames_mid_1080p \
    --output-dir data/train_pairs
```

### Step 2: Train on Modal A10G

```bash
modal run cloud/modal_train.py \
    --data-dir data/train_pairs \
    --max-iters 50000
```

This automatically:
1. Uploads training pairs to Modal volume
2. Uploads NAFNet pretrained checkpoint
3. Runs training on A10G GPU (~1-2 hours)
4. Downloads the best checkpoint to `checkpoints/nafnet_distill/`

### Step 3: Denoise a full episode

```bash
# Local (with torch.compile for speed)
python pipelines/denoise_nafnet.py \
    --input "E:/plex/tv/Firefly.../S01E02.mkv" \
    --checkpoint checkpoints/nafnet_distill/nafnet_best.pth \
    --compile

# Or on Modal
modal run cloud/modal_denoise.py \
    --input "E:/plex/tv/Firefly.../S01E02.mkv"
```

## Scaling Up Training Data

720 frames from one 30s clip may not generalize to all episodes. To improve:

```bash
# Extract frames from multiple episodes
python tools/extract_clip.py --input "E:/plex/tv/.../S01E01.mkv" --output data/frames_s01e01 --max-frames 200
python tools/extract_clip.py --input "E:/plex/tv/.../S01E03.mkv" --output data/frames_s01e03 --max-frames 200

# Generate SCUNet pairs for each
python training/generate_pairs.py --input-dir data/frames_s01e01 --output-dir data/train_pairs
python training/generate_pairs.py --input-dir data/frames_s01e03 --output-dir data/train_pairs
# (crash-safe append — won't overwrite existing pairs)
```

## Key Parameters

| Parameter | Local test | Full training |
|-----------|-----------|---------------|
| `--max-iters` | 100-5000 | 50000 |
| `--batch-size` | 2 (6GB VRAM) | 8 (A10G 24GB) |
| `--lr` | 2e-4 | 2e-4 |
| `--val-freq` | 50-500 | 1000 |
| Training pairs | 5-720 | 500-2000 |

## Troubleshooting

- **OOM locally**: Reduce `--batch-size` to 1, or use `--crop-size 192`
- **Modal upload slow**: Training pairs are ~2GB for 720 frames. Upload happens once, cached on volume.
- **Quality not good enough**: Add more diverse training data (different episodes, scenes with different lighting)
- **NAFNet fp16 issues**: Already fixed in `lib/nafnet_arch.py` — LayerNorm2d casts to fp32 internally
