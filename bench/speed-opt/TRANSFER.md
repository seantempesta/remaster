# Transfer Prompt for Next Conversation

## Read these files first
- `CLAUDE.md` — environment constraints, gotchas, project overview
- `docs/distillation-guide.md` — training setup, quality issues, improvement plan
- `training/train_nafnet.py` — training script (loss, dataset, optimizer)
- `cloud/modal_train.py` — Modal training wrapper
- `cloud/modal_denoise.py` — production pipeline (H100, 27.9 fps)

## Current State

**Speed: SOLVED.** 27.9 fps end-to-end on H100, ~$2.40/episode, 37 min for a full Firefly episode. Pipeline is stable (61K frames, 0 errors).

**Quality: NEEDS WORK.** First episode (Firefly S01E02) has two issues:
1. **Slightly soft** — fine detail (hair, fabric, film grain) is over-smoothed
2. **Dark shadows still noisy** — denoising insufficient in low-light areas

## Root Causes

The current model was trained with:
- **Only 5,350 iters** of planned 50,000 (stopped early)
- **Charbonnier loss only** — smooth L1-like loss that penalizes all errors equally, tends to produce blurry results
- **256px random crops** — model never sees full-frame context, dark regions may be undersampled
- **1,032 training pairs** from a few episodes — limited diversity
- **Best checkpoint from iter 1,000** — very early in training

## What to Do Next

1. **Add perceptual loss** — LPIPS or VGG feature loss preserves texture/sharpness. Weight it alongside Charbonnier (e.g., 1.0 * charbonnier + 0.1 * lpips).

2. **Train longer** — run the full 50K iters. Loss was still improving at 5K.

3. **Larger crops** — try 384px or 512px crops to capture more context (especially dark regions that span large areas). Needs more VRAM — use A100 or reduce batch size.

4. **More training data** — add pairs from dark scenes specifically. The current 1,032 pairs may not have enough shadow content.

5. **Check SCUNet teacher quality in shadows** — if the teacher's shadow denoising is weak, the student can't do better. Compare SCUNet vs input in dark regions.

6. **Consider mixed loss** — Charbonnier for pixel accuracy + FFT loss for frequency preservation + LPIPS for perceptual quality.

## Inference Pipeline (don't change this, it's working)

- `cloud/modal_denoise.py` — H100, cpu=8, apt ffmpeg + libx264, PyAV decode
- `pipelines/denoise_nafnet.py` — PyTorch 2.7.1, torch.compile(reduce-overhead), CUDA graphs with padded shapes, pre-allocated pinned memory, double-buffered
- Best config: `--model nafnet --checkpoint checkpoints/nafnet_distill/nafnet_best.pth --compile`

## Critical Rules
- Do NOT run GPU tasks locally (RTX 3060 6GB)
- Modal command: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/script.py`
- Download large files with `modal volume get` CLI (parallel), not `vol.read_file()` (sequential, hangs)
- H100/A100 have NO NVENC — use libx264 for encoding
