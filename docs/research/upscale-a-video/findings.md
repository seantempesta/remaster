# Upscale-A-Video: Evaluation Findings

**Date:** 2026-04-12
**Status:** Concluded -- too slow/expensive for training target generation
**Cost spent:** ~$5-6 total across all Modal runs

## What We Tested

| Run | Input | Tiles | Steps | GPU | Time | Cost | Result |
|-----|-------|-------|-------|-----|------|------|--------|
| 1 | 256x144, no tile | 0 | 10 | A10G | 25s | ~$0.01 | Pipeline works, too soft |
| 2 | 256x144, no tile | 0 | 30 | A10G | 65s | ~$0.02 | 31.7% more temporally consistent, faces soft |
| 3 | 960x536, tiled | 28 | 30 | L40S | 24min | ~$0.80 | 30.8% temporal improvement, background sharp, faces still soft |
| 4 | 1920x1080, tiled | 60 | 30 | A100-80 | ~58min | ~$2.42 | Running (noise=80, video VAE) |

## Key Findings

### 1. The model works and produces temporally consistent output
- 30-31% reduction in frame-to-frame pixel differences (measured in center 256x256 region)
- Temporal attention within 8-frame windows provides genuine consistency
- Text prompts guide enhancement ("clean, sharp, high quality film")

### 2. It is fundamentally too slow for our use case
The math doesn't work at any scale:

| Input res | Tiles/window | Time/window | 7K frames (875 windows) | Cost |
|-----------|-------------|-------------|------------------------|------|
| 256x144 | 1 | 65s | 16 hours | ~$30 |
| 960x536 | 28 | 24min | 350 hours | ~$680 |
| 1920x1080 | 60 | 58min | 846 hours | ~$2,115 |

Compare: **SCUNet GAN processes 7K frames in ~1 hour for ~$1.**

### 3. The VAE attention bottleneck is severe
The VAE decoder has O(n^2) spatial self-attention in its mid-block. This forces:
- Per-frame decoding (T=1) to avoid T*H*W token explosion
- Small tiles (128-192px) to keep attention under GPU VRAM
- More tiles = more pipeline calls = more time

This is an architectural limitation, not an engineering one. You can't tile your way out of O(n^2) per-pixel attention in the decoder.

### 4. Higher input resolution helps quality but kills speed
- 144p input: pure hallucination, soft faces
- 536p input: better structure, background sharp, faces still soft (motion blur from temporal averaging)
- 1080p input: best quality (model sees all real detail) but 60 tiles per 8-frame window

### 5. The diffusion process itself is the bottleneck
Each tile needs 30 UNet forward passes (with CFG = 60 forwards). Even with fast GPUs, this is ~50-60s per tile. There's no way around this without:
- Fewer steps (hurts quality)
- Consistency distillation (needs retraining the model)
- A completely different architecture

## Why UAV Is Wrong for This Use Case

UAV was designed for **offline video upscaling** -- process one short clip and wait. Our use case requires processing **thousands of training frames** as a batch job. The per-frame cost is 100-1000x too high.

The temporal consistency is real and valuable, but we're paying for it with a diffusion model when simpler approaches could achieve similar consistency at a fraction of the cost.

## Better Paths Forward

### Path A: SCUNet GAN + Optical Flow Post-Processing (recommended)
**Cost: ~$2-5 for 7K frames. Implementation: ~1 day.**

1. Run SCUNet GAN per-frame (existing pipeline, ~1 hour, ~$1)
2. Compute RAFT optical flow between consecutive frames (~50ms/pair, trivial)
3. For each frame: warp previous cleaned frame forward, blend with current cleaned frame
4. Result: SCUNet-quality spatial detail + flow-enforced temporal consistency

This borrows UAV's best idea (flow-guided temporal propagation) without the diffusion cost. The RAFT model is already in our repo (`reference-code/Upscale-A-Video/models_video/RAFT/`).

### Path B: Temporal DRUNet Teacher (already planned)
**Cost: ~$10-20 for training. Speed: 5fps at inference.**

The 9-channel input approach from `docs/research/temporal-context/prd.md`:
- Input: prev_cleaned + current_noisy + next_noisy (concatenated)
- The teacher learns temporal consistency during training, not at inference
- Student distills this into the same fast architecture

This is the most architecturally sound approach and was already on our roadmap.

### Path C: UAV for Golden Reference Subset
**Cost: ~$5-10. One-time.**

Process 100-200 diverse frames with UAV at 1080p to create a small "golden" reference set. Use these for:
- Perceptual loss calibration (DISTS reference targets)
- Visual quality benchmarking
- GAN discriminator training data (if we ever add adversarial loss)

Not for bulk training targets, but for calibrating what "ideal" output looks like.

## Architecture Notes for Future Reference

- UNet: [256,512,512,1024] channels, CLIP cross-attention, temporal self-attention at 6 locations
- VAE: 3D with spatial attention in mid-block (the OOM bottleneck). Video VAE adds LR conditioning skip via SFT.
- Per-frame VAE decode required (monkey-patch `pipeline.decode_latents_vsr`)
- Weights: 2.6GB UNet + 211MB VAE (3d) + text encoder from HF
- Stored on Modal volume at `/mnt/data/uav_weights/upscale_a_video/`
- Dependencies: torch 2.0.1, diffusers 0.16.0, huggingface-hub 0.23.5 (pinned for compat)

## Files

- `cloud/modal_uav_eval.py` -- Modal evaluation script (working, tiled)
- `cloud/uav_patches/pipeline_upscale_a_video.py` -- Patched pipeline (short_seq=1)
- `docs/research/upscale-a-video/evaluation.md` -- Architecture analysis
- `docs/research/upscale-a-video/prd.md` -- Original test plan
