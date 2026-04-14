# Upscale-A-Video: Evaluation for Training Target Generation

**Date:** 2026-04-11
**Status:** Evaluation in progress -- testing at multiple input resolutions
**Repository:** `reference-code/Upscale-A-Video/` (CVPR 2024, NTU S-Lab)

## Summary

Upscale-A-Video is a diffusion-based video super-resolution model with temporal consistency via flow-guided propagation. It is a **4x spatial upscaler**, not a same-resolution denoiser. Using it requires a downscale-then-upscale roundtrip. The key question: does the temporally consistent hallucinated detail from the diffusion prior produce better training targets than SCUNet GAN's per-frame denoising?

Three test paths are proposed (see `prd.md`): 270p->1080p (pure hallucination), 540p->4K->1080p (moderate + supersampling), and 1080p->8K->1080p (max detail, expensive). Initial evaluation cost: ~$1.25 on A100.

## Architecture Overview

Built on Stable Diffusion x4 Upscaler with video-specific modifications:

| Component | Details |
|-----------|---------|
| **UNet** | 3D UNet, channels [256, 512, 512, 1024], cross-attention with CLIP text embeddings, temporal attention modules at all levels. Input: 7 channels (4 latent + 3 from noisy LR image) |
| **VAE** | 3D VAE (encoder + decoder with temporal convolutions). Decoder has LR image conditioning via SFT (spatial feature transform) -- fuses low-res input directly into decode path |
| **Text encoder** | CLIP (from SD x4 upscaler). Accepts text prompts like "best quality, extremely detailed" |
| **Flow network** | RAFT bidirectional optical flow for temporal propagation |
| **Propagator** | Feature propagation module -- warps latent features along optical flow with forward-backward consistency check. Non-learnable mode (fuse with 0.5 blending) or learnable (deformable convolution alignment) |
| **Captioner** | Optional LLaVA 1.5 13B for auto-generating per-video text descriptions |
| **Scheduler** | DDIM with custom step_v0/step_vt for x0-prediction + propagation interleaving |

### Inference Pipeline

1. Read video frames -> normalize to [-1, 1]
2. (Optional) LLaVA generates caption from first frame
3. Add noise to input at specified `noise_level` (0-200, default 120)
4. Encode noisy input -> latent space (4x spatial downsample)
5. DDIM denoising loop (default 30 steps):
   - UNet forward with classifier-free guidance (2x forward per step)
   - At specified propagation steps: RAFT flow + feature propagation for temporal consistency
   - Process in 8-frame windows with 2-frame overlap for long videos
6. VAE decode latents with LR skip connection -> output at **4x input resolution**
7. (Optional) Color correction (AdaIn or Wavelet)

### Key Parameters

- `noise_level` (0-200): Controls fidelity vs. quality tradeoff. Higher = more creative freedom, lower = closer to input
- `guidance_scale` (default 6): CFG weight for text prompt adherence
- `inference_steps` (default 30): DDIM steps, more = higher quality
- `propagation_steps` (e.g., [24,26,28]): Which denoising steps apply temporal propagation
- `tile_size` (default 256): Auto-tiling kicks in at input >= 384x384

## Why It Doesn't Fit Our Use Case

### 1. Fundamental 4x Upscaling Architecture

The model is hardwired for 4x spatial upscaling:

- **UNet input**: 7 channels = 4 latent + 3 from LR image. The LR image is at the *input* resolution; the latent space operates at 1/8 of the *output* resolution. For a 480p input, latents are 60x (480/8) -- producing 1920p output.
- **VAE decoder**: Has 3 upsampling blocks that each 2x the spatial dimension (total 8x on latents = 4x on input).
- **VAE skip connection**: The `condition_fuse` SFT block injects the low-res input into the decoder, but the decoder still outputs at 4x input resolution.

There is no configuration or parameter to make this produce same-resolution output.

### 2. The Downscale-Upscale Roundtrip Problem

To get 1080p output from 1080p input, we'd need to:
- Downscale 1080p to 480x270
- Run UAV to get 1920x1080
- Use 1080p output as training target

**Problems:**
- **Information loss**: Downscaling to 270p destroys fine HEVC artifact structure. The model can't remove artifacts it can't see.
- **Hallucinated detail**: Diffusion 4x upscaling generates plausible-looking detail from its training distribution, not from the actual source content. This is the opposite of what we want -- our student model should learn to recover *real* detail, not hallucinated textures.
- **Content drift**: With `guidance_scale=6` and `noise_level=120`, the model substantially transforms the input. Fine details (text, thin lines, subtle textures) may be replaced with plausible but incorrect alternatives.
- **Our training setup**: The student learns input->target mapping. If targets contain hallucinated detail, the student learns to hallucinate too -- but without the diffusion model's distribution knowledge, it would produce inconsistent artifacts.

### 3. Processing Cost

Even on high-end Modal GPUs, diffusion-based processing is extremely slow:

| Config | Per-frame estimate | 7K frames | Cost (Modal) |
|--------|-------------------|-----------|--------------|
| 480p input, 30 steps, A100 | ~30-60s | 58-117 hrs | $122-$246 |
| 480p input, 30 steps, B200 | ~15-30s | 29-58 hrs | $181-$362 |
| 480p input, 20 steps, L40S | ~40-80s | 78-156 hrs | $152-$304 |

Compare to SCUNet GAN: ~0.5s/frame on A10G = ~1 hour for 7K frames (~$1.10).

Even if quality were dramatically better, 100-300x cost increase for offline target generation is hard to justify.

### 4. VRAM Requirements

- **UNet**: ~2.5B parameters (channels [256,512,512,1024], cross-attention). FP16 = ~5GB
- **VAE**: ~150M params (3D encoder/decoder). FP16 = ~300MB
- **CLIP text encoder**: ~350M params. FP16 = ~700MB
- **RAFT** (if using propagation): ~5M params = ~10MB
- **LLaVA** (optional): 13B params = 13-26GB
- **Activations**: 8-frame temporal window at 480p with CFG doubles -> significant
- **Total without LLaVA**: ~20-30GB for 480p input with tiling
- **Total with LLaVA on same GPU**: 40-50GB+

**Minimum GPU**: L40S (48GB) without LLaVA, or A100-80GB with LLaVA.

### 5. Dependency Hell

requirements.txt pins ancient versions:
- `torch==2.0.1` (we use 2.11)
- `diffusers==0.16.0` (current is ~0.30+)
- `transformers==4.28.1` (current is ~4.45+)
- `xformers>=0.0.20` (we avoid xformers on Windows; would need SDPA patches)
- `accelerate==0.18.0`

Would need significant porting work for a Modal image, likely incompatible with our existing training environment.

### 6. License: Non-Commercial Only

NTU S-Lab License 1.0: "Redistribution and use for **non-commercial purpose**." Commercial use requires contacting the authors. While our current use is personal, this constrains future options.

## What About the Temporal Propagation?

The propagation module is the most interesting piece. It uses:

1. **RAFT bidirectional flow**: Computes forward/backward optical flow between frames
2. **Forward-backward consistency check**: Identifies occluded regions where flow is unreliable
3. **Feature warping**: Warps latent features along flow vectors
4. **Blending**: In non-learnable mode, blends warped features with current frame (50/50)

This is applied at specific denoising steps (e.g., steps 24, 26, 28 out of 30), which means it refines temporal consistency in the *latent space* during the later (fine-detail) stages of denoising.

**Relevance to our project**: The propagation concept is more directly applicable to our planned recurrent temporal DRUNet (see `docs/research/temporal-context/prd.md`). But using RAFT flow at training time (not inference) to generate temporally-aligned targets would be much simpler and more efficient than running the full UAV pipeline.

## Comparison: SCUNet GAN vs Upscale-A-Video for Target Generation

| Criterion | SCUNet GAN + USM | Upscale-A-Video |
|-----------|-----------------|-----------------|
| **Same-res enhancement** | Yes (native) | No (4x upscale only) |
| **Content fidelity** | High -- denoises/enhances without hallucination | Low -- generates plausible but potentially incorrect detail |
| **Temporal consistency** | None (per-frame) | Good (flow-guided propagation) |
| **HEVC artifact removal** | Good -- trained on real noise/compression | Unknown -- trained on bicubic downscale + noise degradation |
| **Speed** | ~2 fps (A10G) | ~0.02 fps (A100) |
| **VRAM** | ~4GB | ~20-30GB |
| **Cost for 7K frames** | ~$1 | ~$150-300 |
| **Dependencies** | Drop-in, already integrated | Major porting needed |
| **License** | Apache 2.0 | Non-commercial only |
| **Text-guided** | No | Yes -- but guidance tends to hallucinate |

## Verdict: Not Recommended

**Don't use Upscale-A-Video as a training target generator.** The 4x upscaling architecture is a fundamental mismatch for same-resolution denoising. The information loss from the required downscale-upscale roundtrip, combined with hallucinated detail, would produce worse training targets than SCUNet GAN -- at 100-300x the cost.

## What Would Work Better

If we want to improve training target quality beyond SCUNet GAN, the more promising paths are:

1. **SCUNet GAN ensembling**: Run SCUNet GAN with different noise levels, average results for smoother targets with less GAN artifact. Cheap and easy.

2. **Temporal consistency post-processing**: Apply optical flow warping (RAFT) to SCUNet GAN outputs to enforce temporal consistency between frames. This borrows UAV's best idea without the diffusion overhead.

3. **SwinIR/HAT at native resolution**: These are same-resolution image restoration models (not upscalers). SwinIR-L or HAT-L could produce higher-quality per-frame denoising than SCUNet GAN. Still per-frame, but better quality ceiling.

4. **Recurrent temporal DRUNet** (already planned): The 9-channel input approach in `docs/research/temporal-context/prd.md` would give the student model temporal awareness directly, rather than trying to bake temporal consistency into the targets.

5. **Fine-tune a diffusion denoiser at native res**: Models like Stable Diffusion Image Variations or ControlNet inpainting operate at native resolution. Would require fine-tuning on HEVC artifact data, but avoids the upscaling mismatch.

## Files Reviewed

- `reference-code/Upscale-A-Video/README.md` -- Overview, usage examples
- `reference-code/Upscale-A-Video/inference_upscale_a_video.py` -- Full inference pipeline
- `reference-code/Upscale-A-Video/requirements.txt` -- Dependencies (ancient, incompatible)
- `reference-code/Upscale-A-Video/LICENSE` -- NTU S-Lab License 1.0 (non-commercial)
- `reference-code/Upscale-A-Video/configs/unet_video_config.json` -- UNet: 7ch input, [256,512,512,1024] channels
- `reference-code/Upscale-A-Video/configs/vae_3d_config.json` -- Basic VAE without conditioning
- `reference-code/Upscale-A-Video/configs/vae_video_config.json` -- Video VAE with LR conditioning + temporal blocks
- `reference-code/Upscale-A-Video/configs/CKPT_PTH.py` -- LLaVA 1.5 13B path
- `reference-code/Upscale-A-Video/models_video/pipeline_upscale_a_video.py` -- Pipeline with tiled denoising, propagation interleaving
- `reference-code/Upscale-A-Video/models_video/propagation_module.py` -- RAFT flow + deformable alignment propagation
- `reference-code/Upscale-A-Video/models_video/autoencoder_kl_cond_video.py` -- VAE with LR skip connection
- `reference-code/Upscale-A-Video/models_video/vae_video.py` -- 3D VAE decoder with SFT conditioning from LR input
- `reference-code/Upscale-A-Video/utils.py` -- Video I/O utilities
