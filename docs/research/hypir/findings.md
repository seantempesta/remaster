# HYPIR: Evaluation Findings

**Date:** 2026-04-14
**Status:** Concluded -- produces AI-look artifacts, not suitable for training targets
**Cost spent:** $0 (local GPU only)

## What Is HYPIR

HYPIR (Harnessing Diffusion-Yielded Score Priors for Image Restoration) is a single-forward-pass diffusion restoration model. Unlike iterative diffusion (30+ steps), HYPIR runs one UNet pass through Stable Diffusion 2.1 with LoRA weights, using a configurable timestep `t` to control how much the diffusion prior influences the output. Lower `t` = more faithful to input, higher `t` = more hallucination.

Key appeal: single forward pass means it's much faster than full diffusion SR models like Upscale-A-Video.

## What We Tested

10 diverse frames (2 per show: Foundation, Dune 2, Squid Game, The Expanse, Firefly, One Piece) across multiple configurations:

### Configurations tested

| Config | Scale | Timestep | Input | Prompt |
|--------|-------|----------|-------|--------|
| 2x t=25 | 2x (4K) | 25 | Original HEVC | Empty |
| 2x t=50 | 2x (4K) | 50 | Original HEVC | Empty |
| 2x t=100 | 2x (4K) | 100 | Original HEVC | Empty |
| 2x t=200 | 2x (4K) | 200 | Original HEVC | Empty |
| 1x t=25 on SCUNet | 1x | 25 | SCUNet + USM target | Empty |
| 2x t=25 on SCUNet | 2x (4K) | 25 | SCUNet + USM target | Empty |
| 2x t=25 (35mm prompt) | 2x (4K) | 25 | Original HEVC | "35mm film photograph..." |

All outputs received wavelet color correction (preserve original color/lighting) and Lanczos downscale to 1080p for comparison.

### Prompt experiment

Tested `"35mm film photograph, natural skin texture, fine detail, realistic"` vs empty prompt (`""`). The 35mm prompt biased textures toward photographic grain regardless of content (bad for anime, CGI). Empty prompt was more neutral. HYPIR's text conditioning is weak (single forward pass, no CFG), so prompt impact is subtle but noticeable.

**Conclusion:** Empty prompt is correct for mixed content. Per-image GPT captioning would be ideal but unnecessary given the other findings below.

## Key Findings

### 1. Higher timesteps produce obvious AI hallucination

- **t=200:** Completely rewrites textures. Faces look AI-generated. Backgrounds gain fake detail that wasn't in the source. Immediately recognizable as AI output.
- **t=100:** Still significant hallucination. Skin becomes unnaturally smooth, fine structures are invented.
- **t=50:** Moderate hallucination. Better than t=100 but still adds fake texture.
- **t=25:** Most faithful to input. Minimal hallucination but still alters the image character.

### 2. Even t=25 warps facial features

The diffusion prior, even at the lowest tested timestep, distorts human faces enough to give them an "AI-generated" quality. This is subtle but consistent across all shows and particularly visible in close-ups. The model's SD2.1 backbone was trained on web images with heavy face representation, so it has strong priors about what faces "should" look like.

### 3. 1x t=25 on SCUNet was the best config

Processing already-clean SCUNet targets at 1x scale with t=25 produced the most natural-looking results. The model added subtle detail enhancement without the upscale artifacts. However, on closer inspection, it still warped faces and added an AI quality to skin/hair textures that would propagate into the DRUNet student during distillation.

### 4. Upscale-then-downscale doesn't launder the artifacts

The plan was: HYPIR 2x to 4K, color correct, Lanczos back to 1080p. The theory was that generating detail at 4K and downscaling would produce sharper 1080p targets. In practice, the AI-generated textures survive the downscale and are clearly visible in side-by-side comparisons.

### 5. Speed is acceptable but irrelevant

At 2x on RTX 3060 (patch_size=256, stride=128): ~1.5 min/frame. Processing 7K training targets would take ~175 hours locally but could be parallelized on Modal. This is much faster than Upscale-A-Video (which was eliminated for speed). However, quality issues make speed moot.

## Why HYPIR Is Wrong for Training Targets

The fundamental problem: **any diffusion-based enhancement adds the model's learned prior to the output, and that prior includes "what the internet thinks images should look like."** For training targets that a student network will learn to replicate, this means:

1. The student learns to reproduce AI-generated textures, not natural ones
2. Faces trained on HYPIR targets will have subtle but systematic distortion
3. The "AI look" compounds through distillation -- teacher -> student amplifies artifacts

This is different from SCUNet GAN, which uses adversarial training to denoise while preserving natural texture statistics. SCUNet's GAN discriminator specifically penalizes outputs that don't match real image distributions.

## Comparison with Current Approach

| Metric | SCUNet GAN + USM (current) | HYPIR 1x t=25 (best config) |
|--------|---------------------------|------------------------------|
| Denoising | Excellent | Good (not its purpose) |
| Sharpening | Good (via USM) | Moderate |
| Face fidelity | Preserves natural look | Subtle AI distortion |
| Texture naturalness | Natural (GAN-enforced) | AI-biased (diffusion prior) |
| Speed (7K frames) | ~1 hour on Modal | ~175 hours local / ~35 hours Modal |
| Training target safety | Proven (53 dB teacher) | Risk of AI artifact propagation |

**Verdict:** SCUNet GAN + USM remains the best training target pipeline. The DRUNet teacher trained on these targets achieves 53.27 dB PSNR and 107% sharpness -- there's no evidence HYPIR targets would improve on this.

## Files

- `tools/hypir_grid_test.py` -- Test script (runs HYPIR configs, color correction, comparison grids)
- `data/hypir-test/grid/` -- All test outputs (originals, SCUNet, HYPIR variants, comparison grids)
- `data/hypir-test/grid/compare/` -- Side-by-side grids and detail crops (10 frames x 2 each)
- `reference-code/HYPIR/` -- HYPIR source code and weights
