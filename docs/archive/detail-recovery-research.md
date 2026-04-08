# Detail Recovery Research for Denoised Video

Research into recovering fine detail (hair, skin texture, fabric weave, film grain) in denoised/processed video without introducing AI-hallucinated details. Conducted 2026-04-01.

## Context

- **Pipeline:** Compressed 1080p BluRay (x265) -> SCUNet denoise -> NAFNet distilled model -> output
- **Problem:** SCUNet PSNR model over-smooths fine detail. NAFNet distilled student inherits this softness.
- **Goal:** Recover or preserve real detail faithfully. No GAN-style hallucination of fake texture.
- **Current training:** Charbonnier loss only, 5K/50K iters, 256px crops, 1032 pairs. VGG perceptual loss already implemented in `training/train_nafnet.py` but not yet used in a full training run.

## Summary Table

| Approach | Quality | Hallucination Risk | Effort | Pipeline Compat | Stackable | Recommended |
|---|---|---|---|---|---|---|
| **1. Focal Frequency Loss** | High | None | Low | Training-time | Yes | **YES** |
| **2. Detail transfer (high-pass blending)** | Medium-High | None | Low | Post-process | Yes | **YES** |
| **3. VGG perceptual loss (already built)** | High | Very low | Minimal | Training-time | Yes | **YES** |
| **4. Gram matrix / style loss** | Medium-High | Low | Low | Training-time | Yes | Worth trying |
| **5. Film grain synthesis (x265 SEI)** | Medium | None | Medium | Encode-time | Yes | Worth trying |
| **6. SCUNet GAN as teacher** | High | Medium | Medium | Training-time | No | Cautious |
| **7. PSNR+GAN teacher blending** | Medium-High | Low | Medium | Training-time | No | Worth trying |
| **8. ffmpeg unsharp post-process** | Low-Medium | None | Minimal | Post-process | Yes | Quick test |
| **9. Laplacian pyramid fusion** | Medium-High | None | Medium | Post-process | Yes | Worth trying |
| **10. Alternative denoiser (Restormer)** | Potentially higher | Low | High | Requires rearchitect | No | Not now |

**Top 3 recommendations:** Focal Frequency Loss + VGG perceptual loss (training-time, stack together), then detail transfer post-processing. These three can all be combined.

---

## 1. Focal Frequency Loss (FFL)

### What it is

A training loss that operates in the frequency domain (via FFT). Neural networks have a well-documented "spectral bias" -- they learn low frequencies first and struggle with high frequencies. FFL addresses this by adaptively up-weighting frequency components that the model finds hard to reconstruct, and down-weighting easy ones.

Published at ICCV 2021 by Jiang et al. Official PyTorch implementation: `pip install focal-frequency-loss`.

### Why it matters for us

Our NAFNet student is trained with Charbonnier loss, which treats all pixels equally. High-frequency detail (edges, texture, fine hair strands) contributes relatively little to the overall L1-like loss compared to smooth regions. FFL explicitly forces the model to get those high frequencies right.

### Hallucination risk: None

FFL is a regression loss -- it penalizes the difference between predicted and target frequency spectra. It cannot hallucinate detail that isn't in the target. It just makes the model try harder on the frequencies it's currently getting wrong (which are the high frequencies we're losing).

### Implementation

```python
from focal_frequency_loss import FocalFrequencyLoss as FFL

# In training loop:
ffl = FFL(loss_weight=1.0, alpha=1.0)  # alpha controls focus strength

# Combined loss:
loss_charb = charbonnier_loss(pred, target)
loss_ffl = ffl(pred, target)
loss_vgg = vgg_perceptual_loss(pred, target)  # already implemented
total_loss = 1.0 * loss_charb + 0.1 * loss_vgg + 1.0 * loss_ffl
```

Key parameters:
- `alpha`: Larger = more focus on hard frequencies. Start with 1.0.
- `loss_weight`: Weight relative to other losses. Start with 1.0 (same as Charbonnier).
- Requires `pip install focal-frequency-loss` or copy the ~50-line class from the repo.

### Effort: Low

The package is a single `nn.Module`. Add it to `training/train_nafnet.py` alongside the existing Charbonnier and VGG losses. Add a `--ffl-weight` argument. Takes 15 minutes.

### Log Focal Frequency Loss variant

A 2025 paper (arxiv 2601.20878) proposes Log FFL (LFFL) which uses log-space differences for more balanced reconstruction across all frequency bands. May be worth trying if standard FFL over-emphasizes very high frequencies (noise-adjacent).

---

## 2. Detail Transfer (High-Pass Blending from Original)

### What it is

A classic computational photography technique. The idea:
1. Compute the "detail layer" of the original noisy frame: `detail = original - blur(original)`
2. Compute the "detail layer" of the denoised frame: `detail_denoised = denoised - blur(denoised)`
3. The lost detail is approximately: `lost = detail - detail_denoised`
4. Add back a fraction: `output = denoised + weight * lost`

This transfers real high-frequency content from the original back into the denoised result. The key insight: the denoiser removed both noise AND some real texture. By comparing what high frequencies existed before vs after, we can estimate what real detail was lost.

### Why it matters for us

This is a zero-risk, zero-training post-processing step. It uses the actual original frame's texture, not generated content. The weight parameter controls the strength -- 0.0 = pure denoised, 1.0 = all original detail added back (including noise). Values of 0.2-0.5 typically give good results.

### Hallucination risk: None

This is purely additive from the real source image. The worst case is adding back some noise along with the detail, which can be mitigated with a threshold or edge-aware weighting.

### Implementation

```python
import torch
import torch.nn.functional as F

def detail_transfer(original, denoised, sigma=2.0, weight=0.3):
    """
    Transfer lost high-frequency detail from original to denoised.

    Args:
        original: Original noisy frame [B, C, H, W], float32 0-1
        denoised: Denoised frame [B, C, H, W], float32 0-1
        sigma: Gaussian blur sigma for base/detail decomposition
        weight: How much lost detail to add back (0.0 to 1.0)
    """
    kernel_size = int(sigma * 6) | 1  # ensure odd
    # Extract detail layers
    base_orig = gaussian_blur(original, kernel_size, sigma)
    base_denoised = gaussian_blur(denoised, kernel_size, sigma)
    detail_orig = original - base_orig
    detail_denoised = denoised - base_denoised
    # Lost detail = what was in original but not in denoised
    lost_detail = detail_orig - detail_denoised
    # Add back weighted lost detail
    return torch.clamp(denoised + weight * lost_detail, 0, 1)

def gaussian_blur(x, kernel_size, sigma):
    """Apply Gaussian blur per-channel."""
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= kernel_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g /= g.sum()
    # Separable convolution
    g_h = g.view(1, 1, 1, -1).expand(x.shape[1], -1, -1, -1)
    g_v = g.view(1, 1, -1, 1).expand(x.shape[1], -1, -1, -1)
    pad = kernel_size // 2
    x = F.conv2d(F.pad(x, [pad]*4, mode='reflect'), g_h, groups=x.shape[1])
    x = F.conv2d(F.pad(x, [pad]*4, mode='reflect'), g_v, groups=x.shape[1])
    return x
```

### Where it fits in the pipeline

This runs after NAFNet inference, before encoding. In `pipelines/denoise_nafnet.py`, after getting the denoised frame, apply `detail_transfer(original_frame, denoised_frame)`. Adds negligible compute (a few Gaussian blurs).

### Tuning

- `sigma`: Controls what counts as "detail." Smaller sigma = finer detail only. Start with 1.5-3.0.
- `weight`: Start at 0.2, visually compare. Higher = more texture but more noise.
- Can apply edge-aware masking: only transfer detail where the denoised image has edges (avoids adding noise in flat regions).

### Effort: Low

~20 lines of code added to the pipeline. No training required. Can be toggled on/off with a CLI flag.

---

## 3. VGG Perceptual Loss (Already Implemented)

### What it is

L1 distance between VGG19 feature maps of prediction and target. Already implemented in `training/train_nafnet.py` as `VGGPerceptualLoss`. Extracts features at 5 VGG layers (conv1_2, conv2_2, conv3_4, conv4_4, conv5_4) with weights `[0.1, 0.1, 1.0, 1.0, 1.0]`.

### Why it matters

Perceptual loss captures texture and structure similarity at multiple scales. Pure pixel loss (Charbonnier) treats a shifted texture as completely wrong; perceptual loss recognizes it as structurally similar. This encourages the model to preserve texture patterns rather than blur them out to minimize pixel error.

Research consistently shows LPIPS/VGG loss produces sharper results while maintaining fidelity. The key: it preserves existing textures, it does not generate new ones.

### Hallucination risk: Very low

Perceptual loss is still a regression loss between real images. It can occasionally cause slight texture shift (the model might produce a slightly different but perceptually similar texture), but this is far from GAN-level hallucination.

### Implementation

Already done. Just need to enable it during training:

```bash
# Resume from existing checkpoint with perceptual loss
PYTHONUTF8=1 .../python.exe -m modal run cloud/modal_train.py \
    --data-dir data/train_pairs \
    --resume checkpoints/nafnet_distill/nafnet_best.pth \
    --max-iters 20000 \
    --lr 5e-5 \
    --perceptual-weight 0.1 \
    --batch-size 8
```

Weight of 0.1 relative to Charbonnier 1.0 is typical. Too high and you get color shifts.

### Effort: Minimal

The code exists. Just add the `--perceptual-weight` CLI arg if not already wired up, and run training.

---

## 4. Gram Matrix / Style Loss

### What it is

While VGG perceptual loss compares feature activations directly (preserving spatial layout), Gram matrix loss compares the correlation between feature channels (preserving texture statistics regardless of spatial position). This is the loss used in neural style transfer.

`G = F^T * F` where F is the feature map reshaped to (C, H*W). The Gram matrix captures which features tend to co-activate, which encodes texture characteristics.

### Why it matters

Fine textures like fabric weave, skin pores, and film grain have characteristic statistical patterns that Gram loss can capture. Even if the exact pixel positions differ, the texture statistics should match. This complements pixel-level and perceptual losses.

### Hallucination risk: Low

Gram loss encourages matching texture statistics, not generating new texture. However, if the weight is too high, the model might try to "texture-ify" smooth regions to match statistics. Keep the weight modest (0.01-0.05).

### Implementation

```python
class GramLoss(nn.Module):
    """Gram matrix loss for texture statistics preservation."""
    def __init__(self, vgg_extractor):
        super().__init__()
        self.vgg = vgg_extractor  # reuse the existing VGGFeatureExtractor

    def gram_matrix(self, feat):
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W)
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, pred, target):
        pred_feats = self.vgg(pred)
        with torch.no_grad():
            target_feats = self.vgg(target)
        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss += F.l1_loss(self.gram_matrix(pf), self.gram_matrix(tf))
        return loss
```

### Effort: Low

~20 lines. Reuses the existing VGGFeatureExtractor. Add as another loss term.

---

## 5. Film Grain Synthesis

### What it is

Film grain (and to some extent, video compression noise) contributes to the perceived "detail" and "texture" of video. When you denoise, removing grain makes the image look artificially clean/digital. Film grain synthesis adds back realistic grain that matches the source material's characteristics.

### Approaches

**a) x265 film grain SEI metadata (--film-grain)**

x265 supports encoding film grain characteristics as SEI metadata. The decoder synthesizes grain at playback time. This is the standard approach in professional video encoding:

1. Analyze the source video's grain characteristics (per-frame intensity, frequency)
2. Encode the denoised video with grain parameters as metadata
3. The decoder re-synthesizes matching grain on playback

x265 flags: `--film-grain <grain_table_file>` and `--tune grain` (different -- tune grain just adjusts rate control for grainy content).

**b) Simple additive noise in post-processing**

```python
# Quick and dirty grain synthesis
def add_grain(frame, intensity=0.01):
    grain = torch.randn_like(frame) * intensity
    return torch.clamp(frame + grain, 0, 1)
```

This is too simple for production (grain should be spatially correlated, intensity-dependent, and match the source), but good for testing whether grain helps perception.

**c) Neural film grain (FGA-NN, 2025)**

FGA-NN is the first learning-based film grain analysis method that estimates conventional grain parameters compatible with standard video codec grain synthesis. It extracts grain characteristics from the source video and produces grain tables compatible with H.264/H.265/AV1 SEI messages.

**d) ffmpeg noise filter**

```bash
ffmpeg -i denoised.mkv -vf "noise=alls=5:allf=t+u" -c:v libx265 output.mkv
```

Adds uniform + temporal noise. Crude but quick for testing.

### Hallucination risk: None (if matching source characteristics)

Grain synthesis adds statistically matched random texture, not content-bearing detail. It's explicitly random noise -- the opposite of hallucination. It just makes the image look more "filmic" and less "digital."

### Effort: Medium

For the SEI metadata approach, need to build or find a grain analysis tool that produces x265-compatible grain tables. For simple additive noise, trivial. For FGA-NN, need to integrate the model.

### Recommendation

Start with simple additive Gaussian noise to test whether grain helps the perception of quality. If it does, invest in proper grain table generation for x265 SEI.

---

## 6. SCUNet GAN Model as Teacher

### What it is

SCUNet provides two pretrained models:
- `scunet_color_real_psnr` -- trained with L1/MSE loss, optimized for PSNR
- `scunet_color_real_gan` -- trained with L1 + VGG perceptual + GAN adversarial loss, optimized for perceptual quality

The GAN model produces sharper, more detailed output but may hallucinate fine texture that wasn't in the original.

### The tradeoff

GAN-trained models are well-known to produce plausible but fabricated high-frequency detail. For super-resolution this is often severe (generating fake texture at 4x scale). For denoising at native resolution, the hallucination is more subtle -- the GAN model might sharpen edges more aggressively or add slight texture patterns that look natural but weren't exactly in the source.

### Hallucination risk: Medium

At native resolution denoising, GAN hallucination is less severe than in super-resolution. The model is not inventing content at a new resolution -- it's choosing between plausible textures at the same resolution. But it IS still generating detail that may not exactly match the original uncompressed source.

For archival/faithful processing of a video library, this is a meaningful concern. The user explicitly said no GAN hallucination.

### When to consider it

If visual quality is more important than pixel-level fidelity for some content. Could be offered as a "sharper" mode alongside the faithful PSNR mode.

### Recommendation: Cautious

Don't use as the primary teacher. But worth generating a few test frames with the GAN model for visual comparison to understand the quality ceiling.

---

## 7. PSNR + GAN Teacher Blending

### What it is

Generate both PSNR and GAN teacher outputs, then use a weighted blend as the training target:

```python
target = alpha * scunet_psnr_output + (1 - alpha) * scunet_gan_output
# alpha = 0.7 gives mostly faithful with some GAN sharpness
```

### Why it might work

The PSNR model is too smooth. The GAN model may hallucinate. A blend could give the sharpness of the GAN model while being anchored by the PSNR model's fidelity. At alpha=0.7, the target is 70% faithful + 30% perceptually enhanced.

### Hallucination risk: Low (with high alpha)

At alpha >= 0.7, the hallucinated details are attenuated by 70%. Most GAN hallucinations are subtle high-frequency additions, so blending significantly reduces their amplitude while retaining the sharpness benefit.

### Effort: Medium

Requires running SCUNet GAN model on all training frames to generate a second set of targets. This means another ~25 minutes per batch of frames on the RTX 3060, or running on Modal. Then modify the dataset to load both targets and blend.

### Recommendation: Worth trying

Generate GAN outputs for a small subset (100 frames), visually compare PSNR vs GAN vs blended, then decide if full generation is warranted.

---

## 8. ffmpeg Unsharp Post-Processing

### What it is

Apply ffmpeg's built-in unsharp mask filter to the output video:

```bash
ffmpeg -i denoised.mkv -vf "unsharp=5:5:0.5:5:5:0.3" -c:v libx265 output.mkv
```

Parameters: `luma_x:luma_y:luma_strength:chroma_x:chroma_y:chroma_strength`

### Why it's limited

Unsharp masking is a global sharpening operation. It amplifies ALL edges, including compression artifacts, noise remnants, and encoding artifacts. It has no understanding of what's "real detail" vs what's "artifact." After denoising, the remaining edges are relatively clean, so it works better than on raw noisy input, but it still risks:
- Halo artifacts around strong edges (bright/dark ringing)
- Amplifying any residual noise in shadows
- Over-sharpening already-sharp edges

### Hallucination risk: None

Unsharp masking only amplifies what's already there. No content generation.

### Best use case

As a very light final touch. Strength 0.3-0.5 for luma, 0.2-0.3 for chroma. Useful for quick A/B testing of "does sharpening help at all?" before investing in learned approaches.

### Effort: Minimal

One ffmpeg flag. Can be added to the existing encoding command in `cloud/modal_denoise.py`.

---

## 9. Laplacian Pyramid Fusion

### What it is

Decompose both the original and denoised frames into Laplacian pyramids (multi-scale detail decomposition), then selectively blend levels:
- Low-frequency levels (large structures, lighting): take from **denoised** (clean)
- High-frequency levels (fine detail, texture): blend **original** and **denoised**

This is a more sophisticated version of approach #2 (detail transfer) that operates at multiple scales independently.

### Implementation sketch

```python
def laplacian_pyramid(img, levels=4):
    """Decompose image into Laplacian pyramid."""
    pyramid = []
    current = img
    for i in range(levels - 1):
        down = F.interpolate(current, scale_factor=0.5, mode='bilinear',
                             align_corners=False)
        up = F.interpolate(down, size=current.shape[2:], mode='bilinear',
                           align_corners=False)
        pyramid.append(current - up)  # detail at this scale
        current = down
    pyramid.append(current)  # coarsest level (base)
    return pyramid

def fuse_pyramids(pyr_orig, pyr_denoised, weights):
    """
    Blend pyramids. weights[i] = how much original detail to keep at level i.
    weights = [0.3, 0.2, 0.1, 0.0] -- more original at fine scales, none at coarse
    """
    fused = []
    for i, (lo, ld) in enumerate(zip(pyr_orig, pyr_denoised)):
        w = weights[i] if i < len(weights) else 0.0
        fused.append(w * lo + (1 - w) * ld)
    return fused

def reconstruct_from_pyramid(pyramid):
    """Reconstruct image from Laplacian pyramid."""
    img = pyramid[-1]
    for detail in reversed(pyramid[:-1]):
        img = F.interpolate(img, size=detail.shape[2:], mode='bilinear',
                            align_corners=False) + detail
    return img
```

### Hallucination risk: None

Same as detail transfer -- only uses real content from the original frame.

### Advantage over simple detail transfer

Can independently control how much original texture to blend at each scale. Fine grain (level 0) might get weight 0.3, while medium texture (level 1) gets 0.2, and large structures (level 2+) get 0.0 (fully denoised). This avoids adding back low-frequency noise while preserving fine texture.

### Effort: Medium

~50 lines of code. Needs tuning of per-level weights, which may vary by content.

---

## 10. Alternative Denoiser (Restormer, SwinIR)

### What it is

Restormer (CVPR 2022) and SwinIR (ICCV 2021) are transformer-based image restoration architectures that claim better texture preservation than CNN-based denoisers.

- **Restormer** uses multi-Dconv head transposed attention for efficient global context
- **SwinIR** uses Swin Transformer blocks (similar to SCUNet's backbone)

Both are available pretrained on various degradation tasks.

### Why not recommended right now

- SCUNet already uses Swin Transformer blocks -- SwinIR would be architecturally similar
- Restormer is even slower than SCUNet (more attention layers)
- The problem isn't the denoiser architecture per se -- it's the training loss (PSNR-optimized = smooth)
- Switching architecture means redoing all the distillation pipeline work
- Any PSNR-trained model will exhibit the same softness

### When to consider

If training-time fixes (FFL + VGG) don't sufficiently improve detail, then a different architecture WITH perceptual training might help. But fix the losses first.

---

## Stacking Recommendations

These approaches are not mutually exclusive. Here's the recommended stack:

### Training-time (modify `training/train_nafnet.py`)
```
total_loss = 1.0 * charbonnier + 0.1 * vgg_perceptual + 1.0 * focal_frequency
```

Optionally add gram loss (0.02 weight) for texture statistics.

### Post-processing (modify `pipelines/denoise_nafnet.py`)
```
denoised = nafnet(original_frame)
sharpened = detail_transfer(original_frame, denoised, sigma=2.0, weight=0.25)
```

### Encoding (modify ffmpeg command in `cloud/modal_denoise.py`)
```
# Light unsharp as final polish (optional, test first)
-vf "unsharp=5:5:0.3:5:5:0.0"
# OR: film grain synthesis via x265 SEI metadata
-x265-params "film-grain=grain_table.txt"
```

### Phased rollout

1. **Phase 1** (quick wins, 1 day): Resume training with VGG perceptual loss + FFL. Run for 10-20K iters. Compare output sharpness.
2. **Phase 2** (if Phase 1 insufficient, 1 day): Add detail transfer post-processing to the pipeline. Tune sigma and weight visually.
3. **Phase 3** (refinement, 1 day): Try gram loss, test film grain synthesis, try Laplacian pyramid fusion for more control.
4. **Phase 4** (optional): Generate SCUNet GAN teacher outputs for a small test set to evaluate PSNR+GAN blending.

---

## Specific Recommendations for Our Pipeline

### Priority 1: Fix the training loss

The single highest-impact change is adding FFL + VGG perceptual loss to the NAFNet distillation training. This addresses the root cause (spectral bias + pixel-only loss) rather than papering over it with post-processing.

Concrete plan:
1. Add `focal-frequency-loss` to Modal image dependencies
2. Add `--ffl-weight` argument to `training/train_nafnet.py`
3. Resume from existing best checkpoint with combined loss: Charbonnier(1.0) + VGG(0.1) + FFL(1.0)
4. Train 15-20K iters at lr=5e-5 on A10G
5. Compare output visually and via LPIPS metric (not just PSNR -- PSNR penalizes sharpness)

### Priority 2: Detail transfer post-processing

Even with better training, some detail loss is inevitable with any pixel-regression model. Detail transfer from the original is a no-risk way to recover it. This is especially useful for content where the original has fine film grain or fabric texture that the model removes.

Add as an optional `--detail-transfer 0.25` flag to the pipeline. Costs ~2% extra compute.

### Priority 3: Evaluate with LPIPS, not just PSNR

PSNR penalizes sharp results (a sharp edge slightly misaligned scores worse than a blurred edge). Add LPIPS evaluation to the training validation and to `bench/compare.py`. This ensures we're measuring perceptual quality, not just pixel fidelity.

---

## Sources

- [SCUNet paper (arxiv 2203.13278)](https://arxiv.org/abs/2203.13278) -- PSNR vs GAN model variants
- [SCUNet GitHub](https://github.com/cszn/SCUNet) -- pretrained model downloads
- [Focal Frequency Loss (ICCV 2021)](https://github.com/EndlessSora/focal-frequency-loss) -- official implementation, pip installable
- [Log Focal Frequency Loss (arxiv 2601.20878)](https://arxiv.org/abs/2601.20878) -- improved variant for balanced frequency reconstruction
- [IFSR-Net: Implicit Frequency Selection (2025)](https://www.tandfonline.com/doi/full/10.1080/09540091.2025.2465448) -- CNN with implicit frequency selection
- [Unsharp Mask Guided Filtering (arxiv 2106.01428)](https://arxiv.org/abs/2106.01428) -- combines unsharp masking with guided filtering
- [Adaptive Guided Image Filtering for Sharpness Enhancement](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2013.0563) -- edge-aware sharpening without halo artifacts
- [Edge-preserving decompositions for multi-scale detail manipulation (ACM TOG)](https://dl.acm.org/doi/10.1145/1360612.1360666) -- Laplacian pyramid detail manipulation
- [Guided Image Filtering (Kaiming He, PAMI 2012)](https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf) -- edge-preserving filtering with structure transfer
- [LPIPS metric (github.com/richzhang)](https://github.com/richzhang/PerceptualSimilarity) -- learned perceptual image patch similarity
- [FGA-NN: Film Grain Analysis Neural Network (2025)](https://arxiv.org/html/2506.14350v1) -- learning-based grain parameter estimation
- [Neural Film Grain Rendering (2025)](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70076) -- neural approach to grain synthesis
- [Film Grain Synthesis for AV1 (Norkin 2018)](https://norkin.org/pdf/DCC_2018_AV1_film_grain.pdf) -- codec-level grain synthesis
- [x265 film grain SEI documentation](https://x265.readthedocs.io/en/master/releasenotes.html) -- --film-grain parameter
- [Knowledge Distillation for Image Restoration (arxiv 2501.09268)](https://arxiv.org/abs/2501.09268) -- texture-preserving distillation with PIQE-based feature extraction
- [Detail-aware image denoising via structure preserved network (2024)](https://link.springer.com/article/10.1007/s00371-024-03353-y) -- structure-preserving denoising with diffusion refinement
- [VGG Loss explained](https://paperswithcode.com/method/vgg-loss) -- perceptual loss overview
- [Recovering Texture with LMMSE Filter (MDPI 2021)](https://www.mdpi.com/2624-6120/2/2/19) -- detail recovery from denoised images
- [CVEGAN: Compressed Video Enhancement GAN (2024)](https://www.sciencedirect.com/science/article/pii/S0923596524000286) -- perceptual GAN for video compression artifact removal
- [ffmpeg unsharp filter docs](https://ffmpeg.org/ffmpeg-filters.html) -- unsharp mask parameters
