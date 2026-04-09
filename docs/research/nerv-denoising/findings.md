# NeRV Denoising Research Findings

Summary of research conducted 2026-04-09.

## Key Insight

NeRV (Neural Representations for Videos) encodes an entire video as a neural network. The network's spectral bias causes it to learn clean signal before noise. With controlled capacity and regularization, this produces denoised output without any clean training targets.

## NeRV Family Overview

| Model | Year | Key Innovation | License | Code Available |
|-------|------|---------------|---------|----------------|
| NeRV | 2021 | Frame index -> RGB via decoder | MIT | Yes |
| HNeRV | 2023 | Content-adaptive ConvNeXt encoder | MIT | Yes |
| E-NeRV | 2022 | Enhanced temporal encoding | Unspecified | Yes |
| FFNeRV | 2023 | Flow-guided temporal grids | MIT | Yes |
| Boosting-NeRV | 2024 | SFT conditional decoder, HF loss | Apache 2.0 | Yes |
| ActINR | 2025 | Per-frame bias for denoising (+3-5 dB) | Unspecified | Minimal |
| MetaNeRV | 2025 | Meta-learned init for unseen videos | N/A | No code |
| PNVC | 2025 | GOP-based encoding for long videos | MIT | Yes (early) |
| LRConv-NeRV | 2026 | Low-rank separable convolutions | N/A | No code |
| Rabbit NeRV | 2026 | Disentangled component library | N/A | Cannot find |

## Decoding Speed (1080p)

| Model | FPS | GPU |
|-------|-----|-----|
| NeRV | 115 | A100 |
| FFNeRV | 84 | A100 |
| NeRV-Boost | ~44 | V100 |
| HNeRV | ~24 | V100 |
| HNeRV-Boost | ~13 | V100 |

Estimated H100 performance: 50-200+ fps depending on variant.

## Denoising-Specific Results from Literature

1. **NeRV original paper** (Section 4.4): 34.12 dB on noisy Honeybee (salt-and-pepper noise), beating median filtering and Deep Image Prior.
2. **ActINR** (CVPR 2025): 3-5 dB improvement over HNeRV-Boost for photon/readout noise denoising.
3. **Zero-shot blind denoising via INR**: L2 penalty on deeper layers maximizes denoising by suppressing high-frequency noise fitting.
4. **Deep Video Prior**: Networks learn temporal consistency before flickering artifacts. Early stopping = free denoising.

## Architecture Details (from code review)

### HNeRV (model_hnerv.py)
- ConvNeXt encoder downsamples input frame to small spatial embedding
- Decoder: stack of NeRVBlock (Conv + PixelShuffle upsample + Norm + Activation)
- SFT (Spatial Feature Transform) blocks modulate features based on temporal embedding
- Head: 3x3 conv to 3ch RGB, output via tanh -> [0,1]
- Supports temporal interpolation: `img_embed = 0.5*(encoder(pre_img) + encoder(post_img))`

### FFNeRV (model.py)
- Multi-resolution temporal grids (ParameterList of learned grid features)
- Frame index interpolates between grid positions (bilinear in time)
- Multi-scale head with flow-based aggregation across temporal neighbors
- Built-in weight quantization support

### Key Building Blocks (model_blocks.py)
- NeRVBlock: UpConv/DownConv + Norm + Activation
- UpConv: Conv2d + PixelShuffle (preferred) or ConvTranspose2d
- DownConv: PixelUnshuffle + Conv2d or strided Conv2d
- SFTLayer: Scale-and-shift modulation from conditioning signal
- ResBlock_SFT: Residual block with SFT conditioning
- ConvNeXt encoder: V1 blocks with LayerNorm, 7x7 depthwise conv

## HEVC Artifact Considerations

HEVC compression artifacts differ from Gaussian noise:
- Block boundaries (8x8, 16x16) are spatially structured
- Banding in smooth gradients
- Ringing around edges
- Some artifacts persist across frames (P/B-frames referencing same I-frame)

NeRV's spectral bias should handle truly random per-frame noise, but persistent artifacts across frame groups may get memorized. This is the primary uncertainty and must be validated in Phase 1 of the PRD.

## ConvNeXt-V2 Autoencoder (Explored, Parallel Track)

Built and tested at `lib/convnext_autoencoder.py`:
- 5.93M params (Atto variant)
- Pretrained ImageNet-1K encoder (136/140 keys loaded)
- 1080p FP16 inference: 79.6ms / 12.6 fps on RTX 3060
- Supports masked training (FCMAE-style) and temporal conditioning
- Could serve as the HNeRV encoder replacement in Phase 3.2

INT8 analysis found ConvNeXt-V2 introduces new quantization challenges (depthwise conv, LayerNorm, GRN) -- not recommended for the fast inference model. However, for the slow target generator, FP16 performance is sufficient.

## Recommendations

1. **Primary path:** HNeRV with denoising-oriented training (spectral bias + regularization + HF loss)
2. **Key experiment:** Phase 0-1 of PRD validates whether HEVC artifacts are actually denoised
3. **Scaling strategy:** GOP segmentation (300-600 frames per model) with scene-cut-aware boundaries
4. **Fallback:** If NeRV doesn't denoise HEVC well, pivot to temporal DRUNet teacher (see `docs/research/temporal-context/prd.md`)
