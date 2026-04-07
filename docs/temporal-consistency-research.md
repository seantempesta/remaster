# Temporal Consistency Research

## Problem

Our DRUNet teacher/student processes each frame independently. Different noise patterns in adjacent frames produce slightly different outputs, causing visible flicker. We need temporally consistent output without sacrificing real-time speed (30+ fps on RTX 3060).

## Key Architectures to Study

### FastDVDNet (CVPR 2020) — Primary Reference

**The breakthrough:** Eliminates optical flow entirely. Concatenates 5 raw frames and lets the U-Net learn implicit motion alignment.

- **Architecture:** Two-stage cascaded U-Net. Stage 1 processes frame triplets individually, Stage 2 fuses the outputs.
- **Parameters:** 2.49M (similar to our student at 1.06M)
- **Speed:** 24ms/frame at 480p, ~90-100ms at 1080p on RTX GPUs
- **Quality:** 31.86 dB PSNR on Set8 (competitive with flow-based methods)
- **Key insight:** Implicit motion handling through concatenation is nearly as good as explicit flow, at 3-5x faster
- **GitHub:** https://github.com/m-tassano/fastdvdnet (PyTorch, pretrained weights included)

### LiteDVDNet (2024) — Speed Target

Compressed FastDVDNet using depthwise separable convolutions and channel pruning.

- **LiteDVDNet-32:** 0.8M params, 3x faster, only -0.18 dB PSNR
- **LiteDVDNet-16:** 0.64M params, 5x faster (100 fps at 480p possible), -0.61 dB
- **At 1080p:** LiteDVDNet-16 estimated ~35-40ms/frame (25-28 fps)
- This is very close to our deployment target

### PocketDVDNet (2025) — Best Quality/Size Ratio

- **0.6M params** but BETTER quality than FastDVDNet (34.86 vs 33.0 dB)
- Uses physics-informed 5-component noise model + structured pruning + knowledge distillation
- 17ms/frame at 480p, ~60ms at 1080p
- **GitHub:** https://arxiv.org/abs/2601.16780

### TAP — Temporal As Plugin (ECCV 2024)

- Bolts temporal modules onto ANY frozen image denoiser (DRUNet, NAFNet, etc.)
- No paired training data needed (unsupervised)
- Uses 3-5 frames, optional optical flow
- **GitHub:** https://github.com/zfu006/TAP
- **Most relevant to us:** could add temporal consistency to our existing DRUNet without retraining from scratch

### GRTN — Gated Recurrent Transformer (Sep 2024)

- SOTA quality with only single-frame delay (vs 16-frame for competitors)
- Gated recurrence: hidden state carries info across frames
- Uses Swin Transformer blocks
- ~5-7M params, real-time capable
- Good for streaming/playback scenarios

## Integration Strategies for Our Pipeline

### Option A: Direct Frame Concatenation (Simplest)

Modify DRUNet input layer to accept N concatenated frames:
```
Input: (B, 3*N, H, W) with N=5 sequential frames
Add 1x1 conv: 15 channels -> 64 channels (replacing current 3->64)
Rest of DRUNet unchanged
```

- **Pros:** Minimal code change, existing weights mostly transfer
- **Cons:** 5x memory, loses explicit temporal structure
- **Speed impact:** ~1.5x slower (wider first layer, more input data)

### Option B: Cross-Attention at Bottleneck (Our Preferred Approach)

Process each frame through encoder independently, exchange information at bottleneck via cross-attention, decode independently:
```
encoder(frame_i) for each frame  ->  cross_attention(all bottlenecks)  ->  decoder(frame_i)
```

At bottleneck (1/8 resolution = 240x135 for 1080p):
- Cross-attention across N frames is cheap (~2ms for N=8)
- Each frame "borrows" consistent features from neighbors
- Noise averages out in feature space, real detail persists

- **Pros:** Efficient, leverages existing encoder/decoder weights, explicit temporal exchange
- **Cons:** New module to design and train, TensorRT export needs care
- **Speed impact:** ~10-20ms overhead for 8-frame batch

### Option C: Temporal Loss During Training (Zero Inference Cost)

Add RAFT flow-warping consistency loss during training only:
```
L_temporal = |warp(output_t, flow_t->t+1) - output_t+1| * occlusion_mask
```

- **Pros:** Zero runtime cost, trains a naturally consistent single-frame model
- **Cons:** Requires sequential training data, RAFT during training (~5fps overhead)
- **We have RAFT** in reference-code/RAFT/
- Weight: 0.1-0.5x pixel loss

### Option D: VapourSynth Post-Filter (Quick Stopgap)

Add TTempSmooth or FluxSmooth to remaster/encode.vpy:
```python
clip = core.ttsmooth.TTempSmooth(clip, maxr=3, thresh=[4,5,5])
```

- **Pros:** 30 minutes to implement, immediate improvement
- **Cons:** Loses some detail in motion areas, doesn't fix root cause
- **Speed impact:** ~5-10% slower

### Option E: Multi-Scale FFT Temporal Fusion (Preferred Architecture v2)

Hybrid approach: spatial convolutions for local features, frequency-domain attention for temporal fusion at EVERY U-Net scale (not just bottleneck). Inspired by LaMa's FFC but applied temporally.

```
Encoder level 1 (1920x1080, 16ch):
  spatial conv -> rfft2 -> temporal freq attention -> irfft2 -> downsample

Encoder level 2 (960x540, 32ch):
  spatial conv -> rfft2 -> temporal freq attention -> irfft2 -> downsample

Encoder level 3 (480x270, 64ch):
  spatial conv -> rfft2 -> temporal freq attention -> irfft2 -> downsample

Bottleneck (240x135, 128ch):
  spatial conv -> rfft2 -> temporal freq attention -> irfft2

Decoder mirrors with skip connections
```

- **Pros:** Multi-scale temporal fusion — fine detail at high res, global consistency at low res. Existing encoder/decoder weights transfer. ~3ms total FFT overhead.
- **Cons:** More complex than bottleneck-only. Needs sequential training data.
- **Speed:** 80fps -> ~55fps (still 2x real-time)
- **Key advantage over Option B:** captures temporal correlations at every frequency scale, not just the most compressed representation

## Recommended Implementation Order

1. **Now:** Option D (VapourSynth temporal filter) — instant improvement, no training
2. **Next training run:** Option C (temporal loss) — requires sequential frame pairs, but zero inference cost
3. **Architecture v2:** Option E (multi-scale FFT temporal fusion) — the proper solution
4. **Fallback:** Option B (spatial cross-attention at bottleneck) if FFT approach doesn't pan out
5. **Research:** Study FastDVDNet, TAP, and LaMa/FFC codebases for implementation patterns

## Training Data Changes Needed

For temporal approaches (B and C), we need **sequential frame pairs** instead of random frames:
- Extract frames in order from each episode (every Nth frame, keeping adjacency)
- Store as ordered sequences, not shuffled pairs
- Modify dataset to return (frame_t, frame_t+1) or (frame_t-2, frame_t-1, frame_t, frame_t+1, frame_t+2)
- Validation should also use sequential clips for temporal metrics (tOF, tLP)

## Speed Comparison at 1080p (RTX 3060 estimates)

| Model | Params | 1080p ms/frame | FPS | Notes |
|-------|--------|----------------|-----|-------|
| Our DRUNet student | 1.06M | ~12ms | 80+ | Single frame, no temporal |
| Our DRUNet teacher | 32.6M | ~200ms | 5 | Single frame |
| FastDVDNet | 2.49M | ~100ms | 10 | 5 frames, implicit motion |
| LiteDVDNet-16 | 0.64M | ~35ms | 28 | 5 frames, depthwise convs |
| PocketDVDNet | 0.6M | ~60ms | 17 | 5 frames, best quality/size |
| DRUNet + cross-attn | ~1.5M | ~25ms | 40 | 8-frame batch, bottleneck attn |
| DRUNet + temporal loss | 1.06M | ~12ms | 80+ | Same model, trained differently |

## Novel Idea: Cross-Frame FFT Attention at Bottleneck

**Status: Confirmed novel** — no prior work combines (1) cross-frame attention (2) in frequency domain (3) at a U-Net bottleneck. This is a learned, differentiable version of VBM4D's core insight.

### Why FFT for Temporal Denoising

- Noise is flat-spectrum (white/weakly colored) — equal energy at all frequencies
- Signal has structured spectrum — concentrated, consistent patterns
- Across N frames: signal frequency bins are consistent, noise bins are random
- Attention in frequency domain naturally weights consistent bins (signal) and suppresses varying bins (noise)
- This is effectively **learned temporal Wiener filtering**

### Architecture

```
Frame 1 -> Encoder -> Bottleneck_1 -> rfft2 --┐
Frame 2 -> Encoder -> Bottleneck_2 -> rfft2 --┤
Frame 3 -> Encoder -> Bottleneck_3 -> rfft2 --┼-> Freq Attention -> ifft2 -> Decoder -> Output
Frame 4 -> Encoder -> Bottleneck_4 -> rfft2 --┤
Frame 5 -> Encoder -> Bottleneck_5 -> rfft2 --┘
```

### Implementation Details

At 1080p bottleneck (240x135, 128ch student):
- rfft2 produces 240x68 complex tensors per frame (~microseconds on GPU)
- 5 frames x 128ch x 240x68 x 2 (complex) x 2 bytes = ~16MB (trivial)
- Cross-frame attention: O(T^2 * F) where T=5, F=frequency bins — negligible
- torch.fft supports autograd natively, no custom CUDA needed
- Pad to 256x128 for power-of-2 FFT performance

### Prototype Strategy

Start simple, add complexity:
1. **Baseline:** magnitude-weighted cross-frame averaging (no learnable params)
   - Average magnitudes across frames, keep reference phase
   - Compare to spatial-domain frame averaging
2. **Learned gating:** element-wise learnable weights per frequency bin
   - Similar to SPECTRE's adaptive spectral gating
3. **Full attention:** complex-valued QKV attention across frames
   - Compute attention weights from magnitudes, apply to complex values

### Estimated Speed Impact

- FFT at bottleneck: ~0.1ms (240x135 is tiny)
- Cross-frame attention: ~0.5ms for 5 frames
- Total overhead: ~1-2ms per frame group
- Student at 80fps -> ~60fps with FFT attention (still 2x real-time)

### Key Insight for Training

The cross-attention doesn't need to produce better detail than the teacher — it just needs to produce the SAME detail consistently across frames. Having multi-frame input reduces prediction variance (noise averages out in feature space), even if the loss target is single-frame quality.

### Prior Art (Related but Distinct)

- **FNet (Google 2021):** FFT replaces attention within single frame, 80% faster training
- **GFNet:** Learnable frequency filters for token mixing
- **FFTformer (CVPR 2023):** Frequency-domain element-wise products, O(N) attention
- **SPECTRE (Feb 2025):** Adaptive spectral gating, 7x faster than FlashAttention-2
- **FADNet:** Inter-channel frequency attention for single-frame denoising
- **VBM4D:** Classical non-learned transform-domain temporal denoising (closest analog)

All of these operate within single frames. None combine cross-frame temporal attention with frequency-domain processing.

## References

- FastDVDNet: https://github.com/m-tassano/fastdvdnet
- DVDNet: https://github.com/m-tassano/dvdnet
- TAP (Temporal As Plugin): https://github.com/zfu006/TAP
- PocketDVDNet: https://arxiv.org/abs/2601.16780
- GRTN: https://arxiv.org/abs/2409.06603
- RAFT (optical flow): reference-code/RAFT/
- FNet: https://arxiv.org/abs/2105.03824
- GFNet: https://openreview.net/pdf?id=K_Mnsw5VoOW
- FFTformer: https://arxiv.org/abs/2211.12250
- SPECTRE: https://arxiv.org/abs/2502.18394
- FADNet: https://ieeexplore.ieee.org/document/10541898/
