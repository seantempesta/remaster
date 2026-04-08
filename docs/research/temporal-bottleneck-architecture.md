# Temporal Bottleneck Cross-Attention Architecture

## Core Idea

The DRUNet student is a U-Net. At the bottleneck (135x240x128 for 1080p), the representation is maximally compressed — it's the "plan" for what the output frame should look like. The decoder expands this plan back to full resolution.

**What if we enrich this plan with information from neighboring frames?**

The bottleneck doesn't need to see raw pixels from other frames. It needs to know what's *in* those frames semantically and where things moved. A pretrained vision foundation model (self-supervised, trained on millions of images) already encodes this — "this patch is a face," "this patch is fabric," "this moved 2 positions right since last frame."

## Architecture v1: External Embedding Injection

```
Frame N ──> DRUNet Encoder ──> Bottleneck (135x240x128)
                                    │
                                    ▼
                          Cross-Attention (small, ~200K params)
                          Q = bottleneck features
                          K,V = vision model embeddings from frame N
                                    │
                                    ▼
                          Enriched Bottleneck (135x240x128)
                                    │
                          DRUNet Decoder ──> Clean Frame N
```

Start with a single frame's embedding (same frame, no temporal component yet). This tests whether semantic conditioning improves the bottleneck at all. If the model learns to activate different "filters" for faces vs walls vs text, that alone is a win.

**Vision model requirements:**
- Open-source, permissive license (MIT, Apache 2.0, BSD)
- Pretrained on millions of diverse images
- Produces dense spatial features (not just a single vector)
- Feature resolution close to our bottleneck (~60-140 spatial positions for 1080p)
- Small/fast enough to run alongside inference (ideally ConvNet, not giant ViT)
- Noise-robust features (trained with augmentation/self-supervised)

**NOT DINOv3** — Meta's license requires permission we haven't received. Need an alternative with similar feature quality.

## Architecture v2: Temporal Embedding Window

```
Cached embeddings: [E(N-4), E(N-3), E(N-2), E(N-1), E(N), E(N+1), E(N+2), E(N+3), E(N+4)]
                                                         │
Frame N ──> DRUNet Encoder ──> Bottleneck ──> Cross-Attention ──> Decoder ──> Output
                                              Q = bottleneck
                                              K,V = concat(all cached embeddings)
```

With +/- 4 frames of cached embeddings, the cross-attention sees:
- **What moved where** — same semantic content at shifted positions across frames
- **Multiple noise observations** — same content with different noise in each frame
- **Occlusion/reveal** — content visible in neighbor frames but hidden in center frame

Embeddings are ~10MB each, cached sequentially. Total cache for 9 frames: ~90MB. The attention computation is small — 135x240 queries attending over 9 * 68x120 keys.

## Architecture v3: Bottleneck Exchange (Batch Cross-Attention)

This is potentially the most elegant and efficient approach.

**Instead of external embeddings, exchange information between bottlenecks of adjacent frames processed in the same batch.**

```
Frame N-1 ──> Encoder ──> Bottleneck(N-1) ──┐
Frame N   ──> Encoder ──> Bottleneck(N)   ──┼── Cross-Attention ──> Enriched(N) ──> Decoder ──> Output N
Frame N+1 ──> Encoder ──> Bottleneck(N+1) ──┘
```

Process 3 consecutive frames as a batch. Each frame goes through the encoder independently (same weights, same compute). At the bottleneck, frame N cross-attends to bottlenecks N-1 and N+1. Then only frame N goes through the decoder.

**Why this might be better:**
- No external vision model needed at all (no DINOv3, no license issues)
- The encoder bottleneck features are already task-specific (trained for this exact denoising task)
- Three encoder passes are just batch_size=3 — same VRAM per frame, just 3x compute at encoder
- The decoder only runs once (for the center frame)
- The bottleneck features inherently encode noise patterns AND content, so cross-attention can learn to average out noise AND capture motion

**Speed estimate for C++ pipeline:**
- Encoder: ~6ms per frame (1/3 of the 19ms full forward)
- 3 encoder passes (batched): ~6ms (parallel on GPU)
- Cross-attention: ~1ms (tiny)
- Decoder: ~13ms per frame (2/3 of forward)
- Total: ~20ms per frame = **50 fps**
- With CUDA stream pipelining: could approach 40-45 fps end-to-end

**This might actually be tractable at 30 fps even on the RTX 3060.**

## Implementation Plan

### Phase 1: Single-frame semantic embedding (v1)
1. Find a permissive-license vision backbone (research needed)
2. Pre-extract embeddings for all training data
3. Add cross-attention module at DRUNet bottleneck (~200K params)
4. Freeze encoder/decoder, train only the cross-attention
5. Measure: does semantic conditioning improve PSNR / visual quality?

### Phase 2: Temporal embedding window (v2)
1. Pre-extract embeddings for consecutive frame sequences
2. Extend cross-attention to attend over 3-9 frames
3. Train with temporal window
4. Measure: does temporal context reduce noise / flickering?

### Phase 3: Bottleneck exchange (v3)
1. Modify training to process 3 consecutive frames per sample
2. Add cross-attention between bottlenecks (no external model)
3. Decoder runs only on center frame
4. This is self-contained — no external embedding model needed at inference
5. Measure: quality vs compute tradeoff, minimum viable window size

### Phase 4: Deploy
1. Export to ONNX with batch=3 input
2. Build TRT engine
3. Integrate with C++ pipeline (sequential frame feeding)
4. Benchmark: can we hit 30 fps?

## Architecture v4: Recurrent Output Feedback (PREFERRED)

The simplest and potentially most powerful variant. Feed the previous frame's cleaned output back as additional input channels.

```
Frame 1: [noisy_1, zeros]     -> Encoder -> Bottleneck -> Decoder -> output_1
Frame 2: [noisy_2, output_1]  -> Encoder -> Bottleneck -> Decoder -> output_2
Frame 3: [noisy_3, output_2]  -> Encoder -> Bottleneck -> Decoder -> output_3
...
```

**The previous cleaned output is the best possible reference:**
- It has the right content (same scene, slightly shifted)
- No noise (already cleaned by the model)
- Compatible features (produced by the same network)
- Implicitly encodes what the model "decided" about the scene

**Architecture change:** Just widen m_head from 3 input channels to 6 (or use a separate 3->16 conv for the reference and add/concat at the first feature level). Adds ~48-800 parameters depending on approach. Everything else stays identical.

**Why this is better than v1-v3:**
- Zero additional models (no DINO, no ConvNeXt, no licensing issues)
- Zero additional compute (just copy previous output to input buffer)
- Zero caching infrastructure
- Temporal consistency comes free (each frame is conditioned on previous output)
- Quality improves over a sequence (frame 10 benefits from chain of 9 cleaned frames)
- Training is simple: consecutive frame pairs, detach previous output gradient

**Training:**
1. Load consecutive frame pairs (frame_N, frame_N+1) with their targets
2. Forward pass on frame_N -> output_N (detach gradient)
3. Concatenate [frame_N+1, output_N] as 6-channel input
4. Forward pass -> loss against teacher target for frame_N+1
5. For first frame in a sequence, use zeros as the reference (model learns to handle cold start)

**Risk: error accumulation.** If the model makes an artifact in frame N, frame N+1 sees it as "reference." Mitigations:
- The original noisy frame is always the primary input (3 of 6 channels)
- Training with occasional zeros as reference (dropout on the reference channel, maybe 10-20%)
- The teacher target provides the correct answer regardless of reference quality

**Speed:** Identical to current model. One forward pass per frame. The C++ pipeline just holds the previous output buffer and copies it to the next input. At 39 fps NVEncC, no change.

## Architecture v5: Recurrent + Semantic (Future)

Combine v4 (recurrence) with v1 (semantic embedding) for maximum quality:

```
Frame N: [noisy_N, output_N-1] -> Encoder -> Bottleneck
                                                 |
                                   Cross-Attention with ConvNeXt features
                                                 |
                                              Decoder -> output_N
```

This gives the model:
- Previous cleaned frame (temporal continuity, noise-free reference)
- Semantic understanding (face vs wall vs text -> content-adaptive processing)
- Spatial awareness (where things are in the frame)

But start with v4 alone. If recurrence helps, THEN add semantic conditioning (v5) to see if the combination is additive.

## Vision Backbone (if needed for v1/v2/v5)

See `docs/research/vision-backbone-candidates.md`.

**Top pick: ConvNeXt-Tiny IN-22k** (MIT, 28.6M, pure CNN, stride-16 features at 67x120x384 for 1080p).

## Open Questions

- Does recurrent feedback (v4) improve quality significantly over single-frame?
- How many frames of recurrence until quality saturates?
- Does reference dropout rate (feeding zeros instead of previous output) matter for robustness?
- For v4, should the reference be the raw previous output, or downsampled/blurred to prevent the model from just copying it?
- Is error accumulation a real problem in practice, or does the noisy primary input keep the model grounded?
- If v4 works well, does adding semantic conditioning (v5) provide additional benefit, or is the recurrent reference sufficient?
- Can the model learn to use the reference for temporal consistency without explicit temporal loss?
