# Temporal Context Architecture for Video Enhancement

## The Insight

Our DRUNet student processes one frame at a time. It's remarkably good for 1.06M params, but it has no context about neighboring frames. Video has massive temporal redundancy — the same face, the same wall, the same fabric appears across dozens of consecutive frames with different noise in each. A model that can see even one neighboring frame has strictly more information to work with.

The question is how to give it that context with minimal cost.

## Approach: Recurrent Output Feedback (Start Here)

The simplest temporal signal: **feed the previous frame's cleaned output back as extra input channels.**

```
Frame 1: [zeros,    noisy_1, noisy_2] -> model -> output_1
Frame 2: [output_1, noisy_2, noisy_3] -> model -> output_2
Frame 3: [output_2, noisy_3, noisy_4] -> model -> output_3
```

9-channel input: previous cleaned output (3ch) + current noisy frame (3ch) + next noisy frame (3ch).

### Why this should work

**Previous cleaned output** is the best possible temporal reference:
- Already denoised — shows what the scene looks like without artifacts
- Same model produced it — features are compatible
- Implicitly encodes the model's "decision" about the scene
- The model can compare it against the current noisy frame to see what changed

**Next noisy frame** gives a second observation of the same content:
- Same scene, different noise realization
- Slightly shifted in time (motion provides different viewpoints)
- The model can internally compare current vs next to separate signal from noise
- Two noisy observations of the same pixel is mathematically better than one

**Cold start:** Frame 1 gets zeros in the prev_output slot. Training includes random zero-dropout (10-20%) on the prev channel so the model stays robust when the reference is missing.

### Architecture change

Minimal. Just widen the input convolution:

```python
# Old: 3-channel input
model = UNetRes(in_nc=3, out_nc=3, nc=[16,32,64,128], nb=2)

# New: 9-channel input
model = UNetRes(in_nc=9, out_nc=3, nc=[16,32,64,128], nb=2)
```

This changes m_head from `Conv2d(3, 16, 3x3)` to `Conv2d(9, 16, 3x3)` — adding exactly **864 new parameters** (from 432 to 1296). The rest of the 1.06M network stays pretrained and frozen initially.

### Training plan

1. Initialize m_head with pretrained weights for the center 3 channels (current frame), small random values for the other 6
2. Load consecutive frame triplets from training data
3. For each triplet (N-1, N, N+1):
   - Run model on frame N-1 to get prev_output (detach gradient)
   - Input = [prev_output, noisy_N, noisy_N+1]
   - Target = teacher output for frame N
   - Standard loss: Charbonnier + DISTS + feature matching
4. With 10-20% probability, replace prev_output with zeros (cold start robustness)
5. Fine-tune: first only m_head (100 iters), then full model

### Speed impact

**Effectively zero.** One conv layer processes 9 instead of 3 input channels — 3x compute for a single layer out of ~30. The rest of the network is identical.

At inference, we need a 1-frame lookahead buffer (next frame decoded before processing current). With async NVDEC in the C++ pipeline, the next frame is already in GPU memory. We also hold the previous output in a GPU buffer — one extra 1080p frame at fp16 = 12MB.

**Estimated speed: 35-38 fps** (from current 39 fps). Negligible difference.

### What we're testing

Does the model learn to USE the temporal context? Specifically:
- Does PSNR improve? (more information should = better reconstruction)
- Does temporal consistency improve? (less flickering between frames)
- Does the model learn different behavior for cold-start (zeros) vs warmed-up (real reference)?
- How quickly do the benefits saturate? (is frame 3 much better than frame 2?)

## Future: Bottleneck Transformer

If the recurrent approach shows signal, the natural next step is adding a small cross-attention transformer at the DRUNet bottleneck. The bottleneck is already a compressed representation (135x240x128 for 1080p) — it's the "plan" for what the output frame should look like.

A transformer here could:
- Attend across spatial positions to propagate temporal information ("this face patch at position (50,30) was at (52,31) in the previous frame")
- Learn to weight the previous output vs current observation per-region
- Serve as an integration point for external embeddings (see below)

This would add ~100K-500K parameters. But get signal from the simple version first.

## Future: External Semantic Embeddings

**ConvNeXt-Tiny (ImageNet-22k, MIT license, 28.6M params)** produces 384-dim features at stride 16 — almost exactly matching our bottleneck spatial resolution (67x120 vs 135x240). These features encode semantic content: "this patch is a face," "this is fabric," "this is text."

If injected at the bottleneck via cross-attention:

```
Bottleneck (135x240x128) ──> Q
ConvNeXt Stage 3 (67x120x384) ──> K, V (upsampled to 135x240)
                                    │
                              Cross-Attention
                                    │
                              Enriched Bottleneck
```

The model could learn content-adaptive processing — different "filters" for faces vs walls vs text. The ConvNeXt features run once per frame (~30ms), can be cached, and are noise-invariant due to large-scale pretraining.

This is appealing because:
- The resolution match at stride 16 is natural, not forced
- ConvNeXt is pure CNN, TRT-compatible, MIT licensed
- 384 dimensions encode rich semantic information from 21K ImageNet categories
- The transformer at the bottleneck is the right integration point — it decides how to blend semantic context with the denoiser's own features

But this is strictly additive to the recurrent approach. Start with recurrence, measure, then layer on semantic conditioning if needed.

## Future: Bottleneck Exchange (Batch Processing)

Instead of external embeddings, process 3 consecutive frames as a batch and cross-attend between their bottlenecks:

```
Frame N-1 ──> Encoder ──> Bottleneck(N-1) ──┐
Frame N   ──> Encoder ──> Bottleneck(N)   ──┼── Cross-Attention ──> Enriched(N)
Frame N+1 ──> Encoder ──> Bottleneck(N+1) ──┘
                                                    │
                                              Decoder ──> Output N
```

No external model, no licensing issues. The encoder features are task-specific. Three encoder passes (batchable) + one decoder pass. Potential for encoder output caching since frame N's encoder features can be reused when it becomes N-1 for the next step.

**Speed estimate:** ~20ms per output frame with caching = 50 fps. Potentially tractable at 30 fps even on RTX 3060.

## Implementation Priority

1. **Recurrent output feedback (9ch input)** — prototype first. Minimal code change, maximal signal.
2. **Measure** — does temporal context help? How much? Where?
3. **Bottleneck transformer** — if recurrence helps, add cross-attention to better integrate the temporal signal
4. **Semantic embeddings** — if content-adaptive processing shows value, add ConvNeXt conditioning
5. **Bottleneck exchange** — if we want to go fully self-contained with no external model
6. **Next-frame prediction loss** — auxiliary training signal that forces the bottleneck to encode motion (training only, thrown away at inference)

## Reference

- Vision backbone candidates: `docs/research/vision-backbone-candidates.md`
- ConvNeXt-Tiny IN-22k: `convnext_tiny.fb_in22k` via timm (MIT license, 28.6M params)
- DINOv2 ViT-S/14: Apache 2.0 but needs reduced-resolution input for 1080p
- Current student: DRUNet nc=[16,32,64,128] nb=2, 1.06M params, 39 fps
