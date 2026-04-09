# PRD: NeRV-Based Video Denoising for Training Target Generation

## Goal

Build a NeRV-based pipeline that fits a neural video representation to noisy TV episodes, exploiting the network's spectral bias to produce denoised frames. These frames replace SCUNet GAN as training targets for our fast DRUNet student model.

**Success criteria:** Output frames are visually cleaner than SCUNet GAN targets, with better temporal consistency and preserved fine detail (faces, text, fabric texture). Measured by: PSNR vs originals, visual inspection of dark scenes, temporal flicker metrics.

## Why NeRV for Denoising

Neural video representations (NeRV family) encode an entire video as a neural network. When trained on noisy video frames:

1. **Spectral bias**: Networks learn low-frequency structure (edges, faces, textures) before high-frequency components (noise). With controlled capacity and regularization, the network reconstructs clean signal while being unable to memorize per-frame noise.
2. **Temporal consistency**: The network produces a single coherent representation of the video. Frame-to-frame flickering is structurally impossible because adjacent frames share the same weights.
3. **Content memorization**: Over thousands of frames, the network learns the show's visual vocabulary -- recurring faces, sets, costumes, lighting patterns. When reconstructing a dark or blurry frame, it draws on this learned understanding.
4. **Self-supervised**: No clean targets needed. The noisy video IS the training data. The architecture itself acts as the denoising prior.

## Architecture Choice: HNeRV + Modifications

Based on available code, licensing, and published results:

**Base: HNeRV** (Hybrid NeRV, CVPR 2023)
- MIT-licensed via the original author's repo, Apache 2.0 via Boosting-NeRV
- Content-adaptive: lightweight ConvNeXt encoder takes the actual frame as input (not just a frame index), producing embeddings that the decoder upsamples to full resolution
- Fastest decoding in the NeRV family (~24 fps V100, ~50-70 fps H100 at 1080p)
- Code already cloned at `reference-code/Boosting-NeRV/model_hnerv.py`

**Modifications for denoising:**
1. **Weight regularization on late decoder layers** (from zero-shot blind denoising literature) -- L2 penalty prevents high-frequency noise fitting
2. **High-frequency preserving loss** (from Boosting-NeRV) -- prevents over-smoothing that spectral bias can cause
3. **GOP-based segmentation** (from PNVC, AAAI 2025) -- split episodes into 300-600 frame chunks instead of fitting one network to 60K frames
4. **Early stopping with noise-fit detection** -- monitor when the model transitions from learning structure to memorizing noise
5. **Per-frame bias conditioning** (from ActINR, CVPR 2025) -- shared decoder weights + per-frame predicted biases for temporal adaptivity without noise memorization

## Reference Code

All locally available:
- `reference-code/Boosting-NeRV/model_hnerv.py` -- HNeRV model (encoder + decoder)
- `reference-code/Boosting-NeRV/model_blocks.py` -- NeRVBlock, UpConv, DownConv, SFT, ConvNeXt encoder
- `reference-code/Boosting-NeRV/model_nerv.py` -- Original NeRV (frame-index-only, no encoder)
- `reference-code/Boosting-NeRV/train_nerv_all.py` -- Training loop
- `reference-code/FFNeRV/model.py` -- FFNeRV with flow-guided temporal grids
- `reference-code/ConvNeXt-V2/models/convnextv2.py` -- ConvNeXt V2 blocks (GRN)
- `lib/convnext_autoencoder.py` -- Our ConvNeXt-V2 autoencoder (built during exploration)

## Data

Source: Firefly S01 episodes, 1080p HEVC. Already have ~1,876 extracted frames in `data/originals/` with metadata. For NeRV we need sequential frames (not 1/500 sampled), so we'll extract dense frame sequences from selected clips.

## Execution Plan

This is structured as a series of experiments, each validating a specific hypothesis before building on it. A staff ML engineer should treat each phase as a standalone verification -- don't proceed to the next phase until the current one produces expected results.

---

### Phase 0: Environment and Baseline Setup

**Goal:** Get HNeRV running on a short clip, establish baseline metrics.

#### 0.1 Extract a test clip
- Pick a 10-second (240 frame) clip from Firefly S01E01 that contains:
  - A well-lit dialogue scene (faces, sharp detail)
  - A darker scene transition (noise is most visible here)
  - Some camera motion (tests temporal handling)
- Extract as individual PNG frames at 1080p using ffmpeg
- Store at `data/nerv-test/clip_01/` (240 frames, numbered `00000.png` to `00239.png`)

#### 0.2 Run HNeRV baseline
- Use the existing Boosting-NeRV training script (`train_nerv_all.py`) to fit HNeRV to the test clip
- Start with their default hyperparameters for 1080p (check their scripts/ folder for configs)
- If no 1080p config exists, adapt from their 720p config: `--enc_strds 5 4 3 2 --dec_strds 5 4 3 2` (total stride = 120, maps to ~9x16 base resolution for 1080p)
- Train for 300 epochs (their standard)
- Decode all 240 frames
- **Measure:** PSNR of reconstructed frames vs original noisy frames (should be 30+ dB for good fit)
- **Visually inspect:** Are reconstructed frames cleaner than input? Is there visible denoising? Is detail preserved or smeared?

#### 0.3 Compare against SCUNet baseline
- Run SCUNet GAN on the same 240 frames (our existing target generator)
- Side-by-side comparison: HNeRV reconstruction vs SCUNet output vs noisy original
- **Key question:** Does HNeRV preserve more fine detail than SCUNet while still removing noise?

#### 0.4 Run on Modal
- Wrap the training in a Modal function (similar to `cloud/modal_train.py`)
- Verify it runs on L40S/H100
- Benchmark: training time for 240 frames x 300 epochs
- Benchmark: decoding speed (fps) on the trained model

**Exit criteria for Phase 0:** HNeRV fits the test clip at 30+ dB PSNR, decoding produces visually denoised frames, and it runs on Modal without issues.

---

### Phase 1: Spectral Bias Denoising Validation

**Goal:** Prove that NeRV's spectral bias actually denoises HEVC compression artifacts (not just synthetic Gaussian noise).

#### 1.1 Noise-fit curve analysis
- Train HNeRV on the test clip but save checkpoints every 10 epochs
- For each checkpoint, decode all frames and measure:
  - PSNR vs noisy input (fit quality -- should increase monotonically)
  - PSNR vs SCUNet target (denoising quality -- should increase then plateau/decrease)
  - High-frequency energy ratio (FFT): measure how much HF content the model has learned
- **Plot the noise-fit curve:** There should be a clear inflection point where the model transitions from learning structure to memorizing noise
- **Key finding:** What epoch range produces the best denoising? Is early stopping sufficient or do we need explicit regularization?

#### 1.2 Capacity vs denoising tradeoff
- Train HNeRV at 3 different model sizes (e.g., 0.5M, 1.5M, 3M params) on the same clip
- Smaller models = more aggressive spectral bias = more denoising but less detail
- Larger models = can memorize more noise = better reconstruction but less denoising
- **Find the sweet spot:** model size that maximizes denoising quality on HEVC artifacts

#### 1.3 Weight regularization experiment
- Add L2 weight decay specifically to the last 2-3 decoder UpConv layers
- Compare: no regularization vs lambda=1e-4 vs lambda=1e-3 vs lambda=1e-2
- **Hypothesis:** Regularizing late layers suppresses noise fitting while allowing early layers to learn structure
- Measure PSNR vs SCUNet target at convergence for each regularization strength

#### 1.4 HEVC artifact analysis
- HEVC artifacts are NOT random per-frame like Gaussian noise. They have structure:
  - Block boundaries (8x8, 16x16 quantization blocks)
  - Banding in smooth gradients
  - Ringing around edges
  - Some artifacts persist across frames (in P/B-frames referencing the same I-frame)
- **Test:** Do persistent artifacts get memorized by NeRV? Compare denoising quality on I-frames vs P/B-frames
- **If persistent artifacts are memorized:** We may need to add explicit artifact suppression (e.g., augment training with random block-boundary masks)

**Exit criteria for Phase 1:** Clear evidence that spectral bias denoises HEVC artifacts. Identified optimal model size, regularization strength, and training duration. Understanding of which artifact types are handled well vs poorly.

---

### Phase 2: Scaling to Full Episodes

**Goal:** Make NeRV practical for 60K-frame episodes.

#### 2.1 GOP segmentation design
- A single HNeRV model for 60K frames would need enormous capacity (and training time)
- Instead: split the episode into Groups of Pictures (GOPs), process sequentially
- Design decisions:
  - **GOP size:** Test 150, 300, 600 frames. Tradeoff: smaller = faster training but less temporal context; larger = more context but needs bigger model
  - **GOP overlap:** Overlap by 10-30 frames at boundaries to avoid discontinuities. Blend with linear ramp in overlap region

#### 2.2 Serial warm-start training (key optimization)

Instead of training each GOP from scratch independently, **warm-start each GOP from the previous GOP's trained weights**. The network already knows what the show looks like from prior frames — it just needs to learn what changed.

```
GOP 1 (frames 0-299):     train from scratch, 300 epochs -> decode -> save
GOP 2 (frames 300-599):   resume from GOP 1 weights, ~100 epochs -> decode -> save
GOP 3 (frames 600-899):   resume from GOP 2 weights, ~100 epochs -> decode -> save
...
GOP N:                     resume from GOP N-1, ~100 epochs -> decode -> save
```

**Why this works:**
- The decoder weights encode the show's visual vocabulary (faces, textures, sets, lighting). This vocabulary is shared across GOPs — Badger's face in GOP 5 is the same face the network learned in GOP 1.
- Only the specific pixel arrangements change between GOPs, not the underlying visual vocabulary.
- Prodigy optimizer auto-tunes the learning rate on each warm-start, adapting step size to how much the new GOP differs from the previous one.
- **Catastrophic forgetting is irrelevant** — we decode and save each GOP's frames immediately. We never need the network to reconstruct an old GOP.

**Benefits:**
- ~3x faster convergence per GOP after the first (100 vs 300 epochs)
- Better denoising quality — the network accumulates knowledge across the episode
- Natural scene-cut detection: if convergence is slow, the content changed significantly
- The decoder becomes a rolling compressed representation of "everything the show looks like so far"

**Training strategy:**
- GOP 1: Full 300 epochs from scratch with Prodigy
- GOP 2+: Resume from previous GOP with `--fresh-optimizer` (reset Prodigy, keep model weights). Train until val PSNR plateaus or max 150 epochs.
- After scene cuts: Allow up to 200 epochs (more new content to learn)
- Monitor convergence rate per GOP — if a GOP converges in 30 epochs, the content was very similar to the previous GOP

#### 2.3 Scene-cut-aware segmentation
- Detect scene cuts (frame difference threshold or ffmpeg scene detection)
- Start new GOPs at scene boundaries
- Variable-length GOPs: 150-600 frames, cut at natural scene transitions
- After a scene cut, allocate more training epochs (new content requires more adaptation)

#### 2.4 Processing pipeline on Modal
- Serial within an episode (warm-start requires sequential processing)
- **Parallel across episodes** — process episode 1-14 simultaneously on separate GPUs
- Design Modal wrapper:
  - Upload episode frames to Modal volume
  - For each episode: serial GOP loop (train -> decode -> save -> next GOP)
  - Each episode runs on one GPU continuously
- **Cost estimate (revised):**
  - GOP 1: 300 epochs x 2s/epoch (H100) = 10 min
  - GOP 2-200: 100 epochs x 2s/epoch = 3.3 min each
  - Total per episode: 10 + (199 x 3.3) = ~11 GPU-hours = ~$43 at $3.95/hr
  - **Firefly S01 (14 episodes in parallel): ~11 hours wall time, ~$605 total**
  - Compare to independent training: ~17 GPU-hours/episode = $67/ep = $940 total

#### 2.5 Boundary blending
- At GOP boundaries, decode both adjacent models for the overlap frames
- Linear blend: `output[t] = alpha * gop_a[t] + (1 - alpha) * gop_b[t]` where alpha ramps from 1 to 0
- With warm-start, boundaries should be smoother since the models share most of their weights
- Verify: no visible seam at GOP transitions

**Exit criteria for Phase 2:** Full episode processing pipeline works end-to-end on Modal. No visible GOP boundary artifacts. Warm-start convergence confirmed faster than from-scratch.

---

### Phase 3: Quality Optimization

**Goal:** Maximize denoising quality while preserving fine detail.

#### 3.1 Loss function design
The standard HNeRV uses L2 (MSE) loss. For denoising, we want:
- **Base loss:** L1 or Charbonnier (less sensitive to outliers than L2, better for preserving edges)
- **Perceptual loss (DISTS or LPIPS):** At low weight (0.01-0.05) to preserve structural similarity
- **HF-preserving loss** (from Boosting-NeRV): Penalize difference in high-frequency components between reconstruction and input, preventing over-smoothing
- **Do NOT use adversarial/GAN loss:** We want faithful reconstruction, not hallucination

Test each loss component independently, then in combination. Measure PSNR + SSIM + visual quality.

#### 3.2 Encoder architecture
HNeRV uses a ConvNeXt V1 encoder. Consider:
- Upgrading to ConvNeXt V2 (GRN for better feature competition) -- we have the code at `lib/convnext_autoencoder.py`
- Using pretrained ImageNet weights for the encoder (gives semantic understanding from the start)
- Encoder depth/width vs denoising quality tradeoff
- **Key insight:** The encoder is the denoising bottleneck. It compresses the noisy frame to a compact embedding. If the embedding is small enough, noise can't fit through.

#### 3.3 Temporal conditioning
Basic HNeRV treats each frame independently (encoder maps frame to embedding). Add temporal context:
- **Option A:** Concatenate prev/next frame embeddings (average or attention-weighted)
- **Option B:** Use FFNeRV-style flow-guided temporal grids
- **Option C:** Feed prev/next frames as extra encoder input channels
- Test each option on the 240-frame test clip. Does temporal conditioning improve denoising on dark/noisy frames?

#### 3.4 Dark scene enhancement
Dark scenes have the worst noise and the least detail. The network should learn to:
- Recognize that dark frames of the same scene share the same content
- Reconstruct detail in dark frames by borrowing from brighter frames of the same scene
- **Test:** Extract a clip with brightness variation (e.g., character walking from shadow to light). Does the network transfer detail from bright frames to dark frames?

**Exit criteria for Phase 3:** Optimal loss function, encoder architecture, and temporal conditioning identified. Dark scene quality exceeds SCUNet targets.

---

### Phase 4: Target Generation Pipeline

**Goal:** Production pipeline that processes entire shows and outputs training targets.

#### 4.1 Full pipeline implementation
```
Input: Episode MKV file
  -> FFmpeg: extract all frames as PNG (or direct decoding in training loop)
  -> Scene detection: identify cut points
  -> GOP segmentation: split into 150-600 frame chunks at scene boundaries
  -> Modal parallel training: fit HNeRV to each GOP
  -> Modal parallel decoding: reconstruct all frames from trained models
  -> Boundary blending: smooth GOP transitions
  -> Quality validation: automated checks (PSNR, no black frames, no artifacts)
  -> Output: denoised frame sequence
```

#### 4.2 Quality validation checks
Automated per-frame and per-GOP checks:
- No NaN/Inf pixels
- PSNR vs input > threshold (model actually fit)
- No sudden brightness shifts at GOP boundaries
- Color consistency with input (no color drift)
- Sharpness metric doesn't drop below input (no over-smoothing)

#### 4.3 A/B comparison: NeRV vs SCUNet targets
- Process 3-5 episodes with both NeRV and SCUNet
- Train separate DRUNet students on each target set
- Compare final student quality: PSNR, SSIM, visual quality, temporal consistency
- **This is the ultimate validation:** Do NeRV targets produce a better student model?

#### 4.4 Cost optimization
- Profile GPU utilization during training -- are we compute or memory bound?
- Test L40S ($1.95/hr) vs H100 ($3.95/hr) -- L40S may be more cost-effective
- Can we reduce training epochs without losing quality? (Phase 1 noise-fit curve informs this)
- Model quantization for faster decoding?

**Exit criteria for Phase 4:** End-to-end pipeline processes a full episode unattended. Student model trained on NeRV targets outperforms student trained on SCUNet targets.

---

### Phase 5: Iteration and Extensions (Future)

These are ideas to explore only after Phase 4 is validated:

#### 5.1 Cross-episode warm-start
- The serial warm-start concept extends across episodes: start episode 2 from episode 1's final GOP weights
- Characters, sets, and costumes recur across episodes -- the network already knows them
- Process episodes in chronological order to maximize knowledge accumulation
- Could reduce per-GOP epochs even further for later episodes (the visual vocabulary is well-established)

#### 5.3 Selective enhancement
- Not all frames need equal treatment. Dark/noisy frames benefit most
- Route easy frames through a lighter model or skip NeRV entirely
- Adaptive model capacity per GOP based on estimated noise level

#### 5.4 Integration with temporal DRUNet
- If the temporal DRUNet PRD (`docs/research/temporal-context/prd.md`) is also implemented, NeRV targets + temporal student could compound the quality gains

---

## Key Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| HEVC artifacts are too structured for spectral bias | Medium | High | Phase 1.4 validates this early. Fallback: combine NeRV with explicit artifact mask |
| Over-smoothing destroys fine detail | Medium | High | HF-preserving loss, encoder capacity tuning, visual validation at every phase |
| 60K frames per episode doesn't scale | Low | High | GOP segmentation (Phase 2) is the standard solution, well-validated in PNVC |
| Cost too high per show | Medium | Medium | Serial warm-start (~$605 vs $940), L40S testing, early stopping |
| Temporal discontinuity at GOP boundaries | Low | Medium | Overlap + blending (Phase 2.4) |
| No improvement over SCUNet | Low | High | Phase 0.3 catches this immediately on first test clip |

## Hardware and Budget

- **Development/testing:** Local RTX 3060 (6GB) for small clips, Modal L40S for larger tests
- **Production processing:** Modal H100 ($3.95/hr)
- **Estimated cost per episode:** ~$43 (serial warm-start: 1 GOP from scratch + 199 warm-started)
- **Estimated cost for Firefly S01 (14 episodes):** ~$605 (parallel across episodes, serial within)
- **Training data storage:** ~30GB per episode of extracted frames (Modal volume)

## Timeline Estimates

Not providing time estimates per CLAUDE.md guidance. The phases are ordered by dependency -- each phase's exit criteria must be met before proceeding. Phase 0-1 are the critical validation steps. If Phase 1 shows NeRV doesn't denoise HEVC artifacts well, we pivot to the temporal DRUNet approach instead.

## Files Created During This Exploration

- `lib/convnext_autoencoder.py` -- ConvNeXt-V2 autoencoder (may be useful for encoder experiments in Phase 3.2)
- `tools/test_convnext_autoencoder.py` -- Test suite for the autoencoder
- `checkpoints/convnextv2_pretrained/convnextv2_atto_1k_224_ema.pt` -- Pretrained Atto weights
- `reference-code/ConvNeXt-V2/` -- ConvNeXt V2 reference (CC-BY-NC weights)
- `reference-code/ConvNeXt/` -- ConvNeXt V1 reference (MIT)
- `reference-code/FFNeRV/` -- FFNeRV (MIT)
- `reference-code/Boosting-NeRV/` -- HNeRV + Boosting framework (Apache 2.0)
