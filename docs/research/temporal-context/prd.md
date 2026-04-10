# PRD: Recurrent Temporal Context for DRUNet Student

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture](#2-architecture)
3. [Training Data Generation](#3-training-data-generation)
4. [Model Initialization & Sanity Checking](#4-model-initialization--sanity-checking)
5. [Dataset Implementation](#5-dataset-implementation)
6. [Training Loop Changes](#6-training-loop-changes)
7. [Progressive Unfreezing](#7-progressive-unfreezing)
8. [Validation Strategy](#8-validation-strategy)
9. [Loss Functions](#9-loss-functions)
10. [File-by-File Changes](#10-file-by-file-changes)
11. [Inference Pipeline Changes](#11-inference-pipeline-changes)
12. [ONNX Export Changes](#12-onnx-export-changes)
13. [Modal Wrapper Changes](#13-modal-wrapper-changes)
14. [Local Testing Before Modal](#14-local-testing-before-modal)
15. [Risks & Mitigations](#15-risks--mitigations)
16. [Success Metrics](#16-success-metrics)
17. [Cost Estimate](#17-cost-estimate)
18. [Implementation Sequence](#18-implementation-sequence)
19. [Training Command](#19-training-command)

---

## 1. Problem Statement

The DRUNet student (1.06M params, 49.98 dB PSNR) processes each frame independently. This causes:

**Temporal flickering.** Adjacent frames contain different noise. The model makes slightly different decisions per frame, causing visible flicker in flat regions and dark areas.

**Suboptimal denoising from single observation.** Each pixel is observed once through noise. With neighboring frames, the model has mathematically more information -- two noisy observations of the same pixel is strictly better than one.

**Expected gains.** Video SR literature shows 0.5-2.0 dB from temporal context. Conservative estimate for denoising: 0.3-0.8 dB PSNR plus temporal consistency.

---

## 2. Architecture

### 2.1 Model Change

Widen input from 3 to 9 channels. Everything else unchanged.

```
Old: UNetRes(in_nc=3, out_nc=3, nc=[16,32,64,128], nb=2)  -> m_head = Conv2d(3, 16, 3x3)
New: UNetRes(in_nc=9, out_nc=3, nc=[16,32,64,128], nb=2)  -> m_head = Conv2d(9, 16, 3x3)
```

**Input channel layout:**
- Channels 0-2: Previous frame's cleaned output (or zeros for cold start)
- Channels 3-5: Current noisy frame (being enhanced)
- Channels 6-8: Next noisy frame (1-frame lookahead)

**Parameter delta:** +864 parameters (1,063,776 -> 1,064,640). Speed impact: negligible (~35-38 fps from 39 fps).

### 2.2 DRUNet Parameter Structure (Student nc=[16,32,64,128] nb=2)

From actual model inspection:

| Module | Params | % of Total | Key Parameter Names |
|--------|--------|-----------|---------------------|
| `m_head` | 432 (3ch) / 1,296 (9ch) | 0.0% / 0.1% | `m_head.weight` |
| `m_down1` | 11,264 | 1.1% | `m_down1.0.res.{0,2}.weight`, `m_down1.1.res.{0,2}.weight`, `m_down1.2.weight` |
| `m_down2` | 45,056 | 4.2% | `m_down2.0.res.{0,2}.weight`, `m_down2.1.res.{0,2}.weight`, `m_down2.2.weight` |
| `m_down3` | 180,224 | 16.9% | `m_down3.0.res.{0,2}.weight`, `m_down3.1.res.{0,2}.weight`, `m_down3.2.weight` |
| `m_body` | 589,824 | 55.4% | `m_body.{0,1}.res.{0,2}.weight` |
| `m_up3` | 180,224 | 16.9% | `m_up3.0.weight`, `m_up3.1.res.{0,2}.weight`, `m_up3.2.res.{0,2}.weight` |
| `m_up2` | 45,056 | 4.2% | `m_up2.0.weight`, `m_up2.1.res.{0,2}.weight`, `m_up2.2.res.{0,2}.weight` |
| `m_up1` | 11,264 | 1.1% | `m_up1.0.weight`, `m_up1.1.res.{0,2}.weight`, `m_up1.2.res.{0,2}.weight` |
| `m_tail` | 432 | 0.0% | `m_tail.weight` |
| **Total** | **1,063,776** | | |

Note: `m_head` is a bare `Conv2d`, not wrapped in `Sequential`. The checkpoint key is `m_head.weight`, not `m_head.0.weight`. The PRD's `extract_drunet_features()` in `training/train.py` (line 88) already calls `model.m_head(x0)` directly.

---

## 3. Training Data Generation

### 3.1 Current Data Analysis

The training data uses 1/500 frame sampling. Frame naming: `{source}_{tag}_{index:05d}.png`. Consecutive indices within the same source+tag represent frames from the same episode sampled uniformly, but they are **not temporally adjacent** in the video -- they are approximately 500/24 = ~21 seconds apart.

**Current training set (6,266 frames):**
- 5,040 valid triplets (80.4%) -- three consecutive *index* values exist
- These triplets have ~21 second gaps between frames -- very large motion, completely different scenes

**Current validation set (692 frames):**
- Only 5 valid triplets -- the random split broke nearly all sequences

### 3.2 Is Re-Extraction Needed?

**No.** The 21-second frame gap is actually acceptable for this approach:

1. **The model cannot pixel-copy between frames.** With 21 seconds between them, the prev/next frames show completely different scenes. The model must learn robust temporal *statistics* (e.g., "prev_output is denoised, use it to understand the noise pattern") rather than naive frame averaging.

2. **This forces robustness.** If we trained with truly consecutive frames (33ms apart), the model might learn to simply copy from prev_output (since the scene barely changed). With large gaps, the model must learn what temporal context is *generally* useful.

3. **Cold-start behavior is the same.** With large gaps, the prev_output often shows a different scene, similar to the cold-start zeros case. The model learns that prev_output is *sometimes* useful (same scene) and sometimes not (scene change). This matches real inference where scene cuts happen.

4. **Existing triplet coverage is good.** 5,040 triplets from 6,266 frames means 80% of training data is usable. The remaining 20% (isolated frames) can still be used with dropout -- set prev and next to zeros.

**Decision: Use existing data as-is.** No re-extraction needed. The model will learn robust temporal features from the large-gap triplets, and cold-start dropout will handle isolated frames.

### 3.3 If Dense Temporal Data Is Later Needed

If temporal consistency metrics show the model isn't learning useful temporal features from the large-gap data (evaluated after Phase 1), we can add a dense subset:

```bash
# Extract 3 episodes at 1/100 sampling (~1,800 frames, ~22GB)
python tools/build_training_data.py --extract-only --sample-rate 100 \
    --only firefly  --episodes S01E01,S01E02,S01E03
```

This would require adding `--sample-rate` and `--episodes` flags to `tools/build_training_data.py` (currently hardcoded at `SAMPLE_RATE = 500` on line 48). Defer this to v2 if needed.

---

## 4. Model Initialization & Sanity Checking

### 4.1 Weight Initialization Code

The m_head weight for the 3ch student is shape `[16, 3, 3, 3]` (key: `m_head.weight`). For 9ch, it becomes `[16, 9, 3, 3]`.

```python
def init_temporal_head(model_9ch, pretrained_state_dict):
    """Initialize 9ch m_head from pretrained 3ch weights.
    
    Strategy: duplicate the pretrained 3ch weights across all three
    input groups (prev_output, current, next). The center group (channels 3-5)
    is an exact copy, so when prev/next are zeros, the model produces
    identical output to the 3ch pretrained model.
    
    Args:
        model_9ch: UNetRes with in_nc=9 (freshly constructed)
        pretrained_state_dict: state dict from 3ch checkpoint (ckpt['params'])
    """
    old_head = pretrained_state_dict["m_head.weight"]  # [16, 3, 3, 3]
    assert old_head.shape == (16, 3, 3, 3), f"Unexpected shape: {old_head.shape}"
    
    new_head = torch.zeros(16, 9, 3, 3, dtype=old_head.dtype)
    new_head[:, 0:3] = old_head  # prev_output channels
    new_head[:, 3:6] = old_head  # current frame channels (exact pretrained match)
    new_head[:, 6:9] = old_head  # next_noisy channels
    
    # Replace in state dict
    pretrained_state_dict["m_head.weight"] = new_head
    
    # Load all weights (m_head will use the expanded version, rest are unchanged)
    missing, unexpected = model_9ch.load_state_dict(pretrained_state_dict, strict=False)
    
    # Should have 0 missing, 0 unexpected
    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    
    return model_9ch
```

**Why duplicate (not zero-initialize) the outer channels:**
- With all three groups having the same weights, passing `[zeros, current, zeros]` through the 9ch model produces `old_weight @ zeros + old_weight @ current + old_weight @ zeros = old_weight @ current`. This is exactly the same as the 3ch model output.
- Wait -- that's wrong. The Conv2d operates as `W[:, 0:3] @ ch[0:3] + W[:, 3:6] @ ch[3:6] + W[:, 6:9] @ ch[6:9]`. If all three weight groups are identical, then `output = W_old @ (prev + current + next)`. With zeros in prev and next, output = `W_old @ current` = same as 3ch model. Correct.
- **But with non-zero prev/next**: output = `W_old @ (prev + current + next)` = 3x the signal if all three are similar. This will be too bright. We need to scale.

**Corrected initialization:**

```python
def init_temporal_head(model_9ch, pretrained_state_dict):
    """Initialize 9ch m_head from pretrained 3ch weights.
    
    Center channels (3-5) get full pretrained weights so the model
    starts at baseline quality when prev/next are zeros.
    Outer channels (0-2 and 6-8) are initialized to zero so they
    contribute nothing initially. The model learns to use them.
    """
    old_head = pretrained_state_dict["m_head.weight"]  # [16, 3, 3, 3]
    assert old_head.shape == (16, 3, 3, 3), f"Unexpected shape: {old_head.shape}"
    
    new_head = torch.zeros(16, 9, 3, 3, dtype=old_head.dtype)
    new_head[:, 3:6] = old_head  # current frame = exact pretrained weights
    # Channels 0-2 (prev) and 6-8 (next) start at zero
    
    pretrained_state_dict["m_head.weight"] = new_head
    missing, unexpected = model_9ch.load_state_dict(pretrained_state_dict, strict=False)
    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    
    return model_9ch
```

**Why zero (not small random or duplicate):**
- Zero-initialized outer channels mean the model starts at *exactly* baseline quality on day 1 -- prev/next channels produce zero activations regardless of their content.
- The head-only training phase (Phase 1) learns to pull useful signal from prev/next starting from zero.
- This avoids the 3x signal magnitude problem with duplicate initialization.
- The gradient signal is clean: the model can only improve by learning to use temporal context.

### 4.2 Sanity Checks (Run Before Training)

Three mandatory checks, implemented as a test function that runs before any training:

```python
def sanity_check_temporal_init(model_9ch, model_3ch, device="cuda"):
    """Verify 9ch model matches 3ch baseline when temporal channels are zeros.
    
    Must pass before training starts. Returns True if all checks pass.
    """
    model_9ch.eval()
    model_3ch.eval()
    
    # Test input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 256, 256, device=device)
    zeros = torch.zeros(1, 3, 256, 256, device=device)
    
    with torch.no_grad():
        # Check 1: zeros in prev/next -> identical to 3ch model
        input_9ch = torch.cat([zeros, x, zeros], dim=1)  # [1, 9, 256, 256]
        out_9ch = model_9ch(input_9ch)
        out_3ch = model_3ch(x)
        
        max_diff = (out_9ch - out_3ch).abs().max().item()
        psnr_match = 10 * torch.log10(1.0 / ((out_9ch - out_3ch)**2).mean()).item()
        
        print(f"  Check 1 (zeros in temporal): max_diff={max_diff:.2e}, PSNR={psnr_match:.1f} dB")
        assert max_diff < 1e-5, f"FAIL: 9ch model with zeros != 3ch model (max_diff={max_diff})"
        
        # Check 2: random noise in prev/next -> output should be same as 3ch
        # (because outer channel weights are zero-initialized)
        noise_prev = torch.randn_like(x)
        noise_next = torch.randn_like(x)
        input_noisy = torch.cat([noise_prev, x, noise_next], dim=1)
        out_noisy = model_9ch(input_noisy)
        
        max_diff_noisy = (out_noisy - out_3ch).abs().max().item()
        print(f"  Check 2 (random temporal):   max_diff={max_diff_noisy:.2e}")
        assert max_diff_noisy < 1e-5, (
            f"FAIL: 9ch model with random noise != 3ch model. "
            f"Temporal channels are not zero-initialized (max_diff={max_diff_noisy})"
        )
        
        # Check 3: PSNR at iteration 0 (rough estimate via center crop)
        # The 9ch model should match baseline ~49.98 dB
        print(f"  Check 3: Init PSNR will match baseline 49.98 dB (verified by checks 1-2)")
    
    print("  All sanity checks PASSED")
    return True
```

**Expected results at iteration 0:**
- PSNR should be exactly 49.98 dB (identical to 3ch baseline) since zero-initialized outer channels contribute nothing.
- If PSNR differs by more than 0.01 dB, the initialization is wrong. Do not proceed.

---

## 5. Dataset Implementation

### 5.1 TemporalTripletDataset

New class in `training/dataset.py`. Loads consecutive triplets `(frame_N-1, frame_N, frame_N+1)` with the same crop coordinates and augmentation applied to all three.

```python
class TemporalTripletDataset(Dataset):
    """Loads temporal triplets (prev, current, next) for recurrent training.
    
    Parses filenames to find consecutive frame indices within the same
    source+episode. Non-consecutive frames are handled via dropout:
    - Isolated frames: prev and next set to zeros (cold start)
    - Edge frames: missing neighbor set to zeros
    
    Directory structure:
        data_dir/input/source_tag_00042.png
        data_dir/target/source_tag_00042.png
    
    Returns:
        (prev_inp, cur_inp, next_inp, cur_tgt) as 4 tensors,
        each (3, crop_size, crop_size) float32
    """
    
    def __init__(self, data_dir, crop_size=256, augment=True,
                 prev_dropout=0.15, next_dropout=0.10,
                 max_frames=-1, cache_in_ram=False):
        self.crop_size = crop_size
        self.augment = augment
        self.prev_dropout = prev_dropout
        self.next_dropout = next_dropout
        self.cache_in_ram = cache_in_ram
        self.cached_images = None
        
        input_dir = os.path.join(data_dir, "input")
        target_dir = os.path.join(data_dir, "target")
        
        input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        if not input_files:
            raise FileNotFoundError(f"No PNG files in {input_dir}")
        
        # Build frame index: group by source+tag, find consecutive indices
        self._build_triplet_index(input_files, target_dir)
        
        if max_frames > 0:
            self.triplets = self.triplets[:max_frames]
        
        print(f"TemporalTripletDataset: {len(self.all_pairs)} pairs, "
              f"{len(self.triplets)} triplets + {len(self.isolated)} isolated, "
              f"crop={crop_size}, augment={augment}")
    
    def _build_triplet_index(self, input_files, target_dir):
        """Parse filenames, group by source+episode, find consecutive triplets."""
        import re
        from collections import defaultdict
        
        # Map filename -> (inp_path, tgt_path)
        self.all_pairs = {}
        for inp_path in input_files:
            fname = os.path.basename(inp_path)
            tgt_path = os.path.join(target_dir, fname)
            if os.path.exists(tgt_path):
                self.all_pairs[fname] = (inp_path, tgt_path)
        
        # Group by source+tag prefix
        groups = defaultdict(list)
        for fname in self.all_pairs:
            m = re.match(r'^(.+)_(\d{5})\.png$', fname)
            if m:
                prefix = m.group(1)
                idx = int(m.group(2))
                groups[prefix].append((idx, fname))
        
        # Sort each group by index, find consecutive triplets
        self.triplets = []    # (prev_fname, cur_fname, next_fname)
        self.isolated = []    # frames with no consecutive neighbors
        
        used_as_center = set()
        for prefix in sorted(groups):
            frames = sorted(groups[prefix], key=lambda x: x[0])
            indices = [f[0] for f in frames]
            fnames = [f[1] for f in frames]
            
            for i in range(len(frames)):
                has_prev = (i > 0 and indices[i] - indices[i-1] == 1)
                has_next = (i < len(frames)-1 and indices[i+1] - indices[i] == 1)
                
                if has_prev and has_next:
                    self.triplets.append((fnames[i-1], fnames[i], fnames[i+1]))
                    used_as_center.add(fnames[i])
                elif has_prev:
                    # Has prev but no next -- use as triplet with next=None
                    self.triplets.append((fnames[i-1], fnames[i], None))
                    used_as_center.add(fnames[i])
                elif has_next:
                    # Has next but no prev -- use as triplet with prev=None
                    self.triplets.append((None, fnames[i], fnames[i+1]))
                    used_as_center.add(fnames[i])
        
        # Frames not used as center of any triplet -> isolated
        for fname in self.all_pairs:
            if fname not in used_as_center:
                self.isolated.append(fname)
        
        # Combined list: triplets + isolated (with None neighbors)
        for fname in self.isolated:
            self.triplets.append((None, fname, None))
    
    def __len__(self):
        return len(self.triplets) * 100  # oversample like PairedFrameDataset
    
    def __getitem__(self, idx):
        triplet_idx = idx % len(self.triplets)
        prev_fname, cur_fname, next_fname = self.triplets[triplet_idx]
        
        # Load current frame (always exists)
        cur_inp_path, cur_tgt_path = self.all_pairs[cur_fname]
        cur_inp = self._load_image(cur_inp_path)
        cur_tgt = self._load_image(cur_tgt_path)
        
        h, w, _ = cur_inp.shape
        cs = self.crop_size
        
        # Random crop coordinates (shared across all frames in triplet)
        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        
        cur_inp = cur_inp[top:top+cs, left:left+cs]
        cur_tgt = cur_tgt[top:top+cs, left:left+cs]
        
        # Load prev (with dropout)
        use_prev = (prev_fname is not None and random.random() > self.prev_dropout)
        if use_prev:
            prev_inp = self._load_image(self.all_pairs[prev_fname][0])
            prev_inp = prev_inp[top:top+cs, left:left+cs]
        else:
            prev_inp = np.zeros((cs, cs, 3), dtype=np.float32)
        
        # Load next (with dropout)
        use_next = (next_fname is not None and random.random() > self.next_dropout)
        if use_next:
            next_inp = self._load_image(self.all_pairs[next_fname][0])
            next_inp = next_inp[top:top+cs, left:left+cs]
        else:
            next_inp = np.zeros((cs, cs, 3), dtype=np.float32)
        
        # Augmentation (same transform for all frames)
        if self.augment:
            if random.random() < 0.5:
                prev_inp = prev_inp[:, ::-1, :].copy()
                cur_inp = cur_inp[:, ::-1, :].copy()
                cur_tgt = cur_tgt[:, ::-1, :].copy()
                next_inp = next_inp[:, ::-1, :].copy()
            if random.random() < 0.5:
                prev_inp = prev_inp[::-1, :, :].copy()
                cur_inp = cur_inp[::-1, :, :].copy()
                cur_tgt = cur_tgt[::-1, :, :].copy()
                next_inp = next_inp[::-1, :, :].copy()
            if cs > 0:
                k = random.randint(0, 3)
                if k > 0:
                    prev_inp = np.rot90(prev_inp, k).copy()
                    cur_inp = np.rot90(cur_inp, k).copy()
                    cur_tgt = np.rot90(cur_tgt, k).copy()
                    next_inp = np.rot90(next_inp, k).copy()
        
        # Convert to tensors (C, H, W)
        prev_inp = torch.from_numpy(prev_inp.transpose(2, 0, 1))
        cur_inp = torch.from_numpy(cur_inp.transpose(2, 0, 1))
        cur_tgt = torch.from_numpy(cur_tgt.transpose(2, 0, 1))
        next_inp = torch.from_numpy(next_inp.transpose(2, 0, 1))
        
        return prev_inp, cur_inp, next_inp, cur_tgt
    
    def _load_image(self, path):
        """Load image as float32 RGB numpy array."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
```

### 5.2 Key Design Decisions

**Same crop coordinates for all three frames:** Critical. The model expects spatially aligned input. Even though consecutive frames are from the same source video, they were extracted at different timestamps. The crop provides a fixed spatial window -- the *content* differs between frames (21s gap), but the crop coordinates are consistent.

**Dropout rates:** 15% prev dropout, 10% next dropout. Prev dropout is higher because cold start (first frame) is the most common real-world scenario. Next dropout covers sequence-end and streaming scenarios.

**Isolated frames included:** The 20% of frames without consecutive neighbors (1,226 frames) are still used with both prev and next as zeros. This provides extra cold-start training examples.

**No RAM caching for triplets:** Each `__getitem__` loads 3-4 images from disk. At 256x256 crop with cv2, this takes ~2ms per image. With `num_workers=16` and prefetch, disk I/O is not the bottleneck on Modal (NVMe storage). If it becomes one, add crop caching later.

---

## 6. Training Loop Changes

### 6.1 Temporal Forward Pass

The training loop in `training/train.py` (starting at line 939) needs modification for the `--temporal` flag. The key addition: compute `prev_output` via a detached forward pass before the main training step.

```python
# Inside the training loop, after getting batch:

if temporal_mode:
    # Unpack temporal batch: (prev_inp, cur_inp, next_inp, cur_tgt)
    prev_inp, cur_inp, next_inp, cur_tgt = batch
    prev_inp = prev_inp.to(device, non_blocking=True)
    cur_inp = cur_inp.to(device, non_blocking=True)
    next_inp = next_inp.to(device, non_blocking=True)
    tgt_batch = cur_tgt.to(device, non_blocking=True)
    
    # Step 1: Compute prev_output (detached, no gradients)
    # Simulates inference: model processes prev frame with cold start
    with torch.no_grad():
        cold_input = torch.cat([
            torch.zeros_like(prev_inp),  # no prior output for prev frame
            prev_inp,
            cur_inp,  # current frame is prev's "next"
        ], dim=1)  # [B, 9, H, W]
        with torch.amp.autocast("cuda", enabled=use_amp):
            prev_output = model(cold_input).clamp(0, 1).detach()
    
    # Step 2: Build 9-channel input for current frame
    inp_batch = torch.cat([prev_output, cur_inp, next_inp], dim=1)  # [B, 9, H, W]
    
    # Step 3: Teacher target (3ch, unchanged)
    if teacher_model is not None:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            if use_feat_matching:
                tgt_batch, teacher_features = extract_drunet_features(
                    teacher_model, cur_inp,  # teacher still gets 3ch
                    needs_noise_map=teacher_needs_noise_map,
                    noise_level=teacher_noise_level,
                )
                tgt_batch = tgt_batch.clamp(0, 1)
            else:
                tgt_batch = teacher_model(cur_inp).clamp(0, 1)
    
    # Step 4: Student forward + loss (standard from here)
    with torch.amp.autocast("cuda", enabled=use_amp):
        if use_feat_matching:
            pred, student_features = extract_drunet_features(model, inp_batch)
        else:
            pred = model(inp_batch)
        pixel_loss = criterion(pred, tgt_batch)
else:
    # Original non-temporal path (unchanged)
    ...
```

**Compute cost of detached forward pass:** ~50% more GPU time per iteration (one extra forward, no backward). With batch=128 on L40S, the training loop currently runs at ~25 it/s. Temporal mode will be ~17 it/s. Total training time increases from ~2.8 hrs to ~4.2 hrs.

### 6.2 Feature Extraction for 9ch Input

The existing `extract_drunet_features()` (line 65-97 of `train.py`) works without modification -- it calls `model.m_head(x0)` which handles any input channel count. The teacher still receives 3ch input (`cur_inp`), so no changes needed there either.

---

## 7. Progressive Unfreezing

### 7.1 Phase Schedule

| Phase | Iters | What's Trainable | Params | Purpose |
|-------|-------|-----------------|--------|---------|
| 1: Head only | 0-500 | `m_head` | 1,296 | Learn temporal channel integration |
| 2: + Encoder | 500-2,500 | `m_head`, `m_down1`, `m_down2`, `m_down3` | 237,840 | Extract temporal features |
| 3: Full model | 2,500-4,500 | All parameters | 1,064,640 | Full fine-tuning |
| 4: Polish | 4,500-9,500 | All parameters (lower LR) | 1,064,640 | Convergence |

### 7.2 Implementation

```python
def apply_unfreeze_schedule(model, iteration, schedule):
    """Apply progressive unfreezing based on iteration count.
    
    Args:
        model: UNetRes instance
        iteration: current training iteration
        schedule: list of (iter_threshold, module_names_to_unfreeze)
    
    Example schedule:
        [(0, ['m_head']),
         (500, ['m_down1', 'm_down2', 'm_down3']),
         (2500, ['m_body', 'm_up3', 'm_up2', 'm_up1', 'm_tail'])]
    """
    # Start with everything frozen
    if iteration == 0:
        for p in model.parameters():
            p.requires_grad_(False)
    
    for threshold, module_names in schedule:
        if iteration == threshold:
            for name in module_names:
                module = getattr(model, name)
                for p in module.parameters():
                    p.requires_grad_(True)
                n_params = sum(p.numel() for p in module.parameters())
                print(f"  Unfreezing {name}: {n_params:,} params (iter {iteration})")
```

### 7.3 Prodigy Optimizer Across Phases

**Problem:** Prodigy auto-tunes learning rate based on gradient statistics. When new parameters are unfrozen, their gradients change the D estimate dramatically.

**Solution:** Rebuild optimizer at each phase transition with `--fresh-optimizer` semantics. This is clean and avoids D estimate corruption.

```python
# At each phase transition:
if iteration in unfreeze_thresholds:
    apply_unfreeze_schedule(model, iteration, schedule)
    
    # Rebuild optimizer with only trainable params
    train_params = [p for p in model.parameters() if p.requires_grad]
    if feat_criterion is not None:
        train_params += list(feat_criterion.parameters())
    
    optimizer = Prodigy(
        train_params,
        lr=1.0,
        d_coef=d_coef,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        safeguard_warmup=True,
        use_bias_correction=True,
    )
    # Rebuild scheduler for remaining iterations
    remaining = max_iters - iteration
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=0,
    )
    print(f"  Rebuilt optimizer for {sum(p.numel() for p in train_params):,} params")
```

### 7.4 Monitoring Each Phase

**Phase 1 (head only, 500 iters):**
- Watch: training loss should decrease. If flat, temporal channels aren't learning anything.
- If PSNR *decreases*, the learning rate is too high -- the head is forgetting the current-frame mapping. Lower d_coef.
- Expected: slight loss decrease as the model starts pulling signal from temporal channels.

**Phase 2 (+ encoder, 2000 iters):**
- Watch: training loss should accelerate its decrease. The encoder can now extract temporal features.
- Monitor PSNR at val_freq=500: should be at or above baseline by iter 2000.
- If no improvement: the temporal context may not be useful with 21s gaps. Consider dense data.

**Phase 3 (full model, 2000 iters):**
- Watch: PSNR should trend upward. If it drops, catastrophic forgetting -- need to lower LR.
- Monitor cold-start PSNR specifically (see validation strategy).

**Phase 4 (polish, 5000 iters):**
- Convergence. PSNR should plateau at target (50.3+ dB).
- If still improving, extend training.

### 7.5 What If Phase 1 Shows No Improvement?

If loss doesn't decrease during head-only training:
1. **Check initialization:** Run sanity checks again. If 9ch model with zeros doesn't match 3ch, fix init.
2. **Check data loading:** Verify triplets load correctly (see testing section).
3. **Check gradients:** `m_head.weight.grad` should be non-zero. If zero, the loss isn't propagating to the head.
4. **Increase Phase 1 to 1000 iters** -- 500 may not be enough for Prodigy to calibrate D.
5. **If truly no signal after 1000 iters:** The 21s frame gap may be too large. Pivot to dense temporal data extraction.

---

## 8. Validation Strategy

### 8.1 Current Problem

The validation set has only 5 valid triplets out of 692 frames. This is because `tools/build_training_data.py` (line 517-519) does a random 10% split per source, breaking temporal sequences.

### 8.2 Recommended Solution: Temporal-Aware Split

**Option B: Hold out entire episodes.** This is the cleanest approach:

| Held-out Episode | Frames | Why |
|-----------------|--------|-----|
| `firefly_S01E01` | ~249 (218 train + 31 val currently) | Largest single episode, sci-fi content |
| `expanse_S02E01` | ~134 (124 train + 10 val) | Different visual style |
| `squidgame_S02E02` | ~149 (131 train + 18 val) | Different content type |

This gives ~532 validation frames with excellent triplet coverage (most are consecutive within their episode). The training set keeps ~6,424 frames.

**Implementation in `tools/build_training_data.py`:**

Add to `do_denoise()` (line 513-528), replace the random split:

```python
# Temporal-aware split: hold out entire episodes for validation
VAL_EPISODES = {"firefly_S01E01", "expanse_S02E01", "squidgame_S02E02"}

def temporal_val_split(df):
    """Split by holding out entire episodes to preserve temporal sequences."""
    df["split"] = "train"
    for idx, row in df.iterrows():
        fname = row["filename"]
        # Parse source_tag from filename: source_tag_NNNNN.png
        parts = fname.rsplit("_", 1)  # split from right to get prefix
        if len(parts) == 2:
            prefix = parts[0]  # e.g., "firefly_S01E01"
            if prefix in VAL_EPISODES:
                df.loc[idx, "split"] = "val"
    return df
```

**Pros:**
- All frames within an episode stay together -- full triplet coverage in val
- No information leakage (train and val episodes are completely separate)
- Representative: 3 different sources in val
- ~8% val ratio (532/6958), close to current 10%

**Cons:**
- Requires rebuilding the train/val split and re-running `--denoise` and `--build-inputs` for moved frames
- Training set slightly larger (gains some current val frames from non-held-out episodes)

**Alternative (simpler): Use `--val-from-train`.**  
The training code already supports validating on a subset of training data. Add a `--val-from-train N` flag that randomly samples N frames from the training set for PSNR measurement. This avoids any data rebuilding.

**Recommendation:** Start with `--val-from-train` for speed (no data rebuild needed). If results are promising, do the full temporal-aware split before the final training run.

### 8.3 Temporal Validation Metrics

Add these to the validation function:

**1. Cold-start PSNR:** Process frames with `prev_output = zeros`. Must not regress below 49.0 dB. Measures single-frame quality preservation.

**2. Warm-start PSNR:** Process the same frames sequentially (prev_output from actual model output on the previous frame). Should be higher than cold-start.

**3. Temporal consistency (optional, Phase 4):** For consecutive frames in val, compute PSNR between consecutive outputs. Lower frame-to-frame difference = more temporally consistent. Define as:
```
consistency = mean(PSNR(output[t], output[t+1])) for consecutive val frames
```
Higher is better (more similar consecutive outputs), but only meaningful for visually similar consecutive frames (which we have in the held-out episodes).

### 8.4 Changes to `training/viz.py`

The `save_val_samples()` function (line 251-375) needs a temporal variant. For temporal models, generate a 4-panel composite:

```
[prev_output | input | teacher | model_output]
```

And add a text label showing "cold=XX.X dB" vs "warm=XX.X dB" in the model output panel.

Implementation: add a `temporal=False` parameter to `save_val_samples()`. When `True`, process 3 consecutive val frames sequentially, show the middle frame's result with cold-start and warm-start PSNR.

---

## 9. Loss Functions

**No changes to loss functions.** Same as current:
- Charbonnier pixel loss (primary)
- DISTS perceptual loss (weight=0.05)
- Feature matching loss (weight=0.1)

The teacher remains 3-channel and unchanged. The student uses temporal context to better approximate the teacher's single-frame output. The teacher target for the current frame is computed from `cur_inp` only (3ch), not from the temporal triplet.

**Temporal consistency loss (deferred to v2):** If temporal consistency metrics show the model flickers despite improved PSNR, add an explicit temporal loss:
```python
# L1 between consecutive model outputs (penalizes flickering)
temporal_loss = F.l1_loss(output_t, output_t_minus_1.detach())
```
This is deferred because (a) the recurrent architecture should implicitly learn consistency, and (b) the temporal loss can conflict with the pixel loss when scene content actually changes.

---

## 10. File-by-File Changes

### 10.1 `training/dataset.py`

**Change type:** Add new class

**What:** Add `TemporalTripletDataset` class (see Section 5.1 for full implementation).

**Location:** After `InputOnlyDataset` class (line 327).

**Dependencies:** None (uses same imports as existing classes).

**Import update:** Add to `training/train.py` (line 57):
```python
from training.dataset import PairedFrameDataset, InputOnlyDataset, GPUCachedDataset, TemporalTripletDataset
```

### 10.2 `training/train.py`

**Change 1: Import** (line 57)
- Add `TemporalTripletDataset` to import

**Change 2: `build_model()`** (line 104-152)
- When `--temporal` is set, construct model with `in_nc=9` instead of `in_nc=3`
- Line 143-145: change from hardcoded `in_nc=3` to `in_nc=getattr(args, 'in_nc', 3)`

```python
# Current (line 143):
model = UNetRes(in_nc=3, out_nc=3, nc=nc_list, nb=nb, act_mode='R', bias=False)

# New:
in_nc = getattr(args, 'in_nc', 3)
model = UNetRes(in_nc=in_nc, out_nc=3, nc=nc_list, nb=nb, act_mode='R', bias=False)
```

**Change 3: Weight initialization** (after line 537, inside `train()`)
- Add `init_temporal_head()` call when loading pretrained weights for temporal mode
- Add sanity check before training starts

**Change 4: Dataset selection** (lines 562-614)
- When `--temporal` is set, use `TemporalTripletDataset` instead of `InputOnlyDataset`/`PairedFrameDataset`
- Handle 4-element batch return `(prev_inp, cur_inp, next_inp, cur_tgt)` instead of `(inp, tgt)`

**Change 5: Training loop** (lines 939-1050)
- Add temporal forward pass block (Section 6.1)
- Add progressive unfreezing logic
- Rebuild optimizer at phase transitions

**Change 6: Validation** (lines 1188-1290)
- Add cold-start vs warm-start PSNR logging for temporal models

**Change 7: `extract_drunet_features()`** (lines 65-97)
- No changes needed -- already handles any input channel count via `model.m_head(x0)`

### 10.3 `training/viz.py`

**Change:** Modify `save_val_samples()` (line 251)
- Add `temporal=False, model_in_nc=3` parameters
- When temporal, generate 4-panel composite with cold/warm PSNR labels
- Process 3 consecutive val frames sequentially for warm-start evaluation

### 10.4 `training/losses.py`

**No changes.** All existing losses work with the temporal setup as-is.

### 10.5 `cloud/modal_train.py`

**Change 1: `train_remote()` function** (line 74)
- Add parameters: `temporal: bool = False`, `in_nc: int = 3`, `prev_dropout: float = 0.15`, `next_dropout: float = 0.10`, `unfreeze_schedule: str = ""`

**Change 2: Args passthrough** (lines 153-254)
- Add to Args object:
```python
args.temporal = temporal
args.in_nc = in_nc
args.prev_dropout = prev_dropout
args.next_dropout = next_dropout
args.unfreeze_schedule = unfreeze_schedule
```

**Change 3: `main()` function** (line 267)
- Add CLI parameters matching train_remote additions
- Pass through to `train_remote.remote()`

**Change 4: Memory** (line 70)
- Keep at `memory=32768` -- temporal training uses ~same RAM (crop cache is per-image, not per-triplet)

### 10.6 `pipelines/remaster.py`

**Change 1: `load_model()`** (line 52-105)
- Already auto-detects `in_nc` from checkpoint (line 62-65). No changes needed.

**Change 2: `remaster_video()`** (line 108-378)
- Add prev_output buffer management:

```python
# Before the inference loop (after line 233):
in_nc = model.m_head.weight.shape[1]  # auto-detect from model
is_temporal = (in_nc == 9)

if is_temporal:
    prev_output = torch.zeros(batch_size, 3, h_pad, w_pad, device=DEVICE, dtype=dtype)

# Inside the inference loop (replace lines 274-276):
with torch.inference_mode():
    if is_temporal:
        # Collect next frame for lookahead
        next_frames, next_done = peek_next_batch()  # need to implement lookahead
        next_tensor = prepare_tensor(next_frames, 1 - buf_idx) if next_frames else zeros
        input_9ch = torch.cat([prev_output[:len(cur_frames)], cur_tensor, next_tensor], dim=1)
        out_t = model(input_9ch)
        prev_output[:len(cur_frames)] = out_t.detach()
    else:
        out_t = model(cur_tensor)
```

**Note on lookahead buffer:** The current reader thread provides frames sequentially. For the 9ch model, we need a 1-frame lookahead. Modify the frame collection to always have the next batch pre-read. Since the reader thread already runs asynchronously via a queue, this means reading 2 frames ahead instead of 1.

**Change 3: Warmup** (line 126-139)
- If temporal, warmup with 9ch input:
```python
if is_temporal:
    dummy = torch.randn(batch_size, 9, h_pad, w_pad, device=DEVICE, dtype=dtype)
else:
    dummy = torch.randn(batch_size, 3, h_pad, w_pad, device=DEVICE, dtype=dtype)
```

### 10.7 `tools/export_onnx.py`

**Change 1: Auto-detect in_nc from checkpoint** (lines 45-48)

```python
# Current (line 46):
model = UNetRes(in_nc=3, out_nc=3, nc=nc_list, nb=nb, act_mode='R', bias=False)

# New:
# Auto-detect in_nc from checkpoint
head_key = "m_head.weight"
if head_key in state:
    in_nc = state[head_key].shape[1]
    print(f"Auto-detected in_nc={in_nc} from checkpoint")
else:
    in_nc = 3
model = UNetRes(in_nc=in_nc, out_nc=3, nc=nc_list, nb=nb, act_mode='R', bias=False)
```

**Change 2: Dummy input shape** (line 67)

```python
# Current:
dummy = torch.randn(1, 3, H, W)

# New:
dummy = torch.randn(1, in_nc, H, W)
```

**Change 3: Input/output names** (lines 73-85)
- Update input name to reflect 9ch when applicable
- Dynamic axes remain the same (height, width are dynamic; channels are fixed at export time)

**Change 4: Sanity check** (line 119)
```python
test_input = np.random.randn(1, in_nc, H, W).astype(np.float32)
```

**Change 5: CLI** (add `--in-nc` flag, though auto-detection from checkpoint is preferred)

### 10.8 `tools/build_training_data.py`

**Change (deferred):** Only needed if we switch to temporal-aware split.

- Line 513-528: Replace random split with episode-based split (see Section 8.2)
- Add `VAL_EPISODES` constant
- Modify `do_denoise()` to use `temporal_val_split()` instead of random split
- Re-run `--denoise` and `--build-inputs` after changing split

### 10.9 `tools/verify_data.py`

**Change:** Add triplet verification.

```python
def check_triplets(data_dir, label):
    """Count valid temporal triplets in a dataset directory."""
    import re
    from collections import defaultdict
    
    input_dir = os.path.join(data_dir, "input")
    files = sorted(os.listdir(input_dir))
    
    groups = defaultdict(list)
    for f in files:
        m = re.match(r'^(.+)_(\d{5})\.png$', f)
        if m:
            groups[m.group(1)].append(int(m.group(2)))
    
    total_triplets = 0
    for prefix in groups:
        indices = sorted(groups[prefix])
        for i in range(1, len(indices) - 1):
            if indices[i] - indices[i-1] == 1 and indices[i+1] - indices[i] == 1:
                total_triplets += 1
    
    print(f"  {label}: {len(files)} frames, {total_triplets} valid triplets "
          f"({100*total_triplets/max(len(files),1):.0f}%)")
    return total_triplets
```

---

## 11. Inference Pipeline Changes

### 11.1 Python Pipeline (`pipelines/remaster.py`)

The main change is adding a prev_output buffer and 1-frame lookahead. See Section 10.6 for details.

**Key constraint:** The model auto-detects `in_nc` from the checkpoint. A 9ch checkpoint will automatically use temporal mode. A 3ch checkpoint uses single-frame mode. No CLI flag needed.

### 11.2 VapourSynth Pipeline (`remaster/encode.vpy`)

**Deferred.** vs-mlrt processes frames independently and doesn't support stateful recurrence. Options:
1. Use the Python pipeline for temporal encoding
2. Write a custom VapourSynth filter that maintains state (complex C++ work)
3. Accept single-frame quality for VapourSynth real-time playback (still very good at 49.98 dB)

### 11.3 ONNX/TRT

Rebuild with 9-channel input shape. TensorRT engines are resolution-and-shape-specific, so new engines will be built automatically on first use.

---

## 12. ONNX Export Changes

See Section 10.7. The main change is auto-detecting `in_nc` from the checkpoint weights. The export command becomes:

```bash
python tools/export_onnx.py \
    --checkpoint checkpoints/drunet_student_temporal/final.pth \
    --nc-list 16,32,64,128 --nb 2 \
    --output checkpoints/drunet_student_temporal/drunet_student_temporal.onnx
```

The script auto-detects `in_nc=9` from the checkpoint. No `--in-nc` flag needed.

---

## 13. Modal Wrapper Changes

See Section 10.5. New CLI flags:

```bash
modal run cloud/modal_train.py \
    --temporal \
    --in-nc 9 \
    --prev-dropout 0.15 \
    --next-dropout 0.10 \
    --unfreeze-schedule "500:m_down1,m_down2,m_down3;2500:m_body,m_up3,m_up2,m_up1,m_tail"
```

The `--temporal` flag triggers:
1. `TemporalTripletDataset` instead of `InputOnlyDataset`
2. 9ch model construction
3. Temporal weight initialization from pretrained checkpoint
4. Detached prev_output forward pass in training loop
5. Progressive unfreezing via `--unfreeze-schedule`

---

## 14. Local Testing Before Modal

Run these tests locally before spending Modal credits. All tests run on CPU (no GPU needed) except the memory estimate.

### Test 1: Dataset Loading

```python
# test_temporal_dataset.py (run locally, no GPU)
import sys
sys.path.insert(0, ".")
from training.dataset import TemporalTripletDataset

ds = TemporalTripletDataset("data/training/train", crop_size=256, augment=False)
print(f"Triplets: {len(ds.triplets)}, Isolated: {len(ds.isolated)}")

# Verify triplet loading
prev, cur, next_, tgt = ds[0]
print(f"prev: {prev.shape}, cur: {cur.shape}, next: {next_.shape}, tgt: {tgt.shape}")
assert prev.shape == (3, 256, 256)
assert cur.shape == (3, 256, 256)

# Verify same crop (visually -- save to disk)
import cv2
import numpy as np
for name, t in [("prev", prev), ("cur", cur), ("next", next_), ("tgt", tgt)]:
    img = (t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imwrite(f"test_{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Saved test_*.png -- verify visually that crops are spatially aligned")

# Verify dropout
zeros_count = 0
for i in range(100):
    prev, cur, next_, tgt = ds[i]
    if prev.sum() == 0:
        zeros_count += 1
print(f"Prev-zeros in 100 samples: {zeros_count} (expect ~15)")
```

### Test 2: Model Initialization

```python
# test_temporal_init.py (run locally, CPU only)
import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "reference-code/KAIR")
from models.network_unet import UNetRes

# Load 3ch pretrained
ckpt = torch.load("checkpoints/drunet_student/final.pth", map_location="cpu", weights_only=True)
state_3ch = ckpt["params"]

model_3ch = UNetRes(in_nc=3, out_nc=3, nc=[16,32,64,128], nb=2, act_mode='R', bias=False)
model_3ch.load_state_dict(state_3ch)
model_3ch.eval()

# Build 9ch model with temporal init
model_9ch = UNetRes(in_nc=9, out_nc=3, nc=[16,32,64,128], nb=2, act_mode='R', bias=False)

# Zero-init temporal channels, copy pretrained to center
state_9ch = dict(state_3ch)
old_head = state_9ch["m_head.weight"]  # [16, 3, 3, 3]
new_head = torch.zeros(16, 9, 3, 3)
new_head[:, 3:6] = old_head
state_9ch["m_head.weight"] = new_head
model_9ch.load_state_dict(state_9ch)
model_9ch.eval()

# Check: zeros in temporal channels -> identical output
torch.manual_seed(42)
x = torch.randn(1, 3, 256, 256)
zeros = torch.zeros(1, 3, 256, 256)

with torch.no_grad():
    out_3ch = model_3ch(x)
    out_9ch = model_9ch(torch.cat([zeros, x, zeros], dim=1))
    
    diff = (out_3ch - out_9ch).abs().max().item()
    print(f"Max diff (zeros): {diff:.2e}")
    assert diff < 1e-5, f"FAIL: diff={diff}"
    
    # Check: random in temporal channels -> still identical (zero-init)
    out_rand = model_9ch(torch.cat([torch.randn_like(x), x, torch.randn_like(x)], dim=1))
    diff_rand = (out_3ch - out_rand).abs().max().item()
    print(f"Max diff (random): {diff_rand:.2e}")
    assert diff_rand < 1e-5, f"FAIL: diff={diff_rand}"

print("PASSED: 9ch model matches 3ch baseline with zero-init temporal channels")
del ckpt, state_3ch, state_9ch, model_3ch, model_9ch  # free RAM
```

### Test 3: Single Training Step

```python
# test_temporal_train_step.py (run locally, CPU)
import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "reference-code/KAIR")
from models.network_unet import UNetRes
from training.losses import CharbonnierLoss

# Small model for CPU testing
model = UNetRes(in_nc=9, out_nc=3, nc=[16,32,64,128], nb=2, act_mode='R', bias=False)

# Freeze everything except m_head (Phase 1)
for p in model.parameters():
    p.requires_grad_(False)
for p in model.m_head.parameters():
    p.requires_grad_(True)

criterion = CharbonnierLoss()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

# Fake batch
B = 2
x = torch.randn(B, 9, 64, 64)
target = torch.randn(B, 3, 64, 64)

# Forward + backward
pred = model(x)
loss = criterion(pred, target)
loss.backward()

# Check gradients
head_grad = model.m_head.weight.grad
print(f"m_head.weight.grad: shape={head_grad.shape}, norm={head_grad.norm():.4f}")
assert head_grad.norm() > 0, "FAIL: no gradient on m_head"

# Check frozen params have no grad
assert model.m_down1[0].res[0].weight.grad is None, "FAIL: m_down1 should be frozen"
assert model.m_body[0].res[0].weight.grad is None, "FAIL: m_body should be frozen"

optimizer.step()
print("PASSED: gradients flow correctly through frozen/unfrozen params")
```

### Test 4: Unfreezing

```python
# test_unfreeze.py (run locally, CPU)
import sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "reference-code/KAIR")
from models.network_unet import UNetRes

model = UNetRes(in_nc=9, out_nc=3, nc=[16,32,64,128], nb=2, act_mode='R', bias=False)

# Phase 1: freeze all, unfreeze head
for p in model.parameters():
    p.requires_grad_(False)
for p in model.m_head.parameters():
    p.requires_grad_(True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Phase 1: {trainable} trainable params (expect 1296)")
assert trainable == 1296

# Phase 2: unfreeze encoder
for name in ['m_down1', 'm_down2', 'm_down3']:
    for p in getattr(model, name).parameters():
        p.requires_grad_(True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Phase 2: {trainable} trainable params (expect 237840)")
assert trainable == 1296 + 11264 + 45056 + 180224  # = 237840

# Phase 3: unfreeze all
for p in model.parameters():
    p.requires_grad_(True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Phase 3: {trainable} trainable params (expect 1064640)")
assert trainable == 1064640

print("PASSED: unfreezing works correctly")
```

### Test 5: Memory Estimate

```python
# Verify batch_size=128 fits in L40S 48GB (run locally, rough estimate)
# Model: 1.064M params * 4 bytes = 4MB
# Teacher: 32.6M params * 4 bytes = 130MB
# Input batch (9ch): 128 * 9 * 256 * 256 * 2 (fp16) = 144MB
# Activations (forward): ~3-4 GB (estimated from current training)
# Activations (backward): ~3-4 GB
# Optimizer (Prodigy): ~3x model params = 12MB
# DISTS VGG: ~80MB
# Total: ~8-10 GB + detached forward (~3GB transient)
# Peak: ~13 GB << 48 GB L40S
print("Memory estimate: ~13 GB peak, 48 GB available. SAFE at batch=128.")
```

---

## 14.5 Insights from NeRV Denoising Experiments (2026-04-10)

The NeRV exploration (`docs/research/nerv-denoising/`) ran 45+ experiments trying to use neural video representations for self-supervised denoising. While NeRV didn't produce usable denoised output, the experiments yielded insights relevant to temporal DRUNet:

### Loss function insights
- **Patch-level color preservation** (`F.avg_pool2d` 32x32 patch L1, weight 5.0) effectively prevents brightness/color shift without constraining per-pixel values. Consider adding this to DRUNet training.
- **Asymmetric residual loss** (`relu(input - output)` for Sobel penalty) only penalizes removed detail, not added detail. Useful if we want the temporal model to enhance beyond its training targets.
- **Edge preservation loss** (`relu(sobel(input) - sobel(output))`) at weight 0.5 maintains sharpness. Weight 5.0 produces artifacts.
- **L1+FFT loss** is a strong regularizer — the FFT component prevents the model from ignoring high-frequency content. Don't remove it.

### Architecture insights
- **3x3 minimum kernel size** in all decoder blocks significantly improves quality (+3.3 dB in NeRV). Check if DRUNet's student could benefit from larger kernels in early layers.
- **Skip connections with learnable scale** (init 0.1) let the model control how much encoder detail bypasses the bottleneck. Too much (scale 1.0) passes noise; too little (0.01) is blurry.
- **Skip dropout 0.15-0.30** regularizes but kills high-frequency output. Only use if overfitting is severe.
- **Stride alignment padding** (replicate-pad to stride-aligned dimensions) prevents edge artifacts. DRUNet already handles this but worth verifying for the 9-channel input.

### Frame2Frame / temporal insights
- **Adjacent frames as targets (Noise2Noise)** successfully prevents noise memorization in NeRV — the model can't memorize per-frame noise when the target has different noise. This validates the temporal DRUNet approach of using multiple frames.
- **Motion between frames causes blur** if frames aren't aligned. The 9-channel DRUNet avoids this because it inputs raw frames (not aligned) and lets the network learn alignment internally.
- **More frames = more temporal context = better** but only up to a point. 32 frames was better than 16, but 8 was worse (not enough diversity). The 3-frame window (prev + current + next) in this PRD is the minimum viable temporal context.

### Autoresearch infrastructure
The autoresearch agent loop (`docs/research/nerv-denoising/autoresearch.md`) worked well for rapid experimentation:
- Agents run one experiment, write to a shared research log, then exit
- Next agent reads the log and builds on previous findings
- `results.tsv` tracks metrics across experiments
- W&B provides visual comparison across runs
- This infrastructure can be applied to DRUNet temporal context experiments

### What NeRV couldn't do (and DRUNet can)
- NeRV is self-supervised — no clean targets. DRUNet has SCUNet targets (53 dB teacher).
- NeRV must fit one video clip per model. DRUNet generalizes across all content.
- NeRV's bottleneck is too narrow to filter structured HEVC noise. DRUNet's U-Net architecture with skip connections is designed for this.
- The 9-channel temporal DRUNet combines the best of both: temporal context (like NeRV) with supervised training on clean targets (like DRUNet).

---

## 15. Risks & Mitigations

### 15.1 Frame Gap Problem (MEDIUM)

**Risk:** 21-second gaps between consecutive training frames mean prev/next frames show completely different scenes. The model might learn that temporal channels are useless noise.

**Mitigation:** This is actually desirable for robustness (see Section 3.2). The model learns that prev_output is sometimes helpful (same scene) and sometimes not (scene change). Cold-start dropout (15%) reinforces this.

**Monitoring:** Watch Phase 1 training loss. If flat after 500 iters, temporal context isn't being learned. Pivot to dense data extraction.

### 15.2 Error Accumulation (HIGH)

**Risk:** In inference, prev_output is the model's own output from the previous frame. Errors in frame N compound in frame N+1. After 50+ frames, quality could degrade.

**Mitigation:**
1. Zero-init ensures temporal channels start with no contribution -- errors must be *learned*, not inherited from initialization.
2. Cold-start dropout (15%) during training makes the model robust to garbage in prev_output.
3. The current frame channels (3-5) always have full pretrained weight -- the model can always fall back to single-frame behavior.
4. **Test:** After training, run inference on a 500-frame clip and plot per-frame PSNR. Should be stable or improving, never degrading.

### 15.3 Model Ignores Temporal Channels (MEDIUM)

**Risk:** Zero-initialized outer channels might stay near zero if the gradient signal is too weak during Phase 1 (only 1,296 trainable params).

**Mitigation:** 
1. Phase 1 runs for 500 iterations with only m_head trainable -- strong gradient signal concentrated on few params.
2. If Phase 1 shows no loss decrease, extend to 1000 iters.
3. Initialize outer channels with very small random values (1e-3) instead of zeros as a fallback.

### 15.4 Catastrophic Forgetting During Full Unfreezing (MEDIUM)

**Risk:** When Phase 3 unfreezes the decoder + body (72% of params), the model might lose what it learned about single-frame processing.

**Mitigation:**
1. Progressive unfreezing prevents sudden large parameter changes.
2. Rebuild Prodigy optimizer at each phase transition with fresh D estimate.
3. Monitor cold-start PSNR at each validation step -- if it drops below 49.0 dB, halt training and reduce d_coef.

### 15.5 Rollback Plan

**Keep the single-frame student checkpoint safe.** The training command uses `--checkpoint-dir checkpoints/drunet_student_temporal` (separate from `checkpoints/drunet_student`). If temporal training fails:
1. The original 3ch student at `checkpoints/drunet_student/final.pth` is untouched.
2. All deployed pipelines continue using the 3ch model.
3. The 9ch experiment is isolated in its own checkpoint directory.

---

## 16. Success Metrics

### 16.1 Sanity Gate (Iteration 0)

| Check | Criteria | Action if Failed |
|-------|----------|-----------------|
| 9ch with zeros = 3ch output | max_diff < 1e-5 | Fix initialization, do not train |
| 9ch with random = 3ch output | max_diff < 1e-5 | Fix initialization, do not train |
| PSNR at iter 0 | = 49.98 dB (within 0.01) | Fix initialization, do not train |

### 16.2 Phase 1 End (Iteration 500)

| Metric | Criteria | Action if Failed |
|--------|----------|-----------------|
| Training loss | Decreasing trend | Extend to 1000 iters, or try small-random init |
| Cold-start PSNR | >= 49.9 dB | Check for regression in m_head weights |

### 16.3 Phase 2 End (Iteration 2,500)

| Metric | Criteria | Action if Failed |
|--------|----------|-----------------|
| PSNR | >= 49.98 dB (at or above baseline) | Temporal context isn't helping. Try dense data. |
| Training loss | Clear downward trend | Model is learning temporal features |

### 16.4 Phase 3 End (Iteration 4,500)

| Metric | Criteria | Action if Failed |
|--------|----------|-----------------|
| PSNR | >= 50.1 dB (upward trend) | Lower LR, extend training |
| Cold-start PSNR | >= 49.5 dB | Catastrophic forgetting. Reduce d_coef, add regularization |

### 16.5 Phase 4 End (Iteration 9,500)

| Metric | Criteria | Action if Failed |
|--------|----------|-----------------|
| **Ship** | PSNR >= 50.3 dB, cold-start >= 49.0 dB, no error accumulation | Deploy |
| **Promising** | PSNR >= 50.0 dB, measurable consistency improvement | Continue training, try dense data |
| **Fail (revert)** | PSNR < 49.9 dB, or cold-start < 49.0 dB, or error accumulation | Revert to 3ch student |

### 16.6 Temporal Consistency Metric

Measured on a 500-frame test clip:

```python
def measure_temporal_consistency(model, video_path, max_frames=500):
    """Measure frame-to-frame output stability."""
    # Process sequentially with prev_output buffer
    psnrs = []  # per-frame PSNR vs teacher
    diffs = []  # L1 between consecutive outputs
    
    prev_output = None
    for frame_idx, (cur_frame, next_frame) in enumerate(frame_pairs):
        if prev_output is None:
            prev_output = torch.zeros_like(cur_frame)
        
        input_9ch = torch.cat([prev_output, cur_frame, next_frame], dim=1)
        output = model(input_9ch)
        
        if frame_idx > 0:
            diffs.append(F.l1_loss(output, prev_output_saved).item())
        
        prev_output_saved = output.detach().clone()
        prev_output = output.detach()
    
    # Lower mean_diff = more temporally consistent
    return {
        "mean_frame_diff": np.mean(diffs),
        "std_frame_diff": np.std(diffs),
        "max_frame_diff": np.max(diffs),
    }
```

**Baseline (3ch model):** Measure the same metric by running the 3ch model on the same clip (no temporal context). The temporal model should have lower `mean_frame_diff`.

---

## 17. Cost Estimate

| Phase | Iters | Time (L40S) | Cost |
|-------|-------|-------------|------|
| Phase 1: Head only | 500 | ~0.3 hrs | $0.59 |
| Phase 2: + Encoder | 2,000 | ~1.2 hrs | $2.34 |
| Phase 3: Full model | 2,000 | ~1.2 hrs | $2.34 |
| Phase 4: Polish | 5,000 | ~3.0 hrs | $5.85 |
| **Total (1 run)** | **9,500** | **~5.7 hrs** | **$11.12** |

**Notes:**
- ~50% slower than non-temporal training due to detached forward pass
- Phase 1-3 could be one continuous run (no restart needed)
- Budget for 3 experimental runs: **$35-45 total**

---

## 18. Implementation Sequence

| Step | Time | Description |
|------|------|-------------|
| 1. Dataset class | 1-2 hrs | Add `TemporalTripletDataset` to `training/dataset.py` |
| 2. Local tests 1-4 | 30 min | Run dataset, init, training step, unfreezing tests |
| 3. Train.py changes | 2-3 hrs | `build_model()`, temporal init, training loop, unfreezing |
| 4. Modal wrapper | 30 min | Add temporal args to `cloud/modal_train.py` |
| 5. Local test 5 | 10 min | Run 10-iteration local training to verify full pipeline |
| 6. Training run | 5-6 hrs | Phases 1-4 on L40S, monitor W&B |
| 7. Evaluation | 1 hr | Temporal inference on test clips, measure consistency |
| 8. Inference pipeline | 1-2 hrs | Update `pipelines/remaster.py` for temporal mode |
| 9. ONNX export | 30 min | Update `tools/export_onnx.py`, export 9ch model |

**Total implementation time: ~12-15 hours** (not counting Modal training time).

---

## 19. Training Command

```bash
# Clear any old stop signals
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -c "
import modal
d = modal.Dict.from_name('train-signals')
d['drunet_student_temporal'] = False
"

# Run temporal training
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run \
    cloud/modal_train.py \
    --arch drunet --nc-list 16,32,64,128 --nb 2 \
    --teacher checkpoints/drunet_teacher/final.pth --teacher-model drunet \
    --checkpoint-dir checkpoints/drunet_student_temporal \
    --feature-matching-weight 0.1 --optimizer prodigy --batch-size 128 \
    --temporal --in-nc 9 --prev-dropout 0.15 --next-dropout 0.10 \
    --unfreeze-schedule "500:m_down1,m_down2,m_down3;2500:m_body,m_up3,m_up2,m_up1,m_tail" \
    --max-iters 9500 --val-freq 500 \
    --ema --wandb --fresh-optimizer
```

**Key differences from current student training:**
- `--temporal` flag enables temporal dataset + training loop
- `--in-nc 9` sets 9-channel input model
- `--checkpoint-dir checkpoints/drunet_student_temporal` isolates from existing student
- `--unfreeze-schedule` controls progressive unfreezing
- `--batch-size 128` (down from 192 to account for extra forward pass memory)
- `--val-freq 500` (more frequent validation to catch problems early)
- No `--resume` (fresh start from pretrained 3ch weights)

---

## Critical Files Summary

| File | Change Type | Priority |
|------|------------|----------|
| `training/dataset.py` | Add TemporalTripletDataset class | P0 |
| `training/train.py` | Temporal training loop, init, unfreezing | P0 |
| `cloud/modal_train.py` | Temporal args passthrough | P0 |
| `pipelines/remaster.py` | Temporal inference with prev_output buffer | P1 |
| `tools/export_onnx.py` | Auto-detect in_nc from checkpoint | P1 |
| `training/viz.py` | Temporal validation samples (cold/warm) | P2 |
| `tools/verify_data.py` | Add triplet counting | P2 |
| `tools/build_training_data.py` | Temporal-aware split (deferred) | P3 |
