# PRD: Recurrent Temporal Context for DRUNet Student

## 1. Problem Statement

The DRUNet student (1.06M params, 49.98 dB PSNR) processes each frame independently. This causes:

**Temporal flickering.** Adjacent frames contain different noise. The model makes slightly different decisions per frame, causing visible flicker in flat regions and dark areas.

**Suboptimal denoising from single observation.** Each pixel is observed once through noise. With neighboring frames, the model has mathematically more information.

**Expected gains.** Video SR literature shows 0.5-2.0 dB from temporal context. Conservative estimate for denoising: 0.3-0.8 dB PSNR plus temporal consistency.

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

**Parameter delta:** +864 parameters (1.06M -> 1.061M). Speed impact: negligible (~35-38 fps from 39 fps).

### 2.2 Weight Initialization

Duplicate pretrained weights across all three channel groups:

```python
old_weight = pretrained["m_head.0.weight"]  # [16, 3, 3, 3]
new_weight = torch.zeros(16, 9, 3, 3)
new_weight[:, 0:3] = old_weight  # prev_output (cleaned frame stats)
new_weight[:, 3:6] = old_weight  # current (exact pretrained match)
new_weight[:, 6:9] = old_weight  # next_noisy (same noise stats)
```

Model starts at full baseline quality on day 1. Extra channels learn to contribute incrementally.

## 3. Training Pipeline Changes

### 3.1 New Dataset: TemporalTripletDataset

Loads consecutive triplets `(frame_N-1, frame_N, frame_N+1)` from same source video.

**Frame naming:** `firefly_S01E01_00042.png` — source prefix + frame index. Parse, group by source, identify consecutive indices.

**Current data:** 6,266 frames -> ~5,040 valid triplets (80.4%).

**Validation set problem:** Current val has only ~5 valid triplets (random split broke sequences). Fix: temporal-aware split keeping consecutive frames together, or use `--val-from-train` subset.

**Cropping:** All three frames in a triplet use same crop coordinates and augmentation. Critical for temporal alignment.

**Dropout:**
- 15% chance: replace prev_output with zeros (cold start robustness)
- 10% chance: replace next_noisy with zeros (sequence end robustness)

### 3.2 Previous Output Computation

For each triplet during training:

```python
# 1. Run student on frame N-1 (no gradients)
with torch.no_grad():
    cold_input = torch.cat([zeros, prev_inp, cur_inp], dim=1)  # 9ch
    prev_output = model(cold_input).clamp(0, 1).detach()

# 2. Build 9-channel input for frame N
student_input = torch.cat([prev_output, cur_inp, next_inp], dim=1)

# 3. Teacher target (unchanged, 3ch input)
with torch.no_grad():
    target = teacher_model(cur_inp).clamp(0, 1)

# 4. Student forward + loss
pred = model(student_input)
loss = criterion(pred, target)
```

Extra forward pass adds ~50% compute but zero extra backward (gradient detached).

## 4. Progressive Unfreezing

| Phase | Iters | Trainable | Params | Purpose |
|-------|-------|-----------|--------|---------|
| 1: Head only | 500 | m_head | 1,296 | Learn channel integration |
| 2: + Encoder | 2,000 | m_head + m_down1/2/3 | ~340K | Extract temporal features |
| 3: Full model | 2,000 | All | 1.061M | Full fine-tuning |
| 4: Polish | 5,000 | All (lower LR) | 1.061M | Convergence |

All parameters initialized in optimizer from start (frozen ones skipped via `requires_grad=False`).

## 5. Loss Functions

**No changes.** Same as current: Charbonnier + DISTS (0.05) + Feature matching (0.1).

Teacher stays 3-channel, unchanged. Student uses temporal context to better approximate teacher's single-frame output.

Temporal consistency loss deferred to v2 (probably unnecessary if recurrence works).

## 6. Data Requirements

Current data is sufficient (5,040 triplets). Need to:
1. Rebuild train/val split to preserve sequences (temporal-aware split)
2. Build frame index on dataset init (group by source, find consecutive indices)

## 7. Inference Pipeline Changes

### Python Pipeline (`pipelines/remaster.py`) — modify first

```python
prev_output = torch.zeros(1, 3, h_pad, w_pad, device=DEVICE, dtype=dtype)

for current_frame, next_frame in frame_pairs:
    input_9ch = torch.cat([prev_output, current_frame, next_frame], dim=1)
    output = model(input_9ch)
    prev_output = output.clone()
    write_frame(output)
```

`load_model()` already auto-detects `in_nc` from checkpoint.

### VapourSynth — defer (needs stateful filter or vs-mlrt C++ changes)

### C++ Pipeline — natural fit (sequential processing, add prev_output buffer)

### ONNX/TRT — rebuild with 9-channel input shape

## 8. Metrics

- **PSNR** vs teacher (target: 49.98 -> 50.3+ dB)
- **Temporal consistency:** frame-to-frame PSNR std deviation
- **Cold start PSNR:** first frame with zeros (must not regress)
- **Visual:** side-by-side video comparison, dark scenes, flickering

## 9. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Error accumulation | HIGH | Cold start dropout (15%), raw frame always present, teacher target from raw |
| Training instability | MEDIUM | Progressive unfreezing, pretrained weight init |
| Val data gap | MEDIUM | Rebuild temporal-aware split before training |
| Frame gap (1/500 sampling) | LOW | ~21s between triplet frames = large motion. Model can't pixel-copy, must learn robust features |

## 10. Cost Estimate

| Phase | Time (L40S) | Cost |
|-------|-------------|------|
| Phases 1-3 | ~1.3 hrs | $2.54 |
| Phase 4 | ~1.5 hrs | $2.93 |
| **Total** | **~2.8 hrs** | **$5.46** |

Budget for 3-4 experimental runs: **$30-50 total**.

## 11. Success Criteria

| Criteria | Threshold |
|----------|-----------|
| **Ship** | PSNR >= 50.3 dB, no cold-start regression, visible temporal improvement |
| **Promising** | PSNR >= 50.0 dB, measurable consistency improvement |
| **Fail (revert)** | PSNR < 49.9 dB, or cold-start < 49.0 dB, or error accumulation after 50+ frames |

## 12. Implementation Sequence

1. **Data prep** (30 min): Temporal-aware train/val split, verify triplet counts
2. **Dataset class** (1-2 hrs): `TemporalTripletDataset`, triplet index, same-crop augmentation
3. **Training loop** (2-3 hrs): `--temporal`, weight init, prev_output computation, progressive unfreezing
4. **Modal wrapper** (30 min): Add temporal args, increase memory
5. **Training** (3-4 hrs): Phases 1-4 on L40S, monitor W&B
6. **Evaluation** (1 hr): Temporal inference on test clips, compare to baseline
7. **Export** (30 min): ONNX with 9ch input, TRT engine rebuild

## 13. Training Command

```bash
modal run cloud/modal_train.py --arch drunet --nc-list 16,32,64,128 --nb 2 \
    --teacher checkpoints/drunet_teacher/final.pth --teacher-model drunet \
    --checkpoint-dir checkpoints/drunet_student_temporal \
    --feature-matching-weight 0.1 --optimizer prodigy --batch-size 128 \
    --temporal --in-nc 9 --prev-dropout 0.15 --next-dropout 0.10 \
    --unfreeze-schedule "500:m_down1,m_down2,m_down3;2500:m_body,m_up3,m_up2,m_up1,m_tail" \
    --max-iters 9500 --ema --wandb --resume --fresh-optimizer
```

## Critical Files

- `training/dataset.py` — new TemporalTripletDataset class
- `training/train.py` — temporal training loop, weight init, progressive unfreezing
- `cloud/modal_train.py` — temporal args passthrough
- `pipelines/remaster.py` — temporal inference with prev_output buffer
- `tools/export_onnx.py` — auto-detect in_nc from checkpoint
