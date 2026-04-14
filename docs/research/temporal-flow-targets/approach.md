# Temporal Flow-Warped Targets: Does It Actually Help?

## The Core Question

"I still don't understand how we generate a sharper frame even if we have the RAFT
movements between frames?"

This is the right question, and the honest answer is: **flow warping alone does NOT
add sharpness. It moves pixels around.** What matters is WHY we're doing it and what
we expect to gain. There are three distinct things people conflate:

### 1. Multi-Frame Super-Resolution (adds real detail)

Burst photography and multi-frame SR exploit sub-pixel motion between frames. If an
object shifts by 0.3 pixels between frames, those two observations sample different
points on the continuous scene. Combining them recovers spatial frequencies above the
Nyquist limit of any single frame.

**Does this apply to us? Mostly no.** Our content is 24fps 1080p. At 24fps with
typical camera/object motion, inter-frame displacement is usually 5-50 pixels, not
sub-pixel. The sub-pixel information exists only along object edges with very slow
motion, and it's dwarfed by HEVC quantization noise. Multi-frame SR works for burst
photography (handheld shake = sub-pixel motion, 10-30 frames in 100ms) and satellite
imagery, not for 24fps cinematic video.

**Verdict: Not a practical source of sharpness for our use case.**

### 2. Noise Averaging (reduces artifacts, does not add detail)

If SCUNet produces slightly different outputs for adjacent frames (different noise
patterns, slightly different artifact suppression decisions), averaging N aligned
frames reduces noise by sqrt(N). With 5 frames, that's a 2.2x noise reduction.

This is real and measurable. But it makes things *smoother*, not *sharper*. The PSNR
goes up because noise power drops, but no new detail appears. In fact, any flow
estimation error introduces blur at motion boundaries.

**Verdict: Reduces flicker and residual artifacts. Does not add sharpness. May
slightly blur motion edges.**

### 3. Temporal Consistency (the actual win)

This is what matters for our training targets. The student model learns a mapping
from input to target. If the target for frame N and frame N+1 show the same static
region with slightly different brightness/color/texture (because SCUNet made
independent per-frame decisions), the student learns that these variations are
acceptable. The student then reproduces similar random variations at inference,
causing visible flicker.

If we make the targets consistent -- same static region maps to same output values
across frames -- the student learns a stable, deterministic mapping. The student's
output becomes temporally smooth without any explicit temporal mechanism at inference.

**Verdict: This is the real benefit. Not sharper frames, but frames where the student
learns that static regions should produce identical output.**

## Honest Assessment: What Flow Warping Gives Us

| Effect | Magnitude | Sharpness? | Worth it? |
|--------|-----------|------------|-----------|
| Sub-pixel SR | Negligible at 24fps | Yes (theoretical) | No |
| Noise averaging | sqrt(N) reduction | No (smoother) | Maybe |
| Consistency | Eliminates target flicker | No (same quality) | **Yes** |
| Flow errors at edges | Blur at motion boundaries | Negative | Risk |

**The calculus:** We gain temporal consistency (valuable for student training) at the
cost of slight blur at motion edges (harmful). The question is whether the consistency
benefit outweighs the edge-blur cost.

## Practical Algorithm

### Backward Warping with Occlusion Masking

Backward warping is standard because it's trivially parallelizable on GPU via
`grid_sample`. Forward warping creates holes and requires scatter operations.

```
For each target frame t:
  1. Compute flow: flow_{t-1->t} and flow_{t+1->t}  (backward flow into frame t)
  2. Warp: warped_{t-1} = grid_sample(frame_{t-1}, flow_{t-1->t})
           warped_{t+1} = grid_sample(frame_{t+1}, flow_{t+1->t})
  3. Compute occlusion masks via forward-backward consistency check
  4. Blend: output_t = weighted_average(frame_t, warped_{t-1}, warped_{t+1})
            using occlusion masks as weights
```

### Occlusion Detection

The RAFT_bi code and Upscale-A-Video's `propagation_module.py` already implement
forward-backward consistency checking (`fbConsistencyCheck`):

```python
# From propagation_module.py (already in our codebase)
def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))
    flow_diff = flow_fw + flow_bw_warped
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw_warped)
    occ_thresh = alpha1 * mag_sq + alpha2
    valid = (length_sq(flow_diff) < occ_thresh).float()
    return valid  # 1 = consistent (not occluded), 0 = occluded
```

Pixels where forward and backward flow disagree are likely occluded (appeared or
disappeared between frames). These get zero weight in the blend.

### Blending Strategy

Conservative temporal averaging that preserves the current frame in occluded regions:

```python
def blend_temporal(frame_t, warped_prev, warped_next, mask_prev, mask_next):
    # mask values: 1 = valid (not occluded), 0 = occluded
    w_prev = mask_prev * 0.25  # 25% weight for valid previous
    w_next = mask_next * 0.25  # 25% weight for valid next
    w_curr = 1.0 - w_prev - w_next  # remainder (50-100%) for current

    output = w_curr * frame_t + w_prev * warped_prev + w_next * warped_next
    return output
```

This is deliberately conservative: the current SCUNet output keeps at least 50%
weight everywhere, and 100% weight in occluded regions. We're smoothing noise, not
replacing content.

### Handling Edge Cases

- **First/last frame in sequence:** Only one neighbor available. Use 75/25 blend
  with the single neighbor, 100% current for occluded pixels.
- **Scene cuts:** Large flow magnitude (>100px mean displacement) indicates a cut.
  Skip blending entirely, keep original SCUNet output.
- **Isolated training frames:** Our training data samples 1 frame per 500. These
  frames have no temporal neighbors. They must be skipped by this pipeline.

## The Data Problem

**Our current training frames are NOT sequential.** They're sampled 1/500 = ~21
seconds apart. There are no temporal neighbors to warp.

To use flow-warped targets, we'd need to either:

1. **Extract new sequential clips** (e.g., 10-second clips at 24fps, ~240 frames
   each) and process those as groups. This requires re-running SCUNet on sequential
   frames, not just isolated samples.

2. **Use the existing nerv-test clips** (clip_01 and clip_02, 240 frames each at
   1080p) as a proof-of-concept before committing to full re-extraction.

Option 2 is the right first step.

## Proof-of-Concept Pipeline

Process the existing nerv-test clips (480 sequential frames total):

```
1. Run SCUNet GAN + USM on all 480 frames -> per-frame targets
2. Run RAFT (raft-things.pth) on per-frame targets:
   - Forward flow: frame_t -> frame_{t+1} for all consecutive pairs
   - Backward flow: frame_{t+1} -> frame_t for all consecutive pairs
3. Compute occlusion masks via fbConsistencyCheck
4. Blend each frame with its warped neighbors (conservative 50/25/25)
5. Compare: per-frame SCUNet vs flow-blended, visually and via metrics
```

### What to Measure

- **Temporal consistency:** Pixel variance across frames in static regions. Lower =
  more consistent. Compare SCUNet-only vs flow-blended.
- **Sharpness:** Laplacian variance or edge strength. Should NOT decrease
  significantly. If it does, the flow errors are blurring edges.
- **Visual inspection:** Play both versions side-by-side. Does the flow-blended
  version eliminate flicker without introducing new artifacts?
- **PSNR vs original:** Not very meaningful here (SCUNet changes the image
  intentionally), but useful as a sanity check that we haven't destroyed the output.

## Using RAFT from Our Codebase

The RAFT model at `reference-code/Upscale-A-Video/models_video/RAFT/` expects:

```python
# Input: [B, 3, H, W] float tensors, 0-255 range, RGB
# H, W must be divisible by 8 (use InputPadder)
# Output: (flow_low, flow_up) where flow_up is [B, 2, H, W]

from reference_code_upscale_a_video.models_video.RAFT.raft_bi import initialize_RAFT

model = initialize_RAFT('raft-things.pth', device='cuda')
# model expects DataParallel-wrapped weights

padder = InputPadder(image1.shape)
image1, image2 = padder.pad(image1, image2)
_, flow = model(image1, image2, iters=20, test_mode=True)
flow = padder.unpad(flow)
```

Weights: Download `raft-things.pth` from the RAFT GitHub releases
(https://github.com/princeton-vl/RAFT). ~20MB.

At 1080p, RAFT needs ~4GB VRAM for a single pair. Process pairs sequentially or on
Modal with an A10G/L40S.

## Cost Estimate

### Proof-of-concept (480 frames, nerv-test clips)

- SCUNet on 480 frames: ~480 / 5 fps = 96 seconds on L40S
- RAFT on 479 pairs (fwd + bwd): ~479 * 2 / 10 fps = 96 seconds on L40S
- Warping + blending: ~10 seconds (trivial GPU ops)
- **Total: ~4 minutes on L40S = ~$0.13**

### Full production (if PoC succeeds)

If we decide to re-extract sequential training data:
- Need ~30 clips of 240 frames each = 7,200 frames (similar to current 7K)
- SCUNet: 7200 / 5 = 1440 seconds = 24 minutes
- RAFT: 7200 * 2 / 10 = 1440 seconds = 24 minutes
- Blending: negligible
- **Total: ~50 minutes on L40S = ~$1.63**

## What Would Need to Change in build_training_data.py

### New Stage: --temporal-blend (between --denoise and --build-inputs)

```
python tools/build_training_data.py --extract-only    # existing
python tools/build_training_data.py --denoise         # existing (SCUNet)
python tools/build_training_data.py --temporal-blend   # NEW
python tools/build_training_data.py --build-inputs    # existing
```

But this only works if we change extraction to produce sequential clips instead of
random isolated frames. The current 1/500 sampling has no temporal neighbors.

### Required Changes to Extraction

1. Instead of sampling 1 frame per 500, sample **sequential clips**: pick N random
   start points in each episode, extract K consecutive frames at each.
   Example: 30 clips x 240 frames = 7,200 frames total.

2. After SCUNet denoising, run RAFT + temporal blending on each clip as a group.

3. Then sample individual frames from the blended clips as training targets.

This is a significant change to the data pipeline, which is why the PoC on
nerv-test clips should come first.

## Alternative: Temporal Loss Instead of Temporal Targets

The existing `docs/research/raft-alignment/findings.md` mentions Option C: add a
temporal consistency loss during training. This doesn't change the targets at all.
Instead, during training, we run RAFT on the student's output for consecutive frames
and penalize inconsistency:

```
L_temporal = |warp(student(frame_t), flow_{t->t+1}) - student(frame_{t+1})| * mask
```

This trains the student to produce consistent output directly, without changing
targets. It requires sequential training data and RAFT during training (slow), but
avoids the entire target-modification pipeline.

**This might be simpler and more effective than flow-warped targets.** The student
learns consistency as an explicit objective rather than implicitly from consistent
targets.

## Recommendation

1. **Do the PoC first** on nerv-test clips (480 frames, ~$0.13 on Modal). Visual
   comparison will immediately show whether flow blending helps or just blurs.

2. **If PoC shows consistency gain without blur:** Redesign extraction for sequential
   clips, run full pipeline.

3. **If PoC shows noticeable blur at motion edges:** Flow-warped targets are
   counterproductive. Consider temporal loss during training instead (Option C), or
   the recurrent 9-channel architecture (already designed in temporal-context/prd.md).

4. **Don't expect sharpness gains.** The realistic outcome is: same per-frame quality
   but more consistent across time. That's still valuable for student training, but
   it won't make individual frames look better.
