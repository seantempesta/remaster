# PRD: Two-Phase Curriculum Training for NeRV Denoising

## Status: PROPOSED (2026-04-10)

## Problem

Frame2Frame (Noise2Noise with adjacent frame targets) successfully prevents noise memorization in NeRV — the residual at epoch 139 is almost pure noise, confirming genuine denoising. However, inter-frame motion causes the output to be blurry (sharpness ratio 0.64).

Same-frame training produces sharp output (sharpness 0.95+) but memorizes noise perfectly (42+ dB, output indistinguishable from noisy input).

**We need both: the denoising of Frame2Frame AND the sharpness of same-frame training.**

## Insight

The Frame2Frame model has learned "what clean looks like" — its weights encode denoised structure. If we resume from this checkpoint with same-frame targets, the model starts from a denoised foundation. It's much harder to re-learn noise from denoised weights than from random init, especially with the right loss pressure.

## Approach: Curriculum Learning

### Phase 1: Denoise (Frame2Frame)

**Goal:** Learn a denoised representation of the video. Output is clean but soft.

- **Targets:** Adjacent frame (`input[t+1]` for frame `t`)
- **Loss:** L1 + FFT (standard `l1_freq`)
- **Architecture:** Full model (4.44M params), skip connections (scale 0.1), skip dropout 0.15
- **Duration:** 150-200 epochs until val plateaus (~36-37 dB against adjacent frame)
- **Outcome:** Model weights encode clean structure. Output is denoised but blurry.

**Key checkpoint:** Save as `phase1_denoised.pth`

### Phase 2: Sharpen (Same-frame, edge-focused)

**Goal:** Add sharp detail to the denoised reconstruction without re-introducing noise.

- **Resume from:** `phase1_denoised.pth` (the denoised model)
- **Targets:** Same frame (`input[t]` for frame `t`) — back to normal
- **Loss changes:**
  - Pixel weight REDUCED: 1.0-3.0 (from 10.0) — loose anchor, not pixel-exact matching
  - Edge preservation weight HIGH: 1.0-2.0 — push for sharp edges
  - Asymmetric residual weight: 0.5 — don't memorize noise patterns
  - Patch color: 5.0 — prevent brightness shift
  - NO FFT loss in phase 2 — FFT matches noise frequencies
- **Optimizer:** Fresh Prodigy (`--fresh-optimizer`) — the denoised model's learned LR is wrong for the new loss landscape
- **Skip dropout:** 0.15 — keeps noise from passing through skips
- **Duration:** 50-100 epochs (shorter — we're fine-tuning, not training from scratch)
- **Monitor:** 
  - Sharpness ratio should climb from 0.64 toward 0.9+
  - Val PSNR will rise (closer to noisy input) but should NOT reach 42+ (that's memorization)
  - Residual should stay mostly noise (check W&B images)
  - If residual shows structure again, reduce pixel weight or increase edge weight

**Key hypothesis:** The denoised weights resist noise memorization. The model's "prior" is clean structure, so when presented with noisy targets at low pixel weight, it will add sharpness (edges, detail) without fully re-memorizing per-frame noise. The edge losses reinforce this by rewarding sharpness over noise reproduction.

## Why This Should Work

1. **Curriculum learning is well-established.** Training easy-to-hard improves convergence and final quality. Denoising is easier than denoising+sharpening simultaneously.

2. **Weight initialization matters.** A model initialized from denoised weights has a fundamentally different loss landscape than one initialized randomly. The denoised solution is a local minimum — the model needs to climb out of it to re-memorize noise, which requires strong gradient signal. With low pixel weight, that signal isn't strong enough.

3. **The Frame2Frame residual proves denoising works.** The epoch 139 residual is almost pure noise — the model genuinely learned to separate signal from noise. This isn't a hypothesis; it's an observed result.

4. **Phase 2 is controlled re-fitting.** We're not asking the model to do something new — we're asking it to sharpen what it already has. The denoised representation is the floor; phase 2 adds detail on top.

## Implementation

### Phase 1 command (already tested, just needs resume support)
```bash
modal run cloud/modal_train_nerv.py \
    --skip-upload --remote-dir nerv-data/clip_02 \
    --frames 32 --batch-size 2 --epochs 200 --max-time 2400 \
    --frame2frame \
    --edge-weight 0.5 --asym-edge-weight 0.5 \
    --run-name nerv-curriculum-phase1
```

### Phase 2 command (resume from Phase 1 checkpoint)
```bash
modal run cloud/modal_train_nerv.py \
    --skip-upload --remote-dir nerv-data/clip_02 \
    --frames 32 --batch-size 2 --epochs 100 --max-time 1800 \
    --resume --fresh-optimizer \
    --pixel-weight 2.0 --edge-weight 1.5 --asym-edge-weight 0.5 \
    --skip-dropout 0.15 \
    --run-name nerv-curriculum-phase2
```

Note: Phase 2 does NOT use `--frame2frame` — it switches back to same-frame targets.

### Required code changes

1. **Disable FFT loss in phase 2.** Add `--no-fft` flag or `--loss l1` (L1 without FFT). The FFT component matches noisy frequency content and fights denoising. Currently `l1_freq` always includes FFT. Options:
   - Add `--loss l1_only` that's L1 + edge + asym + patch color but no FFT
   - Or add `--fft-weight 0.0` to disable the FFT component

2. **Verify resume works on Modal.** The Phase 1 checkpoint must be on the Modal volume and loadable by Phase 2. The agent fixed this but it hasn't been tested end-to-end.

3. **Add sharpness monitoring.** Phase 2 should print a warning if sharpness_ratio drops below 0.60 (the Phase 1 level) — that means the model is diverging instead of sharpening.

## Success Criteria

| Metric | Phase 1 (denoised) | Phase 2 (sharpened) | Same-frame baseline |
|--------|-------------------|--------------------|--------------------|
| Val PSNR (vs noisy input) | 36-37 dB | 37-39 dB | 42+ dB |
| Sharpness ratio | 0.60-0.65 | 0.80-0.95 | 0.95+ |
| Residual | Pure noise | Mostly noise | Shows faces/structure |
| Visual quality | Clean, soft | Clean, sharp | Noisy, sharp |

**Success = Phase 2 output that is BOTH cleaner than the noisy input AND sharper than Phase 1.** Val PSNR between 37-39 (not 42+) with sharpness > 0.80 and a mostly-noise residual.

## Cost Estimate

- Phase 1: T4, ~40 min, ~$0.40
- Phase 2: T4, ~20 min, ~$0.20
- Total: ~$0.60 per full curriculum run
- Budget for 5 iterations: ~$3.00

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase 2 re-memorizes noise despite denoised init | Medium | Wasted phase 2 | Monitor residual — kill if structure appears. Reduce pixel weight further (0.5) |
| Phase 2 output is blurrier than Phase 1 | Low | Regression | Increase edge weight. The edge loss should prevent this |
| Resume from Phase 1 checkpoint fails on Modal | Medium | Can't run phase 2 | Test resume locally first with a small model |
| Motion artifacts in Phase 1 contaminate Phase 2 | Low | Ghosting in output | Phase 2 same-frame targets should correct motion artifacts |
| Prodigy explodes on phase 2 loss landscape | Medium | Training crashes | Use `--fresh-optimizer`. Consider Adam with fixed LR for phase 2 |

## Relationship to Other Work

- **DRUNet temporal context** (`docs/research/temporal-context/prd.md`): If this curriculum approach produces quality denoised+sharpened frames, they could serve as BETTER training targets for DRUNet than SCUNet GAN output. NeRV targets would have temporal consistency built in.
- **NeRV autoresearch** (`docs/research/nerv-denoising/research-log.md`): This builds directly on exp37 (Frame2Frame) findings. All the architecture work (3x3 kernels, skip connections, stride alignment) carries forward.
- **Production pipeline**: If successful, the curriculum approach would need to run per-GOP (300-600 frames). The Modal scaling plan (`docs/research/nerv-denoising/modal-scaling-plan.md`) estimates are still relevant.
