# NeRV Denoising - Implementation Status

## Status: CONCLUDED (2026-04-10)

NeRV-based video denoising was explored over ~45 experiments across local RTX 3060 and Modal T4 GPUs. The approach does not produce usable denoised output for HEVC-compressed video.

### Key Finding
NeRV architecture with pixel-level losses (L1, FFT, etc.) memorizes per-frame noise at all model sizes tested (1.35M-4.44M params). The spectral bias denoising hypothesis does not hold for structured HEVC artifacts with this architecture.

Frame2Frame (Noise2Noise with adjacent frames) successfully prevented noise memorization but produced blurry output (sharpness 0.64) due to inter-frame motion. Motion compensation (RAFT) would fix this but adds significant complexity.

### What Was Built
- `tools/train_nerv.py` — Full HNeRV training script with:
  - U-Net skip connections with learnable scales
  - 3x3 decoder kernels (significant quality improvement)
  - Stride alignment padding
  - Patch-level color preservation loss
  - Edge preservation and asymmetric residual losses
  - Frame2Frame (Noise2Noise) training mode
  - ITS (Iterative Target Substitution)
  - Gradual loss weight ramp
  - Sharpness/residual metrics + 4-panel loss visualization
  - W&B integration, Modal wrapper
- `cloud/modal_train_nerv.py` — Modal training wrapper with resume support
- `docs/research/nerv-denoising/research-log.md` — Detailed log of all 45+ experiments
- `docs/research/nerv-denoising/modal-scaling-plan.md` — Modal GPU scaling analysis

### Experiment Results Summary
| Phase | Experiments | Best Val PSNR | Key Finding |
|-------|-------------|---------------|-------------|
| Baseline (exp01-06) | 6 | 29.64 dB | Encoder bottleneck NOT the issue |
| Architecture (exp07-23) | 17 | 34.50 dB | 3x3 kernels +3.3 dB, skip connections +1.5 dB |
| Skip tuning (exp24-33) | 10 | 36.02 dB | Skip scale=0.1 best balance |
| Loss innovation (exp34-37) | 4 | 42.9 dB | Patch color loss unlocked reconstruction |
| Modal scaling (exp38+) | 5+ | 42.95 dB | 32 frames matches 16, still memorizes |
| Capacity reduction | 2 | 38+ dB | Even 1.35M params memorizes noise |
| Frame2Frame | 1 | 37.1 dB | Denoises but too blurry (sharpness 0.64) |

### Learnings Applicable to DRUNet Temporal Context
See `docs/research/temporal-context/prd.md` — learnings added in "Insights from NeRV" section.

### Related Documents
- Research log: `docs/research/nerv-denoising/research-log.md`
- Original PRD: `docs/research/nerv-denoising/prd.md`
- Findings: `docs/research/nerv-denoising/findings.md`
- Autoresearch instructions: `docs/research/nerv-denoising/autoresearch.md`
- Modal scaling plan: `docs/research/nerv-denoising/modal-scaling-plan.md`
- W&B project: https://wandb.ai/seantempesta/remaster (runs prefixed modal- or exp)
- Git branch: `autoresearch/nerv-apr9`
