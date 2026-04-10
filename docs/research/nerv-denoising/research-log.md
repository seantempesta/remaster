# NeRV Denoising Research Log

Agent-maintained log of experiments, findings, and decisions. Each entry documents what was tried, what was learned, and what to try next. This prevents future agents from repeating failed experiments.

## How to use this log

**After each experiment**, append an entry using this format:

```
### ExpNN: <short title> (YYYY-MM-DD HH:MM)
- **Hypothesis**: What you expected to happen and why
- **Change**: What you modified (file, function, parameter)
- **Result**: val_psnr=XX.X, train_psnr=XX.X, hf_ratio=X.XX, vram=X.XGB
- **Verdict**: keep / discard / crash
- **Learning**: What this tells us about the problem. What to try next.
```

Keep entries concise — 3-5 lines max. The detail is in the git diff and W&B run.

## Key constraints (quick reference)
- **Current best: val_psnr=34.50, hf_ratio=0.27, 4.70M params (exp22/23)**
- VRAM limit: 5.5GB (6GB card, need headroom)
- Time budget: 20 min per run (--max-time 1200)
- Confirmed ceiling at 34.50 dB with current architecture — need structural changes to break through

---

## Experiment Log

*(Entries below are appended by autonomous agents)*

### Human guidance (2026-04-09 20:50)
- If 10 minutes isn't enough per experiment, agents can increase --max-time (up to 1200 = 20 min)
- **Idea worth testing**: Fewer frames (e.g., 8 or 4 instead of 16) with faster convergence to sharper output. Maybe the bottleneck isn't model capacity but fitting too many frames — fewer frames means the model can overfit to structure more quickly.
- The goal is sharp, usable denoising — not just higher PSNR. HF ratio matters as much as PSNR.

### Orchestrator summary: Experiments 01-11 (2026-04-09 21:00-23:30)

**Key findings from first agent run:**

1. **Encoder bottleneck width does NOT help** (exp02): enc_dim=64 (4x wider) made val WORSE (28.34 vs 29.64 baseline). More capacity = more noise passes through. The plateau is NOT caused by too few encoder channels.

2. **Frequency loss is essential as regularizer** (exp03): Pure L1 loss overfits badly (val 27.22). The FFT frequency loss in l1_freq prevents the model from memorizing high-frequency noise.

3. **Fusion6/SSIM losses are broken** (exp05, exp06): Both collapsed regardless of optimizer (Prodigy or Adam). SSIM-based losses diverge with this architecture. Don't try again.

4. **3x3 decoder kernels are THE breakthrough** (exp07, exp08): Increasing minimum decoder kernel size to 3x3 (from 1x1) gave +3.3 dB. Works with both stride patterns. This is the single biggest improvement found.

5. **332 strides help modestly** (exp04 vs exp08): Strides 3,3,2,2,2 (72x downsample) vs 5,3,2,2,2 (120x) contribute ~0.6 dB. Worth combining with 3x3 kernels but not the main driver.

6. **Weight decay 0.01 + longer training = new best** (exp09): Adding weight decay delays overfitting enough to reach 33.88 dB in 15 min. Val peaked later and higher.

7. **Extra decoder blocks overfit** (exp10, exp11): dec_blks 1,1,3,3,3 adds capacity but overfits severely (val 27 vs 33.88). With 332 strides it OOMs (5.9GB). Not worth pursuing unless combined with stronger regularization.

**Current best config (exp09, 33.88 dB):**
- 3x3 min decoder kernels + 3x3 head (commit 2d4bc02)
- Strides 3,3,2,2,2
- Weight decay 0.01
- 15 min training (--max-time 900)
- 4.70M params, 4.7GB VRAM

8. **Gradient checkpointing enables larger models** (exp12): dec_blks 1,1,3,3,3 + 332 OOMed at 5.9GB (exp10) but fits at 4.1GB with --grad-checkpoint. Val 33.32, decent but didn't beat exp09.

9. **20 min training beats 15 min** (exp13): Same exp09 config with 20 min budget reached 34.32 dB (vs 33.88). The key was extra epochs 105-140 where cosine LR drops low and the model refines fine details. HF ratio jumped to 0.31 — the sharpest output yet.

10. **Val PSNR oscillates with weight decay** (exp13): Val is NOT monotonically declining. It dips mid-training (32.58 at epoch 99) then recovers (33.99 at epoch 117). This means "peak val" should be tracked throughout, not just at the end.

**Current best (exp13): 34.32 dB, hf_ratio=0.31, 4.70M params, 140 epochs in 20 min**

### Orchestrator summary: Experiments 12-23 (2026-04-09 23:30 - 2026-04-10 03:00)

11. **Gradient checkpointing works** (exp12): dec_blks 1,1,3,3,3 + 332 fits at 4.1GB with --grad-checkpoint. Val 33.32, decent but didn't beat exp09's simpler config.

12. **20 min training is the sweet spot** (exp13): Same config as exp09 with 20 min budget reached 34.32 dB. Extra epochs 105-140 with dropping cosine LR is where quality refines. HF ratio jumped to 0.31.

13. **pixel_weight=5 is marginal** (exp14): Reducing L1 weight from 10 to 5 in l1_freq gives 33.95 — slightly worse than default 10. Not worth changing.

14. **Adam lr=0.002 is unstable** (exp15): Val oscillates wildly (32.09 -> 31.16 -> 32.43 -> 33.24). Prodigy is smoother and reaches higher peaks. Adam val ceiling: 33.49 vs Prodigy's 34.32.

15. **L2/MSE loss overfits severely** (exp16): Val 29.05 with 8 dB train-val gap. The l1_freq loss is CRITICAL — its frequency component prevents the model from memorizing noise.

16. **Prodigy d_coef=2.0 overfits slightly more** (exp17): Val 33.77, below d_coef=1.0's 34.32. Higher LR causes earlier overfitting. Default d_coef=1.0 is optimal.

17. **Deeper encoder (2 blocks/stage) is competitive** (exp18): Val 34.08 with very tight train-val gap (0.11 dB early on). Adds 0.15M params, uses 5.1GB VRAM. Good generalization but slower (9.3s/epoch).

18. **weight_decay=0.005 overfits more** (exp19): Val 33.10 vs 34.32 with wd=0.01. The 0.01 value is well-calibrated.

19. **30 min training does NOT help** (exp20): Deeper encoder + 30 min gave 32.78 — overfitting past 20 min even with weight decay.

20. **Smaller model (fc_dim=80, 2.36M params) peaks lower** (exp21): Val 33.44. The 4.70M model (fc_dim=120) is the right size.

21. **Cosine schedule matching is the last +0.2 dB** (exp22/23): Setting --epochs to match actual training epochs (~150 instead of 300) makes the cosine LR reach near-zero at the right time. Two independent runs both hit 34.50 dB — this is the **confirmed architectural ceiling**.

---

## Ceiling Analysis (2026-04-10)

**The 34.50 dB ceiling is confirmed.** All hyperparameter variations (optimizer, LR, weight decay, loss balance, training time, model size) converge to 33.5-34.5 dB. The next improvement requires STRUCTURAL changes:

### Untested directions that could break the ceiling:
1. **U-Net skip connections** — let high-frequency detail bypass the bottleneck
2. **Fewer frames** (4-8 instead of 16) — more capacity per frame = sharper
3. **Decoder residual connections** — learn corrections instead of full reconstruction
4. **Perceptual/edge loss terms** — Sobel/Laplacian emphasis on edges
5. **Sin activation** — periodic functions represent fine detail better (Boosting-NeRV uses this)
6. **Data augmentation** — random flips/crops for better regularization

### Exhausted directions (DO NOT re-try):
- Encoder width, stride patterns, depth (all tested)
- Loss functions: L1, L2, l1_freq, fusion6, SSIM (all tested)
- Optimizers: Adam, Prodigy with various d_coef (all tested)
- Model sizes: 2.36M to 6.70M (all tested)
- Training time: 10, 15, 20, 30 min (all tested)
- Weight decay: 0.005, 0.01 (both tested)

