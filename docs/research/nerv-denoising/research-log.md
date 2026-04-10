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
- **VRAM limit: 5.5GB target, 5.7GB absolute max** (6GB card but OS uses ~300MB)
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

### Human guidance (2026-04-10 09:30) — Skip connections are replicating noise
- exp24 skip connections produce SHARP output BUT the residual shows too much structure (faces, clothing visible in residual at epoch 59). This means the model is memorizing noise, not just removing it.
- **The goal is REMASTER quality: sharp + denoised.** The network learns features across frames to reconstruct realistic detail that was lost to compression. The residual should be mostly random noise — no structure.
- Skip connections help sharpness but also let noise bypass the bottleneck. Need to find the balance:
  - Maybe skip connections with REDUCED weight (learnable scale factor, initialize to 0.1?)
  - Or skip connections only at lower-resolution scales (not the highest-res ones where noise lives)
  - Or add dropout/noise to the skip path to prevent noise passthrough
  - Or use attention-gated skip connections that learn what to pass vs block
  - Or soft-thresholding on skip features: small values (noise) -> 0, large values (edges) -> pass through
  - Or channel-wise attention gate (squeeze-and-excite style) that learns which channels carry signal vs noise
- **Simplest approach: learnable scale per skip, init to 0.01** — `out = decoder_out + alpha * skip_features` where alpha is a nn.Parameter starting near zero. The model must learn to use skip info gradually, preventing noise blowthrough early in training.
- **Check the residual image after every run** — if structure is visible, the model is memorizing noise. A good denoiser's residual is pure noise.
- **Residual loss idea**: Add a loss term that penalizes non-noise structure in the residual (input - output). Pure noise has: flat FFT spectrum, no spatial autocorrelation, Gaussian-like distribution. Structure has: peaks in FFT, high autocorrelation, heavy tails. Could use FFT magnitude of the residual as a penalty — flat spectrum = noise, peaked spectrum = structure being removed.
- **Read the reference denoising code** for ideas: `reference-code/SCUNet/`, `reference-code/KAIR/`, `reference-code/DISTS/` — these are proven denoisers. How do they handle the noise vs structure separation?

### Human guidance (2026-04-10 09:00) — VRAM and skip connections
- exp24 (skip connections) uses **5.8GB VRAM** — OVER the limit. Only **5.7GB is actually usable** (OS/display takes ~300MB). At 5.8GB we are 0.2GB into shared system RAM = PCIe spill = degraded speed.
- **Hard VRAM ceiling: 5.5GB target, 5.7GB absolute max.**
- The visual output is SHARP — skip connections are clearly working
- **Priority: keep skip connections but reduce VRAM.** Options:
  - Drop dec_blks back to 1,1,1,1,1 (skip connections alone may be enough without extra decoder blocks)
  - Reduce fc_dim from 120 to 100
  - Use gradient checkpointing on the skip path
- If a promising run emerges, train it LONGER (30 min, or resume from checkpoint)

### Human guidance (2026-04-10 08:30)
- The W&B visual output from exp24 (skip connections) looks SHARP — much better than previous runs visually
- **If a run is promising, train it LONGER** — 30 min, or resume from checkpoint and continue
- The script saves checkpoints — use `--resume` to continue training from where a previous run left off
- Match --epochs to actual expected epochs so cosine schedule lands properly
- We need clear signal that this approach works. Sharp + denoised output is the goal.

### Orchestrator observation (2026-04-10 13:30) — Flatness loss creates stable but flat plateau
Exp29 with residual flatness loss 0.5: val reaches 34.7 at epoch 45 and holds flat — no decline (unlike without flatness where val peaks then drops). But it also doesn't climb further. The flatness loss prevents overfitting but also prevents improvement.

**Root cause**: Flatness loss can't distinguish noise HF from edge HF. Both show up in the residual FFT. So the model keeps output smooth (low HF) to minimize flatness penalty — but this means edges end up in the residual too.

**Solution**: Combine flatness loss with edge preservation loss. Flatness says "residual = noise," edge loss says "but don't put edges there." Together they break the plateau by giving the model a clear signal for which HF to keep vs remove.

**Next experiment MUST try**: Combined loss with two complementary residual signals:

1. **Residual structure penalty** (minimize): Sobel/Laplacian energy of `(input - output)`. Penalizes edges/lines/geometry in the residual. This keeps sharpness in the output.

2. **Output sharpness reward** (maximize): Sobel energy of the output should be >= Sobel energy of the input. The output should be SHARPER than the noisy input, not smoother.

These are complementary:
- Structure penalty says "don't put edges in the residual"  
- Sharpness reward says "the output should have MORE edge energy than the input"
- Together: edges stay in output AND get enhanced, while noise goes to residual

Implementation: `loss = reconstruction + flatness * residual_spectral_flatness + edge * residual_sobel_energy - sharpness * output_sobel_energy`

The sharpness term is NEGATIVE (we maximize it). Start with small weights and tune.

**Key insight from human**: "focusing on the residual (no lines/structure) AND increasing sharpness from the original" — these are complementary signals, not competing ones. Find the balance.

### Human guidance (2026-04-10 15:30) — New validation metrics needed
val_psnr measures distance from noisy input — but we WANT to differ from the noisy input. If the model enhances beyond the input, PSNR goes DOWN even though quality goes UP.

**New metrics to add to training script and log:**
1. **output_sharpness**: `sobel(output).abs().mean()` — higher = sharper
2. **input_sharpness**: `sobel(input).abs().mean()` — reference
3. **sharpness_ratio**: `output_sharpness / input_sharpness` — >1.0 = sharper than input (GOAL)
4. **residual_flatness**: spectral flatness of `(input - output)` — higher = more noise-like residual
5. **residual_edge_energy**: `sobel(input - output).abs().mean()` — lower = edges kept in output

These should be logged in metrics.jsonl alongside PSNR so we can track them during training. The agent should add these to the validation loop in train_nerv.py.

**Measure these in BOTH training and validation loops** so we can see the same metrics everywhere.

### Human insight (2026-04-10 15:00) — More frames, not fewer + asymmetric residual loss
Exp30 (8 frames) peaked at 33.3 dB — much worse than 16 frames (35+). Confirmed: MORE temporal context is better. The shared weights need diverse frames to learn the scene's visual vocabulary. Don't reduce frames.

**The winning combination:**
1. 16+ frames (temporal context)
2. Skip connections with scale=0.1 (sharpness)
3. Asymmetric residual loss: `relu(input - output)` for flatness/structure penalty — only penalize REMOVED content, not ADDED detail. This way the model can be sharper than the input without the loss fighting it.

**Key: `relu(input - output)` instead of `(input - output)` for the residual-based losses.** This one change enables enhancement while still penalizing noise memorization.

### Human insight (2026-04-10 14:00) — Residual loss penalizes sharpness enhancement
If output is sharper than input, `residual = input - output` has "anti-edges" (inverted structure). The flatness loss sees this as structure and penalizes it — but it's actually GOOD (enhancement).

**Fix options:**
1. Use `relu(input - output)` for flatness loss — only penalize REMOVED content, not ADDED detail
2. Decompose: separate noise component (lowpass of residual) from enhancement (highpass of output - highpass of input). Only penalize noise component.
3. Use absolute residual with asymmetric weighting

This is important — without this fix, the flatness loss actively fights against sharpness enhancement.

### Human insight (2026-04-10 13:00) — Minimize edge energy in residual
Looking at the residual image: straight lines and geometry visible = edges being removed from output into residual. We want edges to STAY in the output.

**Residual edge loss**: `edge_loss = sobel(input - output).abs().mean()` — directly penalizes edges/structure in the residual. Different from spectral flatness: flatness measures overall frequency distribution, edge loss specifically targets visible geometry (lines, contours, hair outlines).

Can combine: `total = reconstruction_loss + flatness_weight * spectral_flatness + edge_weight * residual_edge_loss`

### Human insight (2026-04-10 12:30) — Need explicit sharpness signal
There's no loss that rewards the model for producing SHARPER output than the input. The temporal knowledge is in the shared weights but the loss doesn't exploit it.

**Sharpness-aware losses to try:**

1. **Edge preservation loss**: `edge_loss = relu(sobel(input) - sobel(output)).mean()` — penalize output edges being weaker than input edges. Encourages the model to preserve and enhance edges while removing inter-edge noise.

2. **Temporal median as soft target**: Compute `median(output[t-1], output[t], output[t+1])` as a denoised reference. Use as an auxiliary loss target at low weight. The median of multiple outputs is cleaner than any single one.

3. **Laplacian sharpness reward**: Maximize the Laplacian energy of the output: `sharpness = laplacian(output).abs().mean()`. This directly rewards edge sharpness. Must balance with reconstruction loss to avoid artifacts.

4. **Combined approach**: `total_loss = l1_freq + residual_flatness + edge_preservation`. Each loss targets a different aspect: reconstruct the content, remove noise, keep edges sharp.

### Human insight (2026-04-10 12:00) — Loss function is fundamentally wrong
The current loss trains the model to reconstruct the NOISY input. A "perfect" model would reproduce noise perfectly. We need losses that reward being BETTER than the input.

**Loss directions to try (in priority order):**

1. **Residual spectral flatness loss**: `residual = input - output`, then penalize non-flat FFT spectrum of the residual. Pure noise has flat spectrum; structure has peaks. `flatness = geometric_mean(|fft(residual)|) / arithmetic_mean(|fft(residual)|)` — maximize this.

2. **Temporal consistency loss**: Adjacent frames show the same scene. Noise is random per-frame but structure is stable. Penalize `|output[t] - output[t+1]|` for pixels that should be the same.

3. **Asymmetric HF loss**: Only penalize output having MORE HF than input (prevents adding noise), but don't penalize having LESS (allows removing noise). `loss = relu(|highpass(output)| - |highpass(input)|)`.

4. **Total variation on output**: Encourages smooth output. Balance with other losses to control denoising strength.

These can be ADDED to the existing l1_freq loss, not replacing it. Start with #1 at low weight.

### Orchestrator observation (2026-04-10 11:30) — Skip connection sweep results and residual analysis

**Summary of skip connection experiments (exp24-27):**

| Config | Val PSNR | hf_ratio | Gap | VRAM | Residual quality |
|--------|----------|----------|-----|------|-----------------|
| No skip (exp22) | 34.50 | 0.27 | ~0.5 | 4.7 | mostly noise, but output is blurry |
| Scale=0.01 (exp25) | 34.94 | 0.60 | 0.95 | 5.1 | good, some edge ghosting |
| Scale=0.1 (exp26) | 35.20 | 0.74 | 0.78 | 5.2 | good balance |
| Scale=1.0 (exp24) | 36.02 | 0.72 | 1.19 | 5.8 | BAD — faces visible in residual |
| Scale=0.1 + drop=0.3 (exp27) | ~34.75 | ~0.6 | 0.19 | 5.2 | BEST — mostly pure noise |

**Key insight: val_psnr alone is misleading.** Exp24 gets 36 dB but its residual shows structure = it's memorizing noise. Exp27 gets 34.75 dB with the tightest gap (0.19 dB) and cleanest residual = it's actually denoising.

**The real goal is: residual should be pure noise.** We want the model to separate signal from noise, not just reconstruct the noisy input well.

**Next directions to try:**
1. Dropout=0.1 with scale=0.1 (less aggressive dropout, faster convergence)
2. Dropout=0.3 with --epochs=200+ (give the dropout model more time since it converges slower)
3. Scale=0.2 or 0.3 without dropout (between 0.1 and 1.0)
4. Residual loss: penalize non-noise structure in (input - output) via FFT flatness

### Orchestrator observation (2026-04-10 11:00) — Exp27 residual quality is BEST yet
- Exp27 (dropout 0.3 + scale 0.1) at epoch 89: residual is mostly noise, faces barely visible
- Train-val gap at epoch 92: only 0.19 dB — nearly zero overfitting
- Val still climbing at epoch 92 (34.36), has NOT peaked yet
- This model converges slower but keeps climbing without overfitting
- **The dropout is working as a noise filter on the skip path**
- For epoch optimization: with dropout, the model needs MORE epochs than without (peak later). Set --epochs higher (200-250?) to let it converge fully.

### Human guidance (2026-04-10 10:30) — Dropout on skip connections
- **Add dropout on the skip path** (e.g., nn.Dropout2d(p=0.3) applied to skip features before adding to decoder)
- At training time: randomly drops skip features, forcing the bottleneck to encode more info (bottleneck naturally denoises via compression)
- At inference time: all features flow through (scaled), giving full sharpness
- Try p=0.3 and p=0.5. Can combine with skip-scale-init.
- This is a well-understood regularization approach — forces the model to not rely entirely on either path.

### Exp24: U-Net skip connections (2026-04-10 08:22)
- **Hypothesis**: Adding skip connections from encoder to decoder lets high-frequency detail bypass the bottleneck, breaking the 34.50 dB ceiling.
- **Change**: Used --skip-connections flag (already implemented). Best config from exp22/23 + dec_blks 1,1,2,2,2. skip_scale=1.0 (no scaling).
- **Result**: val_psnr=36.02 (PEAK at epoch 84), train_psnr=37.21, hf_ratio=0.72 at epoch 119, vram=5.8GB
- **Verdict**: keep -- BREAKTHROUGH +1.5 dB and 3x hf_ratio
- **Learning**: Skip connections are THE fix for the blurry bottleneck. However: (1) VRAM at 5.8GB is over limit, (2) model overfits badly in late epochs (val drops from 36.02 to 34.70), (3) residual shows structural content = noise memorization. Need: learnable skip scale, VRAM reduction, and/or skip only low-res stages.

### Exp25: Learnable skip scale init=0.01 + smaller decoder (2026-04-10 08:50)
- **Hypothesis**: Starting skip scale near zero forces the model to gradually learn how much encoder info to pass, preventing noise blowthrough. dec_blks=1,1,1,1,1 saves VRAM.
- **Change**: Added --skip-scale-init 0.01 with nn.Parameter per skip. dec_blks=1,1,1,1,1 (from 1,1,2,2,2).
- **Result**: val_psnr=34.94 (peak epoch 70), train_psnr=35.89, hf_ratio=0.605 at epoch 149, vram=5.1GB
- **Verdict**: keep -- VRAM fixed, good hf_ratio (2.2x baseline), but peak PSNR only marginally above ceiling
- **Learning**: 0.01 init is too conservative -- model can't pass enough detail early. The scale does help prevent overfitting (gap 3.58 dB vs exp24's 4.65 dB). Sweet spot is between 0.01 and 1.0. Try 0.1 next. Also try limiting skips to lower-res stages only.

### Exp26: Skip scale=0.1, dec_blks=1,1,1,1,1 (2026-04-10 09:30)
- **Hypothesis**: scale=0.1 is a middle ground between 0.01 (too conservative) and 1.0 (noise passthrough). Smaller decoder saves VRAM.
- **Change**: --skip-scale-init 0.1, dec_blks=1,1,1,1,1
- **Result**: val_psnr=35.20 (peak epoch 58), hf_ratio=0.74 at epoch 149, vram=5.2GB
- **Verdict**: keep -- +0.70 dB over ceiling, 2.7x hf_ratio, VRAM within limits
- **Learning**: 0.1 scale lets more detail through than 0.01 but still overfits in late epochs (gap 4.26 dB). The scale parameter grows during training via optimizer updates.

### Exp27: Skip dropout=0.3 + scale=0.1 (2026-04-10 10:00)
- **Hypothesis**: Dropout2d on skip features forces bottleneck to encode more info during training, reducing noise memorization. At inference, all features flow through for full sharpness.
- **Change**: Added --skip-dropout 0.3 with nn.Dropout2d on skip path
- **Result**: val_psnr=34.93 (peak epoch 121), hf_ratio=0.44, train-val gap=1.53 dB, vram=5.2GB
- **Verdict**: discard -- dropout too aggressive, dampens skip benefits. hf_ratio 0.44 vs exp26's 0.74.
- **Learning**: Dropout2d=0.3 regularizes well (very tight train-val gap) but prevents skip connections from carrying enough high-freq detail. However, the train-val gap control is remarkable -- near zero overfitting. If given more epochs it may keep climbing. The orchestrator noted the residual at epoch 89 looked cleaner than other runs.

### Exhausted directions (DO NOT re-try):
- Encoder width, stride patterns, depth (all tested)
- Loss functions: L1, L2, l1_freq, fusion6, SSIM (all tested)
- Optimizers: Adam, Prodigy with various d_coef (all tested)
- Model sizes: 2.36M to 6.70M (all tested)
- Training time: 10, 15, 20, 30 min (all tested)
- Weight decay: 0.005, 0.01 (both tested)

