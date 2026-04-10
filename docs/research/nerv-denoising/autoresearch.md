# NeRV Denoising Auto-Research

Autonomous experimentation loop for finding the optimal HNeRV configuration for video denoising.

## Context

We're fitting an HNeRV neural video representation to a 16-frame micro-GOP of 1080p Firefly video. The network's spectral bias denoises by learning structure before noise. The goal is to **maximize PSNR on holdout frames while producing visually sharp (non-blurry) output**.

### Current state (after 23 experiments on 2026-04-09)
- Best config: 4.70M params, fc_dim=120, enc_dim=16, 3x3 decoder kernels, 332 strides, wd=0.01
- **Ceiling at 34.50 dB val** — confirmed by two independent runs (exp22, exp23)
- Output is still somewhat blurry — HF ratio only 0.27 (27% of input high-frequency content)
- Previous plateau at 29.64 dB was broken by 3x3 decoder kernels (+3.3 dB) and weight decay (+1.2 dB)
- The current ceiling requires **architectural innovation**, not hyperparameter tuning

### What's been tried (see research-log.md for full details)
**DO NOT re-try these — they've been thoroughly tested:**
- Wider encoder bottleneck (enc_dim=64): WORSE, more noise passes through
- Pure L1 loss: overfits without frequency regularizer
- L2 (MSE) loss: severe overfitting (29.05 dB)
- Fusion6/SSIM losses: collapse with both Prodigy and Adam
- Adam optimizer: less stable than Prodigy for this task
- Higher Prodigy d_coef (2.0): slightly more overfitting
- Extra decoder blocks (dec_blks 1,1,3,3,3): overfit or OOM
- pixel_weight tuning (5.0 vs 10.0): marginal, not worth it
- Smaller model (fc_dim=80): peaks lower
- Longer training (30 min): overfits past ~20 min

**What worked:**
- 3x3 min decoder kernels + 3x3 head: +3.3 dB (THE breakthrough)
- Strides 3,3,2,2,2 (72x downsample): +0.6 dB
- Weight decay 0.01: +1.2 dB by delaying overfitting
- Cosine schedule matched to actual epochs (--epochs 150): +0.2 dB
- Prodigy optimizer with d_coef=1.0: best optimizer tested
- l1_freq loss: critical — the FFT component regularizes against noise memorization

## Setup

### Prerequisites
- Conda env `remaster` (Python 3.12, PyTorch 2.11+cu130)
- RTX 3060 6GB VRAM, 16GB system RAM
- Run all Python with: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/remaster/python.exe`
- Working directory: `C:/Users/sean/src/upscale-experiment`

### FIRST THING YOU DO — understand the problem (before writing any code)

**Step 1: Read the research log**
`docs/research/nerv-denoising/research-log.md` — Contains findings from all previous agents. Without this, you'll waste hours repeating failed experiments.

**Step 2: Review the git history**
`git log --oneline autoresearch/nerv-apr9` — See what code changes were made and which were kept vs reverted.

**Step 3: Check W&B visualizations**
Go through the W&B comparison images and training curves from recent runs. Look at:
- The **residual images** (input - output): Is the residual mostly noise (good) or does it show structure like faces/edges (bad = memorizing noise)?
- The **val_psnr curve**: Does it plateau, oscillate, or crash?
- The **train-val gap**: Large gap = overfitting = memorizing noise

**Step 4: Read reference denoising code for ideas**
- `reference-code/SCUNet/` — state-of-the-art blind denoiser
- `reference-code/KAIR/` — image restoration toolkit (DRUNet, etc.)
- `reference-code/DISTS/` — perceptual similarity metric
- `reference-code/Boosting-NeRV/` — HNeRV architecture with SFT conditioning
Look at how proven denoisers handle noise vs structure separation.

**Step 5: Then formulate your first hypothesis and start experimenting.**

### Files you modify
- **`tools/train_nerv.py`** — training script, model architecture, loss functions. Everything is in this one file.

### Files you MUST update after EVERY experiment (keep, discard, or crash)
- **`output/nerv/autorun/results.tsv`** — metrics table (append one row per experiment)
- **`docs/research/nerv-denoising/research-log.md`** — **MANDATORY**. Append a detailed entry explaining your hypothesis, what happened, and what you learned. This is how the NEXT agent avoids repeating your work. If you skip this, the next agent starts from scratch and wastes hours. Write the *why*, not just the *what*.

### Extending a promising run
If a run is going well and you want it to train longer, write the extra seconds to the extend file:
```bash
echo 600 > output/nerv/autorun/extend_time   # adds 10 min to current run
```
The training script checks this file every epoch and adds the time. No need to kill and restart.

### Files you DO NOT modify
- `training/` — the main DRUNet pipeline, unrelated
- `lib/` — shared libraries, don't change
- `reference-code/` — read for ideas, copy patterns, but don't modify

### Data
- **Training frames**: `E:/upscale-data/nerv-test/micro_gop_01/` (16 frames, 1080p PNG, from Firefly S01E01)
- These are HEVC-compressed Bluray frames — the "noise" is compression artifacts
- The frames stay fixed. Do not modify or regenerate them.

### Branch
1. Create branch: `git checkout -b autoresearch/nerv-<tag>` from current HEAD
2. Create `results.tsv` with header row
3. Establish baseline with the first run

## Running an experiment

```bash
cd C:/Users/sean/src/upscale-experiment
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/remaster/python.exe tools/train_nerv.py \
    --data-dir E:/upscale-data/nerv-test/micro_gop_01 \
    --epochs 300 --max-time 600 --batch-size 1 --print-interval 10 \
    --fc-dim 120 --output-dir output/nerv/autorun \
    --wandb --run-name "exp01-baseline-enc16-fc120" \
    > output/nerv/autorun/run.log 2>&1
```

**W&B run naming**: Always pass `--wandb --run-name "expNN-short-description"` so each experiment gets a descriptive name in W&B. Number experiments sequentially (exp01, exp02, ...) to match the results.tsv order. The human will be checking W&B on their phone to follow progress.

The `--max-time 600` flag ensures the process **self-terminates after 10 minutes** (saves checkpoint and exits cleanly). No need to manually kill. The script will run up to 300 epochs or 10 minutes, whichever comes first. At ~7s/epoch, expect ~85 epochs per run.

**CRITICAL**: Always use `--max-time 600`. Never launch without it. This prevents a runaway process from hogging the GPU if you forget about it.

### Extracting results
```bash
# PSNR (last epoch)
tail -1 output/nerv/autorun/metrics.jsonl | python -c "import sys,json; d=json.load(sys.stdin); print(f'train_psnr={d[\"train_psnr\"]}, val_psnr={d[\"val_psnr\"]}, hf_ratio={d[\"hf_ratio\"]}')"

# Check for NaN
grep -c "NaN" output/nerv/autorun/metrics.jsonl

# Peak VRAM (from W&B or nvidia-smi during run)
```

### Visual check
Look at `output/nerv/autorun/vis/compare_e0149.png` — is the output sharp or blurry? This is as important as the PSNR number.

## Logging results

### results.tsv (metrics table)

Append to `results.tsv` (tab-separated):

```
exp	commit	val_psnr	train_psnr	hf_ratio	vram_gb	epochs	params_M	status	description
```

- `exp`: experiment number (exp01, exp02, ...)
- `commit`: short git hash (7 chars)
- `val_psnr`: holdout PSNR (higher = better, target: 32+)
- `train_psnr`: training PSNR
- `hf_ratio`: high-frequency energy ratio (higher = sharper output, target: 0.4+)
- `vram_gb`: peak VRAM in GB (must stay under 5.5)
- `epochs`: number of epochs completed in the 10-min window
- `params_M`: model parameter count in millions
- `status`: `keep`, `discard`, or `crash`
- `description`: what you tried

Example:
```
exp	commit	val_psnr	train_psnr	hf_ratio	vram_gb	epochs	params_M	status	description
exp01	a1b2c3d	30.0	34.0	0.17	3.4	85	2.83	keep	baseline (enc_dim=16 fc_dim=120)
exp02	b2c3d4e	32.5	36.0	0.35	4.1	72	3.10	keep	wider bottleneck enc_dim=64
exp03	c3d4e5f	0.0	0.0	0.0	0.0	0	5.20	crash	enc_dim=128 OOM
```

### research-log.md (detailed findings)

After EVERY experiment (keep, discard, or crash), append a short entry to `docs/research/nerv-denoising/research-log.md`:

```
### ExpNN: <short title> (YYYY-MM-DD HH:MM)
- **Hypothesis**: What you expected and why
- **Change**: What you modified
- **Result**: val_psnr=XX.X, train_psnr=XX.X, hf_ratio=X.XX, vram=X.XGB
- **Verdict**: keep / discard / crash
- **Learning**: What this tells us. What to try next.
```

This log is critical — it prevents future agents from repeating failed experiments. Before formulating a hypothesis, ALWAYS read the research log first.

## The experiment loop

**LOOP FOREVER:**

1. Read `results.tsv`, `research-log.md`, and recent git log to understand what's been tried
2. Formulate a hypothesis — what change should improve val_psnr or hf_ratio?
3. Edit `tools/train_nerv.py` (architecture, loss, hyperparameters, anything)
4. `git commit -m "experiment: <description>"`
5. Run the experiment (redirect stdout/stderr to run.log — do NOT flood your context)
6. **ACTIVELY MONITOR** the run:
   - Check metrics at epoch 5-10 (~1 min in). If loss is NaN, PSNR below 15 dB, or VRAM above 5.5GB: kill immediately, log as crash, move on.
   - Check again at epoch 20-30 (~3 min in). Is PSNR tracking above/below the baseline curve? This tells you early if the experiment is promising.
   - Don't just sleep for 10 minutes. Read the tail of metrics.jsonl periodically.
7. After the run completes (~10 min), extract final results from `metrics.jsonl`
8. Check the visual output in `output/nerv/autorun/vis/` — is it sharper than the previous best?
9. Record in `results.tsv` AND append to `research-log.md` — **BOTH ARE MANDATORY, EVERY EXPERIMENT, NO EXCEPTIONS**. The research log entry must include your hypothesis, what happened, and what you learned. Without this, the next agent starts from scratch.
10. If val_psnr improved OR hf_ratio improved significantly: **keep** (advance branch)
11. If worse or equal: **discard** (`git reset --hard` to previous commit)
12. **Clean up**: Kill any stale python.exe processes before next run
13. Go to step 1

## What to try (research directions, in priority order)

**The 34.50 dB ceiling is architectural.** Hyperparameter tuning (LR, weight decay, loss weights, d_coef, training time) has been exhausted — all variations land at 33-34.5 dB. Breaking through requires structural changes to how information flows through the network.

### 1. U-Net skip connections (HIGHEST PRIORITY — untested)
The encoder extracts features at multiple scales, but ALL spatial info is crushed to the bottleneck before the decoder sees it. Adding skip connections from encoder to decoder lets high-frequency detail bypass the bottleneck.
- Connect encoder stage outputs to corresponding decoder stages (add, not concatenate — saves VRAM)
- This is the standard fix for "blurry autoencoder" in the literature (U-Net, ResNet, etc.)
- The reference HNeRV architecture does NOT have skips — adding them is a novel modification
- Start simple: just add skips at 2-3 scales, not all 5

### 2. Fewer frames (untested)
- Currently fitting 16 frames. Try 8, 4, or even 2 frames.
- Fewer frames = the model can spend more capacity per-frame = sharper output
- More epochs per frame in the same wall clock time
- May dramatically improve hf_ratio even if val_psnr doesn't change
- To test: create a subdirectory with fewer frames, or add `--num-frames N` flag

### 3. Decoder residual connections (untested)
- Each decoder PixelShuffle block currently has NO residual path
- Adding `output = block(x) + upsample(x)` lets the block learn residual corrections
- This is standard in modern architectures but missing from HNeRV
- Could help with both PSNR and sharpness

### 4. Perceptual / frequency loss tuning (partially tested)
- The l1_freq loss is `10 * L1 + FFT_loss`. The FFT component helps but the balance may not be optimal.
- Try Charbonnier loss (from `training/losses.py`) instead of L1 — smoother gradient near zero
- Try adding a high-frequency EMPHASIS term: extra weight on edges/textures via Sobel or Laplacian filter
- Do NOT try: pure L1, pure L2, SSIM, fusion6 — all confirmed failures

### 5. Activation function (untested)
- Current: GELU in decoder, tanh output head
- Try: SiLU/Swish (similar to GELU but simpler), Sin (Boosting-NeRV uses this)
- Try: sigmoid output head instead of tanh (different output range mapping)
- Sin activation is theoretically interesting for NeRV — periodic activations can represent fine detail better

### 6. Data augmentation (untested)
- Random horizontal flip, random crop during training
- Could regularize better than weight decay alone
- Standard in image reconstruction but not in NeRV papers (since they overfit by design)

## Constraints

- **VRAM**: Must stay under 5.5GB peak. The card is 6GB but only **5.7GB is usable** (OS/display driver takes ~300MB). At 5.7GB+ PyTorch spills into shared system RAM via PCIe — 10x slower and makes the run useless.
- **Time**: Default `--max-time 600` (10 minutes). You may increase to `--max-time 1200` (20 min) if a promising experiment needs more convergence time. The script exits cleanly, no manual kill needed.
- **Simplicity**: Prefer simple changes. A 0.5 dB improvement from one parameter change beats a 0.5 dB improvement from 50 lines of new code.
- **No new dependencies**: Only use what's already installed in the conda env.

## Success criteria

**Current best: val_psnr=34.50, hf_ratio=0.27. Both must improve to break the ceiling.**

The experiment loop succeeds when:
1. **val_psnr > 35 dB** on holdout frames (currently stuck at 34.50)
2. **hf_ratio > 0.40** (currently stuck at 0.27 — need 50% more HF energy for sharp output)
3. **Visual output is noticeably sharper** than the current best in the comparison images

**BOTH PSNR and sharpness matter.** A config that gets 35 dB but hf_ratio=0.20 is worse than one at 34 dB with hf_ratio=0.45. The goal is usable denoised output, not just a good PSNR number on blurry frames.

## Key gotchas

- **Windows cp1252**: Never use unicode arrows/emoji in print() statements
- **VRAM spill**: If VRAM exceeds 6GB, PyTorch spills to shared system RAM via PCIe — this is 10x slower and makes the run useless. Kill immediately.
- **Prodigy + scaled loss**: Prodigy optimizer explodes with losses that have large scaling factors (e.g., 60x in Fusion10_freq). Use Adam with fixed LR for any scaled loss.
- **FFT on FP16**: `torch.fft.fft2` requires FP32 for non-power-of-2 sizes. Always call `.float()` first.
- **Kill stale processes**: Before starting a new run, always check `wmic process where "name='python.exe'" get ProcessId` and kill any stale GPU processes.
- **Check early**: Always verify the first 3-5 epochs of any new config before leaving it unattended. Catches NaN, OOM, and learning rate issues.

## Reference implementations

Read these for architecture ideas (DO NOT import from them — copy patterns into train_nerv.py):
- `reference-code/Boosting-NeRV/model_hnerv.py` — HNeRV with SFT boost
- `reference-code/Boosting-NeRV/model_blocks.py` — UpConv, SFT, ConvNeXt blocks
- `reference-code/Boosting-NeRV/hnerv_utils.py` — loss functions (Fusion variants)
- `reference-code/HiNeRV/` — hierarchical NeRV with trilinear upsampling (no PixelShuffle)
- `reference-code/FFNeRV/model.py` — flow-guided temporal grids

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. Run experiments continuously until interrupted. If you run out of ideas, re-read the reference code, combine previous near-misses, or try more radical changes. The human may be asleep.
