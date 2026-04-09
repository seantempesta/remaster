# NeRV Denoising Auto-Research

Autonomous experimentation loop for finding the optimal HNeRV configuration for video denoising.

## Context

We're fitting an HNeRV neural video representation to a 16-frame micro-GOP of 1080p Firefly video. The network's spectral bias denoises by learning structure before noise. The goal is to **maximize PSNR on holdout frames while producing visually sharp (non-blurry) output**.

### Current state
- Best config: 2.83M params, fc_dim=120, reduce=1.2, enc_dim=16, SFT conditioning
- **Plateaus at ~33-34 dB train / ~30 dB val** regardless of model size, loss, or training duration
- Output is blurry — low HF energy ratio (~0.17 = only 17% of input high-frequency content reconstructed
- Suspected bottleneck: encoder compresses 1080p to only 2,304 values (16ch at 9x16)

### What's been tried (and didn't break the plateau)
- Model sizes from 0.92M to 8.19M params — same ceiling
- SFT temporal conditioning — faster convergence, same ceiling
- L1 loss, L1+FFT frequency loss, Fusion10_freq — same ceiling (Fusion10_freq was unstable)
- Prodigy vs Adam optimizer — same ceiling
- 16 frames vs 240 frames — same ceiling, just faster with fewer frames

## Setup

### Prerequisites
- Conda env `remaster` (Python 3.12, PyTorch 2.11+cu130)
- RTX 3060 6GB VRAM, 16GB system RAM
- Run all Python with: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/remaster/python.exe`
- Working directory: `C:/Users/sean/src/upscale-experiment`

### Files you modify
- **`tools/train_nerv.py`** — training script, model architecture, loss functions. Everything is in this one file.

### Files you DO NOT modify
- `training/` — the main DRUNet pipeline, unrelated
- `lib/` — shared libraries, don't change
- `reference-code/` — read for ideas only

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

Append to `results.tsv` (tab-separated):

```
commit	val_psnr	train_psnr	hf_ratio	vram_gb	status	description
```

- `commit`: short git hash (7 chars)
- `val_psnr`: holdout PSNR (higher = better, target: 32+)
- `train_psnr`: training PSNR
- `hf_ratio`: high-frequency energy ratio (higher = sharper output, target: 0.4+)
- `vram_gb`: peak VRAM in GB (must stay under 5.5)
- `status`: `keep`, `discard`, or `crash`
- `description`: what you tried

Example:
```
commit	val_psnr	train_psnr	hf_ratio	vram_gb	status	description
a1b2c3d	30.0	34.0	0.17	3.4	keep	baseline (enc_dim=16 fc_dim=120)
b2c3d4e	32.5	36.0	0.35	4.1	keep	wider bottleneck enc_dim=64
c3d4e5f	0.0	0.0	0.0	0.0	crash	enc_dim=128 OOM
```

## The experiment loop

**LOOP FOREVER:**

1. Read the current `results.tsv` and git log to understand what's been tried
2. Formulate a hypothesis — what change should improve val_psnr or hf_ratio?
3. Edit `tools/train_nerv.py` (architecture, loss, hyperparameters, anything)
4. `git commit -m "experiment: <description>"`
5. Run the experiment (redirect output, don't flood context)
6. After ~18 min, extract results from `metrics.jsonl`
7. **Early termination**: Check metrics at epoch 10-20 (~2 min in). If loss is NaN, PSNR is below 15 dB, or VRAM exceeded 5.5GB, kill immediately and treat as crash.
8. Check the visual output — is it sharper than the previous best?
9. Record in `results.tsv`
10. If val_psnr improved OR hf_ratio improved significantly: **keep** (advance branch)
11. If worse or equal: **discard** (git reset to previous commit)
12. Go to step 1

## What to try (research directions, in priority order)

### 1. Wider encoder bottleneck (HIGHEST PRIORITY)
The encoder compresses 1080p to `enc_dim` channels at 9x16. Current: 16ch = 2,304 values.
- Try `enc_dim=32` (4,608 values) — does val_psnr exceed 30 dB?
- Try `enc_dim=64` (9,216 values) — does it exceed 32 dB?
- Try `enc_dim=128` — might OOM, test carefully
- If wider bottleneck helps, find the sweet spot between sharpness and VRAM

### 2. Encoder depth and architecture
- More ConvNeXt blocks per encoder stage (`blocks_per_stage=2`)
- Different encoder strides (less aggressive downsampling = more spatial info)
- Try stride pattern `[3,3,2,2,2]` instead of `[5,3,2,2,2]` (72x total vs 120x)
- Consider adding skip connections from encoder to decoder (U-Net style) — this is a bigger change but the research suggests it's critical for sharp output

### 3. Loss function experiments
- L2 (MSE) instead of l1_freq — the reference HNeRV uses L2
- Lower frequency loss weight: change `10 * l1 + freq_loss` to `1 * l1 + freq_loss`
- SSIM loss component (already available via pytorch_msssim)
- Try Charbonnier loss (available in `training/losses.py`)
- Remove frequency loss entirely and just use L1 — isolate whether the FFT loss helps

### 4. Decoder experiments
- More decoder blocks at high resolution: dec_blks `[1,1,3,3,3]`
- Residual connections within decoder blocks
- Change PixelShuffle to bilinear upsample + conv (fixes grid artifacts in FFT)
- Larger decoder kernels at final stages

### 5. Training dynamics
- Higher learning rate (Adam lr=0.003 matching Adan reference)
- Different warmup schedules
- OneCycleLR instead of cosine
- Gradient clipping (the reference uses clip_max_norm)

### 6. Activation function
- Current: GELU in decoder, tanh output head
- Try: SiLU/Swish, Sin (the Boosting-NeRV reference uses Sin activation)
- Try: sigmoid output head instead of tanh (we used sigmoid in earlier runs)

## Constraints

- **VRAM**: Must stay under 5.5GB peak. 6GB is the hard limit but leave headroom for OS/display.
- **Time**: Each run uses `--max-time 600` (10 minutes). The script exits cleanly, no manual kill needed.
- **Simplicity**: Prefer simple changes. A 0.5 dB improvement from one parameter change beats a 0.5 dB improvement from 50 lines of new code.
- **No new dependencies**: Only use what's already installed in the conda env.

## Success criteria

The experiment loop succeeds when:
1. **val_psnr > 32 dB** on holdout frames (currently stuck at ~30 dB)
2. **hf_ratio > 0.4** (currently stuck at ~0.17 — need 2x more HF energy)
3. **Visual output is noticeably sharper** than the baseline in the comparison images

Any ONE of these would be a significant advance. All three together would validate NeRV denoising as a viable approach for generating training targets.

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
