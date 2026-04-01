# Transfer Prompt for Next Conversation

## Read these files first
- `CLAUDE.md` — environment constraints, gotchas, project overview
- `bench/speed-opt/EXPERIMENT.md` — how to run experiments
- `bench/speed-opt/results.tsv` — all experiment results so far
- `bench/speed-opt/research.md` — all research findings (TRT, INT8, compile, Inductor, GPU pricing)
- `cloud/modal_profile.py` — the profiling script
- `lib/nafnet_arch.py` — model architecture (GroupNorm swap, LayerNorm2dExport for TRT)

## Current State

**Best result: 13.4 fps on A100 with torch.compile(reduce-overhead) + channels_last + cudnn.benchmark, bs=8. ~$2.65/episode at current A100 pricing ($2.10/hr).**

Model is NAFNet-width64 (116M params, pure CNN). Memory-bandwidth-bound.

### What changed this session

1. **Upgraded PyTorch 2.5.1 → 2.7.1 + torch-tensorrt 2.5.0 → 2.7.0 + cu121 → cu124** in both `modal_profile.py` and `modal_denoise.py`. NOT YET TESTED.

2. **Added Inductor optimizations** to `modal_profile.py`:
   - `TORCHINDUCTOR_FREEZING=1` — inlines weights as constants (15-30% expected gain)
   - `conv_1x1_as_mm=True` — converts 1x1 conv to GEMM (5-15% expected gain)
   - These are always-on — no flag needed

3. **Added GroupNorm(1) swap** — `swap_layernorm_for_groupnorm()` in `lib/nafnet_arch.py`. Applied automatically in `_run_profile` for non-TRT runs. Mathematically identical to LayerNorm2d but fusable by Inductor.

4. **Added `engine_cache_dir`** pointing to Modal volume for TRT engine persistence.

5. **Updated A100 pricing** from $2.78 to $2.10/hr.

6. **Added H100 ($3.95/hr) profile functions** — both base and TRT variants.

## What to Do Next (in order)

1. **Sanity check: torch.compile on A100** — run baseline to verify PyTorch 2.7.1 upgrade works:
   ```
   PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_profile.py --compile --channels-last --cudnn-benchmark --gpu A100
   ```
   This also tests the new Inductor flags (freezing, conv_1x1_as_mm) and GroupNorm swap since they're always-on now.

2. **Sanity check: TensorRT fp16 on A100** — verify TRT engine caching works on 2.7.0:
   ```
   ... cloud/modal_profile.py --tensorrt --channels-last --gpu A100
   ```

3. **Test H100** — raw bandwidth scaling:
   ```
   ... cloud/modal_profile.py --compile --channels-last --cudnn-benchmark --gpu H100
   ```

4. **INT8 quantization** (if TRT fp16 works):
   - Add `nvidia-modelopt` to `trt_image`
   - Implement calibration + INT8 compile path
   - Realistic speedup: 1.2-1.4x (not 2x) due to depthwise conv limitations
   - Watch quality on depthwise conv layers — may need per-layer fallback

5. **Process Firefly S01E02** once speed is acceptable:
   `"E:/plex/tv/Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)/Firefly (2002) - S01E02 - The Train Job (1080p BluRay x265 Silence).mkv"`

## Data Location
`data/` is a symlink to `E:/upscale-data/` (exFAT storage drive). Checkpoints on C: at `checkpoints/nafnet_distill/nafnet_best.pth` (443MB). Calibration images in `data/train_pairs/`.

## Compile & Engine Caches
- `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` → `/mnt/data/torch_compile_cache` on Modal volume
- TRT engine cache → `/mnt/data/trt_engine_cache` on Modal volume
- TRT engines are GPU-architecture-specific (A100 sm_80 ≠ H100 sm_90)

## Critical Rules
- Goal: minimize cost per episode AND maximize fps
- Do NOT run GPU tasks locally (RTX 3060 6GB)
- Modal command: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_profile.py [flags]`
- Do web research BEFORE trying new optimizations
- Write research findings to `bench/speed-opt/research.md`
- Record all experiment results to `bench/speed-opt/results.tsv`
- Commit after every meaningful change
