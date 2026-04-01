# Transfer Prompt for Next Conversation

## Read these files first
- `CLAUDE.md` — environment constraints, gotchas, project overview
- `bench/speed-opt/EXPERIMENT.md` — how to run experiments, the Karpathy loop
- `bench/speed-opt/results.tsv` — all experiment results so far
- `bench/speed-opt/research.md` — all research findings (TRT, INT8, compile caching)
- `cloud/modal_profile.py` — the profiling script
- `lib/nafnet_arch.py` — model architecture (includes LayerNorm2dExport for TRT)

## Current State

**Best result: 13.4 fps on A100 with torch.compile(reduce-overhead) + channels_last + cudnn.benchmark, bs=8. $3.51/episode.**

Model is NAFNet-width64 (116M params, pure CNN). Memory-bandwidth-bound — batching barely helps, max-autotune barely helps. Need kernel fusion (TensorRT) or quantization (INT8) to go faster.

## What's Blocked: TensorRT

TRT is blocked on a version mismatch. We're on **PyTorch 2.5.1 + torch-tensorrt 2.5.0 + CUDA 12.1**. Three attempts failed:
1. Old API shape mismatch (fixed in code but API doesn't cache)
2. New API `immutable_weights` option doesn't exist in v2.5.0
3. v2.5.0 uses `make_refittable=True` — this is documented in research.md but NOT TESTED YET

**Two paths forward:**
- **Quick fix**: Try the correct v2.5.0 API (`make_refittable=True` + `engine_cache_dir`) — see research.md for exact code
- **Better fix**: Upgrade to PyTorch 2.7+ and torch-tensorrt 2.7+ in the Modal image. This gets us modern APIs, better Inductor, AOTInductor, and better TRT op coverage. Change the pip_install lines in `cloud/modal_profile.py` and `cloud/modal_denoise.py`.

## What's Researched and Ready: INT8 Quantization

Full research in `bench/speed-opt/research.md`. Key points:
- Use **NVIDIA ModelOpt** (`nvidia-modelopt`) for post-training quantization
- `mtq.quantize(model, INT8_DEFAULT_CFG, forward_loop=calibrate_loop)` then compile with TRT
- Expected: ~0.5-1.0 dB PSNR loss (acceptable), ~1.5-2x speedup over fp16 TRT
- We have 1000 calibration images in `data/train_pairs/`
- A100 INT8 tensor cores: 624 TOPS vs 312 TOPS fp16
- Watch out: depthwise convolutions may be sensitive to quantization

## Data Location
`data/` is a symlink to `E:/upscale-data/` (exFAT storage drive). Checkpoints on C: at `checkpoints/nafnet_distill/nafnet_best.pth` (443MB).

## Compile Cache
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` are set to Modal volume paths in both `modal_profile.py` and `modal_denoise.py`. torch.compile caches persist across runs.

## What to Do Next (in order)

1. **Upgrade PyTorch + torch-tensorrt** in the Modal image. Research the correct compatible versions first. Update `_packages_image` and `trt_image` in `cloud/modal_profile.py`. Also update `cloud/modal_denoise.py`.

2. **Get TensorRT fp16 working** with engine caching. Use the `torch.compile(backend="torch_tensorrt")` API with the correct options for the new version. Run on A100, measure fps. The engine build will be slow (~8 min) but cached for all future runs.

3. **If TRT fp16 works**, try INT8 quantization on top (ModelOpt path).

4. **Process Firefly S01E02** once speed is acceptable:
   `"E:/plex/tv/Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)/Firefly (2002) - S01E02 - The Train Job (1080p BluRay x265 Silence).mkv"`

## Critical Rules
- Do NOT run GPU tasks locally (RTX 3060 6GB)
- Modal command: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_profile.py [flags]`
- Do web research BEFORE trying new optimizations — check the docs for YOUR version
- Write research findings to `bench/speed-opt/research.md`
- Record all experiment results to `bench/speed-opt/results.tsv`
- Commit after every meaningful change
- Follow the experiment loop: improve → commit → next. Don't overthink.
