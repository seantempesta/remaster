# Video Enhancement Experiment Platform

## What This Is
An experimentation platform for improving video quality using ML models. The primary use case is reducing compression artifacts in existing video libraries (e.g., BluRay rips) at native resolution.

### Key Results
| Metric | Value | Details |
|--------|-------|---------|
| Model inference | **78 fps** @ 1080p | NAFNet w32_mid4, 14.3M params, torch.compile, RTX 3060 |
| End-to-end pipeline | **5.2 fps** @ 1080p | Decode + inference + HEVC encode (bottlenecked by decode/encode) |
| Model VRAM | **2.3 GB** | fp16, batch 1, with torch.compile CUDA graphs |
| Model size | **55 MB** | Checkpoint (params only) |
| Training speed | **2.2 it/s** | H100, bs=32, RAM cache, VGG perceptual loss every iter |
| Training cost | **~$13** | 25K iters on H100 (~3.2 hrs) |
| Quality (w32_mid4) | **49.50 dB** PSNR | vs SCUNet GAN+detail teacher targets on held-out val |
| Quality (w64) | **56.82 dB** PSNR | 40K iters, same teacher targets |
| Speedup vs teacher | **78x** | SCUNet teacher: ~1 fps; NAFNet w32_mid4: 78 fps raw |

## Goals
- Denoise/enhance 1080p video content using learned models
- Process full episodes in reasonable time on consumer hardware (RTX 3060 6GB)
- Compare different approaches: per-frame denoisers, temporal fusion, diffusion models
- Eventually train a custom fast model via distillation

## Architecture
- **Local inference** runs on Windows with conda env `upscale` (Python 3.10, PyTorch 2.11.0+cu126)
- **Cloud training** uses [Modal](https://modal.com) for GPU compute — CLI authenticated, account with billing attached
- **Streaming pipeline** reads video -> processes frames -> writes video (no intermediate files)
- **GPU video pipeline** (`pipelines/denoise_gpu.py`) uses PyNvVideoCodec for zero-copy NVDEC decode → inference → NVENC encode (WIP)
- All scripts are standalone Python — no framework dependencies beyond PyTorch

## Key Constraints
- 6GB VRAM — must use fp16, tiling, or half-res tricks to fit
- Dependencies get overwritten easily — always install PyTorch CUDA from `--index-url https://download.pytorch.org/whl/cu126` LAST or with `--no-deps`
- Prefer patching code for modern PyTorch over pinning old dependencies
- xformers is unreliable on Windows — use native `F.scaled_dot_product_attention` instead

## Directory Structure
- `lib/` — shared importable code: paths, ffmpeg utils, metrics, NAFNet architecture
- `pipelines/` — production streaming denoisers (SCUNet batch, NAFNet, episode)
- `experiments/` — one-off experiments and older approaches
- `training/` — NAFNet distillation training (train_nafnet.py, losses.py, dataset.py, viz.py)
- `cloud/` — Modal remote GPU execution scripts
- `bench/` — benchmarking and quality comparison
- `tools/` — small utilities (clip extraction, probing, MP4 repair)
- `playback/` — mpv + VapourSynth + vs-mlrt real-time playback (enhance.vpy, README)
- `docs/` — documentation (setup, architecture, experiment log, approach comparison)

## Key Scripts
- `pipelines/denoise_batch.py` — main pipeline: batched SCUNet with threaded IO
- `pipelines/denoise_nafnet.py` — NAFNet pipeline (configurable arch, torch.compile, NVENC)
- `pipelines/denoise_gpu.py` — zero-copy GPU pipeline (PyNvVideoCodec NVDEC/NVENC, original, 5.4 fps)
- `pipelines/denoise_gpu_v2.py` — CUDA streams + ring buffers + stream-isolated NVDEC/NVENC (6.8 fps, GIL-limited)
- `pipelines/denoise_gpu_v3.py` — NVDEC decode + ffmpeg pipe encode (5.0 fps, GPU→CPU sync bottleneck)
- `pipelines/denoise_fast.py` — pipe-based pipeline (ffmpeg NVDEC/NVENC via stdin/stdout, 3.5 fps)
- `pipelines/denoise_episode.py` — original episode denoiser (simpler, single-frame)
- `training/train_nafnet.py` — NAFNet distillation training loop (configurable arch, profiling, graceful stop)
- `training/losses.py` — Loss functions: Charbonnier, DISTS perceptual, Focal Frequency
- `training/dataset.py` — PairedFrameDataset with optional RAM cache
- `training/viz.py` — Training visualization: sample images + loss curves
- `tools/stop_training.py` — Send graceful stop signal to Modal training via Dict
- `tools/verify_arch_configs.py` — Verify weight loading for different NAFNet architectures
- `cloud/modal_export_onnx_w32.py` — Export NAFNet w32_mid4 to ONNX on Modal
- `playback/enhance.vpy` — VapourSynth script for real-time mpv playback via vs-mlrt TensorRT
- `bench/compare.py` — PSNR/SSIM metrics and side-by-side comparison
- `bench/bench_nafnet.py` — NAFNet vs SCUNet benchmark

## Reference Code Submodules (`reference-code/`)
- `SCUNet/` — Swin-Conv-UNet denoiser (current best approach, patched: thop try/except)
- `RAFT/` — optical flow estimation
- `NAFNet/` — NAFNet architecture reference (pretrained weights)
- `DISTS/` — Deep Image Structure and Texture Similarity (perceptual loss, patched: modern torchvision API)
- `Video-Depth-Anything/` — temporally consistent depth maps (patched: xformers→SDPA)
- `FlashVSR/`, `FlashVSR-Pro/`, `BasicVSR_PlusPlus/` — video SR references
- `KAIR/` — image restoration toolkit reference

## Data Directory
`data/` is a symlink to `E:/upscale-data/` (exFAT storage drive). Contains video clips, extracted frames, model outputs. Git-ignored due to size. Checkpoints remain on C: at `checkpoints/` for fast Modal upload.

## Current Status (2026-04-02)

**Cloud inference:** NAFNet width64 at **27.9 fps on H100** via `cloud/modal_denoise.py`. ~$2.40/episode. Modal on PyTorch 2.11.0+cu126.

**Local inference (w64):** **1.94 fps** with torch.compile on RTX 3060. TensorRT FP16: 1.92 fps, 96MB VRAM.

**Local inference (w32_mid4):** Raw model: **78 fps** (13ms/frame). Best Python pipeline: **6.8 fps** (`denoise_gpu_v2.py` with CUDA stream isolation). **Python's GIL is the ceiling** — three pipeline threads serialize on the GIL regardless of CUDA stream separation. See `docs/realtime-playback-research.md` for full analysis. Reaching 40+ fps requires a C++ pipeline (TensorRT + Video Codec SDK, or vs-mlrt).

**ONNX exports:** w64 at `checkpoints/nafnet_distill/nafnet_w64_1088x1920.onnx`. w32_mid4 at `checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx` (57MB, opset 18, validated). Exported via `cloud/modal_export_onnx_w32.py`.

**Completed training runs:**
- **w64 GAN+detail (40K iters, A100):** Best PSNR 56.82 dB. Checkpoint at `checkpoints/nafnet_distill/nafnet_best.pth` (464MB). Final loss 0.055.
- **w32_mid4 VGG (25K iters, H100):** Best PSNR 49.50 dB, best total loss 0.0629 at iter 11K. Checkpoint at `checkpoints/nafnet_w32_mid4/nafnet_best.pth` (55MB).
- **w32_mid4 DISTS+FFT:** In progress from VGG weights at `checkpoints/nafnet_w32_mid4_dists/`.

**Training infrastructure improvements:**
- Configurable architecture (--width, --middle-blk-num) with strict=False weight surgery
- RAM cache eliminates DataLoader bottleneck (64% → 25% data wait, 0.7 → 2.2 it/s)
- CUDA event profiling (fwd/vgg/bwd/opt breakdown per iteration)
- Graceful shutdown: SIGINT handler + Modal Dict stop signal (`tools/stop_training.py`)
- Best model selection by total loss (pixel + perceptual), not PSNR-only
- Fused AdamW, non_blocking transfers, cuDNN benchmark warmup with cleanup

**Quality:** GAN+detail targets produce sharper output than PSNR-only teacher. Detail transfer adds real texture from original frames (zero hallucination). Best alpha = 0.15. Concern: alpha=0.15 may transfer too much compression noise — consider lower alpha or GAN-only targets.

**Test clips in data/:**
- `clip_mid_1080p.mp4` — original source (24s, 720 frames)
- `clip_mid_1080p_nafnet_psnr.mkv` — w64 PSNR-only distillation (old)
- `clip_mid_1080p_nafnet_gan_best.mkv` — w64 GAN+detail, best PSNR checkpoint
- `clip_mid_1080p_nafnet_gan_final.mkv` — w64 GAN+detail, final 40K iter checkpoint
- `clip_mid_1080p_nafnet_w32mid4.mkv` — w32_mid4 with torch.compile (5.2 fps)

**Pipeline findings (2026-04-02):**
- PyNvVideoCodec accepts `cuda_stream=` (decoder) and `cudastream=` (encoder) for stream isolation
- torch.compile `reduce-overhead` CUDA graphs replay on **default stream** regardless of `torch.cuda.stream()` context — events must be recorded on `torch.cuda.default_stream()`
- Python GIL caps all pipeline architectures at ~5-7 fps (GIL contention between decode/infer/encode threads)
- VLC plugin: not viable (CPU-only filter API). See `docs/realtime-playback-research.md`

**Next steps:**
- Install VapourSynth + vs-mlrt + mpv for real-time playback (see `playback/README.md`)
- Test vs-mlrt batch encoding via `vspipe` + ffmpeg (C++ TensorRT, no GIL)
- Evaluate DISTS+FFT training quality vs VGG
- If vs-mlrt insufficient: custom C++ pipeline (Video Codec SDK + TensorRT)

**Research docs:** `docs/quantization-research.md`, `docs/tensorrt-implementation.md`, `docs/detail-recovery-research.md`, `docs/quantization-aware-training.md`, `docs/gpu-profiling-guide.md`, `docs/modal-graceful-shutdown.md`, `docs/realtime-playback-research.md`, `docs/zero-copy-gpu-pipeline.md`.

## Critical Gotchas

**LOCAL MACHINE HAS ONLY 16GB RAM — MEMORY IS THE BOTTLENECK, NOT VRAM.**
- Training checkpoints are 1.3GB each (model + optimizer). Loading multiple on CPU will fill RAM and cause swap thrashing that freezes the machine.
- NEVER run parallel agents/processes that load models locally. One model-loading process at a time.
- When loading checkpoints for eval/inference, extract only model weights (`ckpt['params']` or `ckpt['model']`, ~464MB) and immediately `del ckpt; gc.collect()`.
- Prefer GPU inference over CPU — GPU VRAM (6GB) is separate from system RAM. NAFNet fp16 batch 1 at 1080p uses ~3.3GB VRAM, well within limits.
- If running eval across multiple checkpoints, load/eval/delete one at a time, never hold multiple in memory.
- pip installs that compile large packages (e.g., tensorrt) can also spike RAM — don't run alongside model loading.

**DO NOT run heavy GPU models from agents locally** — the RTX 3060 has only 6GB VRAM. Running SCUNet or NAFNet at 1080p will spill into shared system RAM and freeze the machine. Do code writing + syntax checks locally, run inference on Modal.

**Windows + Modal:** Never use `conda run -n upscale modal run ...` — breaks with UnicodeEncodeError. Use: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/script.py`

**Modal Volume paths:** `batch_upload.put_file(local, remote)` — remote is volume-relative (e.g., `/input/file.mp4`). Container access uses mount prefix (`/mnt/data/input/file.mp4`). Must call `vol.reload()` inside container functions before reading uploaded files.

**NAFNet fp16:** Fixed in `lib/nafnet_arch.py`. LayerNorm2d casts to fp32 for normalization, returns fp16. Don't revert this.

**torch.compile:** Works on NAFNet (pure CNN). Does NOT work on SCUNet (dynamic W/SW window branches cause infinite recompilation). `reduce-overhead` mode uses CUDA graphs that replay on the **default stream** regardless of `torch.cuda.stream()` context — record events on `torch.cuda.default_stream()` not the custom stream.

**Python GIL limits pipeline to ~7 fps.** PyNvVideoCodec's C extensions hold the GIL during blocking NVDEC/NVENC calls. Threading doesn't help — all three pipeline stages (decode/infer/encode) serialize on the GIL. Reaching 40+ fps requires C++ (vs-mlrt, custom TensorRT pipeline).

**FFmpeg on Modal:** Debian apt ffmpeg has no NVENC. `cloud/modal_denoise.py` builds ffmpeg from source with nv-codec-headers for NVENC support. Don't replace with apt ffmpeg.

**x265 on Modal:** Prints "Failed to generate CPU mask" and falls back to single-threaded unless you add `pools=4` to x265-params.

**Local ffmpeg:** Modern ffmpeg 7.1 with NVENC at `bin/ffmpeg.exe`. `lib/ffmpeg_utils.get_ffmpeg()` prefers this over the old imageio_ffmpeg v4.2.2. All pipeline scripts use `get_ffmpeg()`. NVENC presets: `-preset p4 -tune hq -rc vbr -cq N`.

**PyNvVideoCodec:** Installed (v2.1.0). Zero-copy NVDEC decode to CUDA tensors via `torch.from_dlpack()`. NVENC encode from GPU. Needs `os.add_dll_directory()` for PyTorch CUDA DLLs on Windows. RTX 3060 NVDEC cannot decode H264 High 10 (10-bit) — use HEVC sources. **Stream isolation:** pass `cuda_stream=stream.cuda_stream` to SimpleDecoder and `cudastream=stream.cuda_stream` to CreateEncoder to prevent their internal syncs from blocking other streams.

**ONNX export (PyTorch 2.11):** The dynamo exporter saves weights as external `.data` files. Must merge them inline: load with `load_external_data=True`, clear `data_location` on initializers, re-save. See `cloud/modal_export_onnx_w32.py`. Also requires `onnxscript` pip package.

## Modal Development Guidelines

Full docs: modal.com/docs — markdown for LLMs: modal.com/llms-full.txt — examples: modal.com/docs/examples (github.com/modal-labs/modal-examples)

### Style & Conventions
- Always `import modal` and use qualified names: `modal.App()`, `modal.Image.debian_slim()`, etc.
- Name Apps, Volumes, and Secrets with **kebab-case**
- Modal evolves quickly — never use deprecated features; `modal run` prints deprecation warnings
- Dependencies belong in per-Function `Image` definitions, not global requirements files
- Put `import` statements for heavy deps **inside** the Function body (global scope must run everywhere)

### Core Concepts
- **App** — group of Functions/Classes deployed together
- **Function** — basic unit of serverless execution, each runs in its own container with its own Image
- **Cls** — stateful class with `@modal.enter()` / `@modal.exit()` lifecycle hooks
- **Image** — container image (`modal.Image.debian_slim().pip_install(...)`)
- **Volume** — distributed filesystem; **Secret** — credentials; **Dict** / **Queue** — distributed data structures
- **Sandbox** — run arbitrary code securely at runtime

### Function Configuration
```python
@app.function(
    image=modal.Image.debian_slim(python_version="3.12").pip_install("torch"),
    gpu="A10G",        # or "H100", "A100:2", ["H100", "A100", "any"]
    memory=4096,
    cpu=2,
)
def my_func():
    ...
```

### Invocation Patterns
- `fn.remote()` — run in cloud container (most common)
- `fn.local()` — run in caller's context
- `fn.map(inputs)` — parallel map
- `fn.spawn()` — fire-and-forget (terminated when App exits)

### Web Endpoints
```python
@app.function()
@modal.fastapi_endpoint()
def endpoint():
    return {"status": "ok"}
```

### Scheduling
```python
@app.function(schedule=modal.Period(minutes=5))
# or: schedule=modal.Cron("0 9 * * *")
```

### CLI Commands
- `modal run path/to/app.py` — run during development
- `modal serve path/to/app.py` — serve web endpoints with hot-reload
- `modal deploy path/to/app.py` — deploy to production
- `modal app logs <app_name>` — stream logs
- `modal app list`, `modal volume list`, `modal secret list`, etc. — resource management
