# Video Enhancement Experiment Platform

## What This Is
An experimentation platform for improving video quality using ML models. The primary use case is reducing compression artifacts in existing video libraries (e.g., BluRay rips) at native resolution.

## Goals
- Denoise/enhance 1080p video content using learned models
- Process full episodes in reasonable time on consumer hardware (RTX 3060 6GB)
- Compare different approaches: per-frame denoisers, temporal fusion, diffusion models
- Eventually train a custom fast model via distillation

## Architecture
- **Local inference** runs on Windows with conda env `upscale` (Python 3.10, PyTorch 2.5.1+cu121)
- **Cloud training** uses [Modal](https://modal.com) for GPU compute — CLI authenticated, account with billing attached
- **Streaming pipeline** reads video -> processes frames -> writes video (no intermediate files)
- All scripts are standalone Python — no framework dependencies beyond PyTorch

## Key Constraints
- 6GB VRAM — must use fp16, tiling, or half-res tricks to fit
- Dependencies get overwritten easily — always install PyTorch CUDA from `--index-url https://download.pytorch.org/whl/cu121` LAST or with `--no-deps`
- Prefer patching code for modern PyTorch over pinning old dependencies
- xformers is unreliable on Windows — use native `F.scaled_dot_product_attention` instead

## Directory Structure
- `lib/` — shared importable code: paths, ffmpeg utils, metrics, NAFNet architecture
- `pipelines/` — production streaming denoisers (SCUNet batch, NAFNet, episode)
- `experiments/` — one-off experiments and older approaches
- `training/` — NAFNet distillation training scripts
- `cloud/` — Modal remote GPU execution scripts
- `bench/` — benchmarking and quality comparison
- `tools/` — small utilities (clip extraction, probing, MP4 repair)
- `docs/` — documentation (setup, architecture, experiment log, approach comparison)

## Key Scripts
- `pipelines/denoise_batch.py` — main pipeline: batched SCUNet with threaded IO
- `pipelines/denoise_nafnet.py` — NAFNet pipeline (for distilled model)
- `pipelines/denoise_episode.py` — original episode denoiser (simpler, single-frame)
- `training/train_nafnet.py` — NAFNet distillation training loop
- `bench/compare.py` — PSNR/SSIM metrics and side-by-side comparison
- `bench/bench_nafnet.py` — NAFNet vs SCUNet benchmark

## Reference Code Submodules (`reference-code/`)
- `SCUNet/` — Swin-Conv-UNet denoiser (current best approach, patched: thop try/except)
- `RAFT/` — optical flow estimation
- `NAFNet/` — NAFNet architecture reference (pretrained weights)
- `Video-Depth-Anything/` — temporally consistent depth maps (patched: xformers→SDPA)
- `FlashVSR/`, `FlashVSR-Pro/`, `BasicVSR_PlusPlus/` — video SR references
- `KAIR/` — image restoration toolkit reference

## Data Directory
`data/` is a symlink to `E:/upscale-data/` (exFAT storage drive). Contains video clips, extracted frames, model outputs. Git-ignored due to size. Checkpoints remain on C: at `checkpoints/` for fast Modal upload.

## Current Status (2026-04-01)

**Working pipeline:** NAFNet distilled denoiser runs at **27.9 fps on H100** via `cloud/modal_denoise.py`. Full episodes process in ~37 min for ~$2.40. Uses PyTorch 2.7.1 + torch.compile(reduce-overhead) + CUDA graphs + Inductor optimizations.

**First episode done:** Firefly S01E02 processed end-to-end (61K frames, 27.9 fps, 0 errors). Output has H.264 10-bit video + all audio/subtitle tracks.

**Quality issue:** Output is slightly soft and dark shadows are still noisy. Needs training improvements — current model was only trained for 5K of planned 50K iters, uses Charbonnier loss only (no perceptual loss), 256px crops, limited data.

**Key TODO:** Improve training quality (perceptual loss, more data, longer training, larger crops). See `docs/distillation-guide.md`.

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

**torch.compile:** Works on NAFNet (pure CNN). Does NOT work on SCUNet (dynamic W/SW window branches cause infinite recompilation).

**FFmpeg on Modal:** Debian apt ffmpeg has no NVENC. `cloud/modal_denoise.py` builds ffmpeg from source with nv-codec-headers for NVENC support. Don't replace with apt ffmpeg.

**x265 on Modal:** Prints "Failed to generate CPU mask" and falls back to single-threaded unless you add `pools=4` to x265-params.

**Local ffmpeg:** imageio_ffmpeg ships v4.2.2 — doesn't support modern NVENC presets. Use `libx265` encoder locally or old NVENC syntax (`-rc vbr_hq -cq N`).

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
