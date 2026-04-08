# Remaster -- Video Enhancement Pipeline

## What This Is
A video remastering pipeline that removes compression artifacts AND recovers detail from video content. Uses a DRUNet teacher-student distillation approach: a large teacher model learns to enhance video, then a small student model learns to replicate it at real-time speeds.

### Current Models
| Model | Params | PSNR | Speed (RTX 3060) | Checkpoint |
|-------|--------|------|-------------------|------------|
| **DRUNet Teacher** | 32.6M | 53.27 dB | ~5 fps | `checkpoints/drunet_teacher/best.pth` (125MB) |
| **DRUNet Student** | 1.06M | 49.19 dB | 43 fps FP16, 44 fps INT8 mixed (C++ pipeline) | `checkpoints/drunet_student/final.pth` (4MB) |

Both are DRUNet (UNetRes from KAIR, MIT license). Same architecture, different sizes:
- Teacher: nc=[64,128,256,512] nb=4 -- quality ceiling
- Student: nc=[16,32,64,128] nb=2 -- deployment target

### Checkpoint Format
- `final.pth` — `{"params": state_dict}`. Model weights from the most recent checkpoint. **Use this for inference and as teacher weights.**
- `best.pth` — `{"params": state_dict, "iteration": int, "psnr": float}`. Model at highest validation PSNR. Not always reliable — metrics aren't comparable across runs with different loss configs.
- `latest.pth` — Full training state (model + optimizer + scheduler + EMA + adapters). For resuming training. Only on Modal volume, not synced locally unless explicitly downloaded.

### Key Finding
Mixed training data (HEVC artifact removal + synthetic edge-aware blur) produces a model
that generalizes beyond its training tasks -- it denoises AND sharpens, often exceeding
the quality of the original Bluray source material.

## Goals
- Denoise/enhance 1080p video content using learned models
- Process full episodes in reasonable time on consumer hardware (RTX 3060 6GB)
- Train via distillation: large teacher -> small real-time student

## Architecture
- **Local inference** runs on Windows with conda env `remaster` (Python 3.12, PyTorch 2.11.0+cu130, CUDA 13.2)
- **Cloud training** uses [Modal](https://modal.com) for GPU compute — CLI authenticated, account with billing attached
- **C++ pipeline** (`pipeline_cpp/`) — NVDEC -> TRT -> NVENC zero-copy, 43-44 fps end-to-end
- **Streaming pipeline** reads video -> processes frames -> writes video (no intermediate files)
- All scripts are standalone Python — no framework dependencies beyond PyTorch

## Key Constraints
- 6GB VRAM — must use fp16, tiling, or half-res tricks to fit
- Dependencies get overwritten easily — always install PyTorch CUDA from `--index-url https://download.pytorch.org/whl/cu130` LAST or with `--no-deps`
- Prefer patching code for modern PyTorch over pinning old dependencies
- xformers is unreliable on Windows — use native `F.scaled_dot_product_attention` instead

## Directory Structure
- `training/` — unified training script, losses, dataset, visualization
- `cloud/` — Modal cloud training wrapper (`modal_train.py`)
- `tools/` — utilities: data extraction, training sync, stop signal, frame probing
- `lib/` — shared code: paths, ffmpeg utils, architecture definitions
- `checkpoints/` — active model weights (teacher + student)
- `checkpoints/_archive/` — old models (NAFNet, PlainNet, v1 DRUNet)
- `output/` — training artifacts synced from Modal, local validation images
- `data/` — symlink to E:/upscale-data/ (training pairs, synthetic pairs, clips)
- `remaster/` — production VapourSynth pipeline (encode, playback)
- `playback/` — mpv + vs-mlrt real-time playback
- `pipelines/` — older Python streaming pipelines (archive)
- `bench/` — benchmarking scripts
- `docs/` — documentation, plans, research notes

## Key Scripts

### Training
- `training/train.py` — unified training: DRUNet distillation, feature matching, Prodigy optimizer, DISTS
- `training/losses.py` — Charbonnier, DISTS perceptual, Focal Frequency, Feature Matching
- `training/dataset.py` — PairedFrameDataset, InputOnlyDataset, GPUCachedDataset
- `training/viz.py` — sample images + loss curves (used during training on Modal)
- `cloud/modal_train.py` — Modal cloud training wrapper (L40S default, W&B logging)

### Tools
- `tools/build_training_data.py` — staged data builder: --extract-only, --denoise, --build-inputs
- `tools/calibrate_sigma.py` — bucket frames by noise, pre-render sigma grids, fit calibration curve
- `tools/label_sigma.py` — Streamlit app for human sigma labeling
- `tools/verify_data.py` — check pair completeness, readability, per-source counts
- `tools/measure_sharpness.py` — compute sharpness metrics for training targets
- `tools/visualize_sharpness.py` — plot sharpness distributions and sample grids
- `tools/download_training.py` — sync training artifacts from Modal to local
- `tools/stop_training.py` — graceful stop via Modal Dict

### Deployment
- `remaster/encode_nvencc.py` — **Fastest encoder**: NVEncC + VapourSynth in-process (~36 fps)
- `remaster/encode_nvencc.vpy` — VapourSynth script for NVEncC (TRT inference, no pipe)
- `remaster/encode.py` — VapourSynth + ffmpeg pipe encoder (~20 fps, audio passthrough)
- `remaster/encode.vpy` — VapourSynth batch encoding (BestSource + TRT engine + y4m)
- `remaster/play.vpy` — VapourSynth real-time playback for mpv (TRT engine)
- `remaster/bench_trt.vpy` — Component benchmark: decode, colorspace, inference, full pipeline
- `pipelines/remaster.py` — Python streaming pipeline (PyAV + torch.compile, ~24 fps)
- `tools/export_onnx.py` — Export DRUNet to ONNX (dynamic shapes)
- `tools/build_int8_engine.py` — INT8 TRT engine with proper calibration from real frames
- `tools/build_int8_calibration.py` — Generate calibration data from originals
- `pipeline_cpp/` — C++ NVDEC->TRT->NVENC zero-copy pipeline (44 fps, 10-bit HEVC, audio passthrough)
- `bench/compare.py` — PSNR/SSIM metrics and side-by-side comparison

## Reference Code Submodules (`reference-code/`)
- `SCUNet/` — Swin-Conv-UNet denoiser (current best approach, patched: thop try/except)
- `RAFT/` — optical flow estimation
- `NAFNet/` — NAFNet architecture reference (pretrained weights)
- `DISTS/` — Deep Image Structure and Texture Similarity (perceptual loss, patched: modern torchvision API)
- `Video-Depth-Anything/` — temporally consistent depth maps (patched: xformers->SDPA)
- `sam3/` — SAM 3 (Segment Anything with Concepts) - semantic segmentation (explored, not used for denoising)
- `dinov3/` — DINOv3 self-supervised vision features (ConvNeXt + ViT, explored for feature matching)
- `NVEnc/` — NVEncC NVENC encoder with VapourSynth --vpy support (source for reference)
- `FlashVSR/`, `FlashVSR-Pro/`, `BasicVSR_PlusPlus/` — video SR references
- `ECBSR/` — Edge-oriented Conv Block SR (reparameterizable training, mobile-first)
- `SPAN/` — NTIRE 2024 efficient SR winner (reparameterizable Conv3XC, EMA training)
- `XLSR/` — Quantization-designed SR (Clipped ReLU for INT8, QAT recipe)
- `KAIR/` — image restoration toolkit reference
- `vapoursynth/` — VapourSynth core (C++ frame processing runtime)
- `vs-mlrt/` — ML inference plugin for VapourSynth (TensorRT, ONNX Runtime backends)
- `bestsource/` — Frame-accurate video source filter for VapourSynth (FFmpeg-based)

## Data Directory
`data/` is a symlink to `E:/upscale-data/` (exFAT storage drive). Git-ignored due to size. Checkpoints remain on C: at `checkpoints/` for fast Modal upload.

```
data/
  originals/       ~7K raw extracted frames + meta.pkl (permanent cache)
  training/
    train/         Training pairs (~6,260): input/ + target/
    val/           Validation pairs (~696): input/ + target/
    meta.pkl       DataFrame: sigma, split, degradation_type, metrics
  calibration/     Sigma calibration: grids/, labels.pkl, sigma_model.pkl
  analysis/        Visualization outputs (sharpness plots, etc.)
  archive/         Old data, episode comparisons, demo clips
  output/          Training artifacts synced from Modal
```

**Staged data pipeline** (see `docs/research/training-data/plan.md`):
1. Extract originals (1/500 frames, proportional to content length)
2. Denoise with SCUNet GAN + light USM(1.0) -> **TARGET** (denoise + sharpen in one pass)
3. Build degraded **INPUT**: 33% raw original, 33% +noise, 33% +edge-aware blur+noise

Training data sources (proportional sampling, 1 frame per 500 source frames):
| Prefix | Source | Samples (approx) | Resolution |
|--------|--------|-------------------|------------|
| `firefly_*` | Firefly S01 | ~1,876 | 1920x1080 |
| `expanse_*` | The Expanse S02 | ~1,635 | 1920x1080 |
| `onepiece_*` | One Piece S01 | ~1,306 | 1920x1080 |
| `squidgame_*` | Squid Game S02 | ~1,237 | 1920x960 |
| `dune2_*` | Dune Part Two | ~476 | 1920x802 |
| `foundation_*` | Foundation S03 | ~426 | 1920x800 |
| **Total** | | **~6,956** | |

See `docs/research/training-data/plan.md` for details. Built by `tools/build_training_data.py`.

## Current Status (2026-04-08)

### Architecture
- **DRUNet** (UNetRes from KAIR, MIT license): Conv+ReLU residual U-Net, 4 levels, no BN/LayerNorm/attention
- TRT FP16: 68.9 dB vs PyTorch FP32 (visually identical). TRT INT8 mixed: 67.2 dB (skip-adds + head/tail in FP16).
- Pretrained from `drunet_deblocking_color.pth` (Gaussian denoising)

### Training Approach
- **Teacher-student distillation** with feature matching (1x1 adapter convs align encoder features)
- **Targets**: SCUNet GAN (perceptual/adversarial denoiser) + USM(1.0) -- denoise AND sharpen in one pass
- **Inputs**: 33% raw originals, 33% +noise, 33% +edge-aware blur+noise
- **Losses**: Charbonnier pixel + DISTS perceptual (teacher), Charbonnier + feature matching (student)
- **Optimizer**: Prodigy (auto-tuned LR, safeguard_warmup, bias_correction)
- **Cloud**: Modal L40S ($1.95/hr, 48GB VRAM), W&B logging, 64GB RAM cache
- See `docs/research/training-data/plan.md` for data sources and sampling plan

### Training Commands
```bash
# Resume teacher (fresh optimizer for new data)
modal run cloud/modal_train.py --arch drunet --nc-list 64,128,256,512 --nb 4 \
    --checkpoint-dir checkpoints/drunet_teacher \
    --optimizer prodigy --perceptual-weight 0.05 --batch-size 64 \
    --ema --wandb --resume --fresh-optimizer

# Resume student (with teacher as online distillation target)
modal run cloud/modal_train.py --arch drunet --nc-list 16,32,64,128 --nb 2 \
    --teacher checkpoints/drunet_teacher/final.pth --teacher-model drunet \
    --checkpoint-dir checkpoints/drunet_student \
    --feature-matching-weight 0.1 --optimizer prodigy --batch-size 192 \
    --ema --wandb --resume --fresh-optimizer

# Fine-tune student on full frames (fixes dark area artifacts)
modal run cloud/modal_train.py --arch drunet --nc-list 16,32,64,128 --nb 2 \
    --teacher checkpoints/drunet_teacher/final.pth --teacher-model drunet \
    --checkpoint-dir checkpoints/drunet_student \
    --feature-matching-weight 0.1 --perceptual-weight 0.05 --optimizer prodigy \
    --batch-size 8 --crop-size 0 --max-iters 5000 \
    --ema --wandb --resume --fresh-optimizer
```

**Data upload:** `--skip-upload` is now the default. Training data persists on the Modal volume between runs. Pass `--no-skip-upload` only if you changed the training data (rebuilt pairs, added new sources, etc.).

### Deployment Pipeline (Production)
```bash
# Export ONNX (one-time, dynamo=False required for TRT compat)
python tools/export_onnx.py                    # FP16 ONNX (for FP16 engine)
python tools/export_onnx.py --fp32             # FP32 ONNX (for INT8 calibration)

# Build TRT FP16 engine (one-time per GPU)
tools/vs/vs-plugins/vsmlrt-cuda/trtexec.exe \
    --onnx=checkpoints/drunet_student/drunet_student.onnx \
    --shapes=input:1x3x1080x1920 --fp16 \
    --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
    --useCudaGraph --saveEngine=checkpoints/drunet_student/drunet_student_1080p_fp16.engine

# C++ pipeline (~43 fps FP16, ~44 fps INT8 mixed, 10-bit HEVC, audio passthrough)
pipeline_cpp/build/remaster_pipeline.exe \
    -i input.mkv -o output.mkv \
    -e checkpoints/drunet_student/drunet_student_1080p_fp16.engine --cq 20

# NVEncC + VapourSynth (39 fps, audio passthrough)
python remaster/encode_nvencc.py input.mkv output.mkv

# VapourSynth + ffmpeg pipe (20 fps)
python remaster/encode.py input.mkv output.mkv

# Python streaming with torch.compile (24 fps)
python pipelines/remaster.py -i input.mkv -c checkpoints/drunet_student/final.pth \
    --nc-list 16,32,64,128 --nb 2 --encoder hevc_nvenc --mux-audio --compile

# Real-time playback: configure mpv with remaster/play.vpy
```

### INT8 Mixed-Precision Engine
```bash
# Build INT8 engine with mixed precision (uses FP32 ONNX for calibration, FP16 I/O)
# Sensitive layers (skip-adds, head/tail, ConvTranspose) forced to FP16; interior convs INT8
python tools/build_int8_engine.py

# Quality: 67.2 dB vs FP32 (was 42.7 dB without mixed precision)
# Speed: 44.2 fps C++ pipeline (vs 42.9 fps FP16)
```

### ONNX Export Gotcha
**PyTorch 2.11 defaults `torch.onnx.export()` to `dynamo=True`** which emits opset 20 IR that TRT 10.16 miscompiles (14.5 dB output). Always use `dynamo=False` (opset 17, TorchScript exporter). This is already set in `tools/export_onnx.py`.

### Edge-Replicate Padding
All padding for sub-1080p content uses **edge-replication** (clamp coordinates), not zero-fill. This prevents black-border artifacts at conv boundaries. Applies to: C++ color kernels, Python test suite, INT8 calibrator. Search for "replicate" to find all instances.

### INT8 Quantization Status
Pure INT8 (all layers quantized) produces 26 dB — the skip-connection Add ops are the suspected cause. The hypothesis is that encoder/decoder paths have mismatched value distributions at addition points, and INT8's 256 levels can't represent the fine residuals. However, this has only been tested with PTQ (post-training quantization). **QAT (quantization-aware training) is untested** and could potentially learn quantization-friendly representations that make pure INT8 viable.

Current approach: exclude Add ops from INT8 via `op_types_to_exclude=["Add"]`. This gives 67.2 dB but limits speedup to ~3% over FP16.

**Engine builders:**
- `tools/build_engine.py` — Modern: ModelOpt Q/DQ + strongly typed (TRT 11-ready). Best for FP16. INT8 mixed OOMs on 6GB during ORT calibration at 1080p.
- `tools/build_int8_engine.py` — Legacy: IInt8EntropyCalibrator2 + weak typed. Works for INT8 mixed on 6GB. Manual FP16 layer exclusions.

### Next Steps
1. **QAT (quantization-aware training)** -- Train with fake quantization (pytorch-quantization toolkit, ~5K iters on Modal) to learn INT8-safe representations. Could enable pure INT8 on ALL layers including Adds. Expected: 52+ fps with full INT8 quality matching FP16.
2. **Recurrent temporal context** -- 9-channel input (prev_cleaned + current + next_noisy). PRD at `docs/research/temporal-context/prd.md`. Expected: +0.3-0.8 dB PSNR + temporal consistency
3. **dynamo=True ONNX export** -- Untested with TRT 10.16 + strongly typed. May work now (opset 18/20 support improved). Would eliminate the legacy TorchScript dependency.

### Verified Results (2026-04-08)
- **Teacher**: 13K iters, PSNR 53.27 dB, 107% sharpness, near-perfect color
- **Student**: Fine-tuned with full-frame (5K iters, crop_size=0), PSNR 49.98 dB, 1.06M params
- **C++ pipeline FP16**: ~42 fps end-to-end, 10-bit HEVC MKV, audio passthrough, CUDA graphs
- **C++ pipeline INT8 mixed**: ~44 fps end-to-end, 67.2 dB quality (legacy builder)
- **TRT FP16 (strongly typed)**: 42.6 fps trtexec (19.5ms GPU), 68.0 dB vs PyTorch FP32
- **TRT INT8 mixed (legacy)**: 43.3 fps trtexec (19.2ms GPU), 67.2 dB vs PyTorch FP32
- **TRT INT8 pure (ModelOpt Q/DQ)**: 39 fps trtexec (17.9ms GPU), 26.1 dB (Add ops need FP16)
- **NVEncC pipeline**: 39 fps with audio, VapourSynth in-process
- **Bottleneck**: TRT inference (~20ms/frame = 50 fps theoretical). NVDEC (500+ fps) and NVENC (200+ fps) are fully hidden via async I/O on separate CUDA streams. The profiler shows 22ms "decode time" but this includes waiting for the previous frame's GPU work — it is NOT an NVDEC bottleneck. Pipeline achieves 42 fps due to ~3ms sync/CPU overhead per frame.
- **Full episodes**: Firefly S01E05 (~42 min @ ~42 fps = ~24 min encode)
- **Environment**: Python 3.12, PyTorch 2.11+cu130, CUDA 13.2, TRT 10.16.0, VS R73
- **Build tools**: VS Build Tools 2022, CUDA 13.2, CMake 3.31, TRT 10.16 headers
- **Test suite**: 6/6 pass (FP32/FP16 precision, TRT FP16/INT8, color roundtrip, ONNX)
- **ModelOpt**: 0.42.0 installed, Polygraphy 0.49.26. Requires cuDNN on PATH (torch/lib/ fix in build_engine.py)

**Research docs:** `docs/research/` (temporal-context, raft-alignment, cpp-pipeline, training-data). **Guides:** `docs/guides/` (gpu-profiling, modal-graceful-shutdown). **Archive:** `docs/archive/` (old NAFNet-era docs). See `docs/README.md` for full index.

## Weights & Biases (W&B)

Project: `remaster` (entity: `seantempesta`). All training runs log to W&B automatically on Modal.

- **Modal**: ON by default. Uses Modal Secret `wandb-api-key` (WANDB_API_KEY env var). Pass `--no-wandb` to disable.
- **Local**: OFF by default. Pass `--wandb` flag to enable. Auth stored in `~/_netrc`.
- **What's logged**: train/val losses (pixel, perceptual, FFT, feature matching), PSNR, LR (+ Prodigy D), training speed, VRAM, data wait %, side-by-side sample images (input | target | student), gradient histograms, best model as artifact.
- **Run naming**: Auto-generated from architecture, e.g. `drunet-nc16_32_64_128-nb2-distill`. Full config (all args + param count + GPU) saved.
- **CLI args**: `--wandb-project NAME`, `--wandb-entity ENTITY`, `--wandb-run-name NAME`

## Critical Gotchas

**Windows cp1252 encoding — NEVER use unicode arrows/emoji in print() or strings that hit stdout.** Python on Windows defaults to cp1252 which can't encode characters like `→`, `✓`, `─`. Use ASCII equivalents (`->`, `OK`, `-`). This applies to all scripts, not just Modal.

**LOCAL MACHINE HAS ONLY 16GB RAM — MEMORY IS THE BOTTLENECK, NOT VRAM.**
- Training checkpoints are 1.3GB each (model + optimizer). Loading multiple on CPU will fill RAM and cause swap thrashing that freezes the machine.
- NEVER run parallel agents/processes that load models locally. One model-loading process at a time.
- When loading checkpoints for eval/inference, extract only model weights (`ckpt['params']` or `ckpt['model']`, ~464MB) and immediately `del ckpt; gc.collect()`.
- Prefer GPU inference over CPU — GPU VRAM (6GB) is separate from system RAM. NAFNet fp16 batch 1 at 1080p uses ~3.3GB VRAM, well within limits.
- If running eval across multiple checkpoints, load/eval/delete one at a time, never hold multiple in memory.
- pip installs that compile large packages (e.g., tensorrt) can also spike RAM — don't run alongside model loading.

**DO NOT run heavy GPU models from agents locally** — the RTX 3060 has only 6GB VRAM. Running SCUNet or NAFNet at 1080p will spill into shared system RAM and freeze the machine. Do code writing + syntax checks locally, run inference on Modal.

**NEVER run parallel engine builds or calibration** — each TRT build + calibration loads 200 frames (~1.5GB) plus engine builder VRAM. Two parallel builds will use 6+GB RAM and freeze the laptop. Always build/test engines SEQUENTIALLY, one at a time.

**Windows + Modal:** Never use `conda run -n remaster modal run ...` — breaks with UnicodeEncodeError. Use: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/remaster/python.exe -m modal run cloud/script.py`. Also applies to `modal volume get` (prints checkmarks that crash on cp1252) — always prefix with `PYTHONUTF8=1`.

**Modal Volume paths:** `batch_upload.put_file(local, remote)` — remote is volume-relative (e.g., `/input/file.mp4`). Container access uses mount prefix (`/mnt/data/input/file.mp4`). Must call `vol.reload()` inside container functions before reading uploaded files.

**NAFNet fp16:** Fixed in `lib/nafnet_arch.py`. LayerNorm2d casts to fp32 for normalization, returns fp16. Don't revert this.

**torch.compile:** Works on NAFNet (pure CNN). Does NOT work on SCUNet (dynamic W/SW window branches cause infinite recompilation). `reduce-overhead` mode uses CUDA graphs that replay on the **default stream** regardless of `torch.cuda.stream()` context — record events on `torch.cuda.default_stream()` not the custom stream.

**Python GIL limits pipeline to ~7 fps.** PyNvVideoCodec's C extensions hold the GIL during blocking NVDEC/NVENC calls. Threading doesn't help — all three pipeline stages (decode/infer/encode) serialize on the GIL. Reaching 40+ fps requires C++ (vs-mlrt, custom TensorRT pipeline).

**FFmpeg on Modal:** Debian apt ffmpeg has no NVENC. `cloud/modal_denoise.py` builds ffmpeg from source with nv-codec-headers for NVENC support. Don't replace with apt ffmpeg.

**x265 on Modal:** Prints "Failed to generate CPU mask" and falls back to single-threaded unless you add `pools=4` to x265-params.

**Local ffmpeg:** Modern ffmpeg 7.1 with NVENC at `bin/ffmpeg.exe`. `lib/ffmpeg_utils.get_ffmpeg()` prefers this over the old imageio_ffmpeg v4.2.2. All pipeline scripts use `get_ffmpeg()`. NVENC presets: `-preset p4 -tune hq -rc vbr -cq N`.

**PyNvVideoCodec:** Installed (v2.1.0). Zero-copy NVDEC decode to CUDA tensors via `torch.from_dlpack()`. NVENC encode from GPU. Needs `os.add_dll_directory()` for PyTorch CUDA DLLs on Windows. RTX 3060 NVDEC cannot decode H264 High 10 (10-bit) — use HEVC sources. **Stream isolation:** pass `cuda_stream=stream.cuda_stream` to SimpleDecoder and `cudastream=stream.cuda_stream` to CreateEncoder to prevent their internal syncs from blocking other streams.

**ONNX export (PyTorch 2.11):** Must use `dynamo=False` — the default dynamo exporter produces opset 20 IR that TRT miscompiles (14.5 dB). Legacy TorchScript exporter (opset 17) works correctly. Also, dynamo saves weights as external `.data` files that must be merged inline. See `tools/export_onnx.py`.

## Modal Development Guidelines

Full docs: modal.com/docs — markdown for LLMs: modal.com/llms-full.txt — examples: modal.com/docs/examples (github.com/modal-labs/modal-examples)

### GPU Options & Pricing (2026-04-04)

| GPU | VRAM | FP16 TFLOPS | Mem BW (TB/s) | $/hr | TFLOPS/$ | Best for |
|-----|------|-------------|---------------|------|----------|----------|
| T4 | 16 GB | 65 | 0.32 | $0.59 | 110 | Debug only |
| L4 | 24 GB | 121 | 0.30 | $0.80 | 151 | Cheap tests |
| A10G | 24 GB | 125 | 0.60 | $1.10 | 114 | Quick jobs, ONNX export |
| **L40S** | **48 GB** | **366** | **0.86** | **$1.95** | **188** | **Best value for training** |
| A100-40 | 40 GB HBM2e | 312 | 1.56 | $2.10 | 149 | Large batch, bandwidth-heavy |
| A100-80 | 80 GB HBM2e | 312 | 2.04 | $2.50 | 125 | Very large models |
| H100 | 80 GB HBM3 | 990 | 3.35 | $3.95 | 251 | Fast production training |
| H200 | 141 GB HBM3e | 990 | 4.80 | $4.54 | 218 | Auto-upgrade from H100 |
| **B200** | **192 GB HBM3e** | **2,250** | **8.00** | **$6.25** | **360** | **Fastest, best perf/$** |

**Recommendations for this project:**
- **L40S ($1.95/hr):** 3x A10G compute for <2x price. 48GB VRAM fits dataset in GPU cache. Best budget training option.
- **B200 ($6.25/hr):** 2.3x H100 compute at 1.6x price. A 3.2hr H100 run (~$13) takes ~1.4hr on B200 (~$8.75) — cheaper AND faster.
- **H100 ($3.95/hr):** Still solid, may auto-upgrade to H200. Use `"H100!"` to prevent upgrade.
- **A10G ($1.10/hr):** Fine for debug/export, not cost-efficient for real training.
- Billed per-second, no minimum. CPU: $0.047/core/hr, RAM: $0.008/GiB/hr. No egress fees.

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
