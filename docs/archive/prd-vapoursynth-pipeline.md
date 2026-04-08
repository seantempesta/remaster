# PRD: Production VapourSynth + TensorRT Pipeline

## Goal

Replace the Python GIL-bottlenecked pipeline (5-7 fps) with a C++ pipeline via VapourSynth + vs-mlrt + TensorRT that runs the full decode → inference → encode path without Python in the hot loop. Target: **40+ fps batch encoding**, **real-time 24fps playback**, **<5 second startup** after initial engine build.

## Non-Goals

- Custom C++ pipeline from scratch (vs-mlrt already does this)
- Training or model changes
- Supporting resolutions other than 1080p (single fixed engine)
- Cloud/Modal deployment (this is local RTX 3060 only)

---

## Architecture

```
┌─────────────┐    ┌────────────────┐    ┌─────────────┐    ┌──────────────┐
│ Source       │    │ VapourSynth    │    │ vs-mlrt      │    │ Output       │
│ (BestSource  │───▶│ resize:        │───▶│ TensorRT     │───▶│ vspipe y4m   │
│  or lsmas)   │    │ YUV→RGBS       │    │ FP16 engine  │    │ → ffmpeg     │
│              │    │ BT.709         │    │ batch=1      │    │ NVENC HEVC   │
└─────────────┘    └────────────────┘    └─────────────┘    └──────────────┘
       ↑                                        ↑                    ↑
    C++ decode                          C++ TensorRT           C++ encode
    (no GIL)                            (no GIL)               (no GIL)
```

Everything runs in VapourSynth's C++ thread pool. Python only executes once during graph construction (the .vpy script), then exits the picture entirely.

---

## Model Format: ONNX → Pre-built TensorRT Engine

### Why pre-build the engine

vs-mlrt can accept ONNX and build TensorRT engines on first run, but this takes **2-5 minutes** — unacceptable for production. We pre-build the engine once and point vs-mlrt at the serialized `.engine` file directly.

### Engine build steps

**One-time, done on the target machine (RTX 3060):**

```bash
# trtexec ships with the tensorrt pip package
# Location: C:/Users/sean/miniconda3/envs/upscale/Lib/site-packages/tensorrt_cu12/bin/trtexec.exe

trtexec \
  --onnx=checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx \
  --saveEngine=checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.engine \
  --fp16 \
  --inputIOFormats=fp16:chw \
  --outputIOFormats=fp16:chw \
  --workspace=2048 \
  --buildOnly
```

| Parameter | Value | Why |
|-----------|-------|-----|
| `--fp16` | enabled | Matches training precision, 2x throughput on Ampere tensor cores |
| `--inputIOFormats=fp16:chw` | fp16 I/O | Avoids fp32→fp16 cast at engine boundary |
| `--outputIOFormats=fp16:chw` | fp16 I/O | Same |
| `--workspace=2048` | 2 GB | Algorithm selection workspace (used during build only) |
| `--buildOnly` | skip profiling | Faster build, we don't need latency reports |

**Expected output:** ~110-130 MB `.engine` file (larger than ONNX because it includes fused GPU kernels).

**Build time:** ~3-5 minutes on RTX 3060.

**Engine portability:** The engine is locked to this exact GPU (RTX 3060 SM 8.6) + TensorRT version + CUDA version. Keep the ONNX file — rebuild if any of these change.

### INT8 engine (optional, for max speed)

If FP16 throughput isn't enough, INT8 adds ~1.5-2x. Requires calibration:

```bash
# First generate calibration cache (needs ~200 representative frames)
python tools/trt_calibrate.py \
  --onnx checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx \
  --data data/calibration_frames/ \
  --cache checkpoints/nafnet_w32_mid4/calibration.cache

# Then build INT8 engine with calibration
trtexec \
  --onnx=checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx \
  --saveEngine=checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920_int8.engine \
  --fp16 --int8 \
  --calib=checkpoints/nafnet_w32_mid4/calibration.cache \
  --workspace=2048 \
  --buildOnly
```

Defer INT8 until FP16 pipeline is validated end-to-end.

---

## Implementation Steps

### Phase 1: Environment Setup

**Step 1.1 — Install VapourSynth portable**

```bash
# Download R74 portable from GitHub releases
# Extract to C:\tools\vapoursynth\
# Add C:\tools\vapoursynth\ to PATH
# Verify:
vspipe --version
```

VapourSynth portable ships its own Python — it doesn't conflict with conda.

**Step 1.2 — Install vs-mlrt**

Download the latest release from https://github.com/AmusementClub/vs-mlrt/releases. Need the Windows TensorRT build matching your TRT version.

```
# Extract and copy to VapourSynth plugin directory:
# C:\tools\vapoursynth\vapoursynth64\plugins\
#   vstrt.dll
#   vsort.dll        (ONNX Runtime backend — optional but useful for debugging)
#   models/           (vs-mlrt ships bundled models, we don't use them)
#
# vs-mlrt release also ships required TensorRT + cuDNN DLLs.
# These go in the same plugins directory or on PATH.
```

**Step 1.3 — Install mpv**

```bash
# Download from https://github.com/shinchiro/mpv-winbuild-cmake/releases
# Extract to C:\tools\mpv\
# mpv must be a build with VapourSynth support (shinchiro builds include it)
```

**Step 1.4 — Install BestSource or L-SMASH-Works**

VapourSynth needs a source filter to decode video. Two options:

| Filter | Pros | Cons |
|--------|------|------|
| **BestSource** (`bs.VideoSource`) | Frame-accurate seeking, modern | Slower first open (builds index) |
| **L-SMASH-Works** (`lsmas.LWLibavSource`) | Fast, widely used | Less accurate seeking |

For batch encoding (sequential access), either works. BestSource is better for mpv (seeking).

```bash
# Download from: https://github.com/vapoursynth/bestsource/releases
# or: https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works/releases
# Copy .dll to C:\tools\vapoursynth\vapoursynth64\plugins\
```

**Step 1.5 — Install vsmlrt Python module**

The `vsmlrt.py` wrapper script must be importable by VapourSynth's Python:

```bash
# Copy vsmlrt.py from the vs-mlrt release to VapourSynth's Python site-packages
# Usually: C:\tools\vapoursynth\Lib\site-packages\vsmlrt.py
```

### Phase 2: Pre-build TensorRT Engine

**Step 2.1 — Build FP16 engine with trtexec**

```bash
# Using trtexec from the tensorrt pip package in the upscale conda env
PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/Lib/site-packages/tensorrt_cu12/bin/trtexec.exe \
  --onnx=checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx \
  --saveEngine=checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.engine \
  --fp16 \
  --inputIOFormats=fp16:chw \
  --outputIOFormats=fp16:chw \
  --workspace=2048 \
  --buildOnly
```

**Step 2.2 — Validate engine**

Quick sanity check that the engine loads and runs:

```bash
trtexec \
  --loadEngine=checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.engine \
  --fp16 \
  --iterations=10 \
  --warmUp=0
```

This should report latency ~13ms (matching PyTorch's 78 fps). If it's much slower, the engine build had issues.

**Step 2.3 — Verify vs-mlrt can load the engine**

vs-mlrt's TRT backend can accept `.engine` files directly via the `network_path` parameter (instead of `.onnx`). If vs-mlrt's bundled TRT version differs from the one used to build the engine, there will be a version mismatch error — this means you must use trtexec from the **same TRT version** that vs-mlrt ships.

Alternative: let vs-mlrt build the engine itself from ONNX on first run, then find and reuse the cached engine. vs-mlrt caches engines in a model-specific directory. Check vs-mlrt docs for the cache path (typically alongside the ONNX file or in `%LOCALAPPDATA%`).

### Phase 3: VapourSynth Script for Batch Encoding

**Step 3.1 — Write `playback/encode.vpy`**

Separate script from `enhance.vpy` (playback). Batch encoding has different requirements:
- Accepts input path as argument (not `video_in` from mpv)
- Uses BestSource/lsmas for decoding (not mpv's decoder)
- May use `num_streams=2` for better throughput (overlap inference on consecutive frames)
- Outputs 10-bit YUV for higher quality encoding

```python
"""
VapourSynth script for batch encoding with NAFNet denoising.

Usage:
  vspipe encode.vpy -c y4m -a "input=path/to/video.mkv" - | \
    ffmpeg -i pipe: -c:v hevc_nvenc -preset p4 -tune hq -rc vbr -cq 18 \
    -pix_fmt p010le output.mkv

  Or with audio passthrough:
  vspipe encode.vpy -c y4m -a "input=path/to/video.mkv" - | \
    ffmpeg -i pipe: -i path/to/video.mkv -map 0:v -map 1:a -c:v hevc_nvenc \
    -preset p4 -tune hq -rc vbr -cq 18 -pix_fmt p010le -c:a copy output.mkv
"""
import vapoursynth as vs
import os

core = vs.core

# Accept input path from command line: --arg input=path/to/video.mkv
input_path = globals()["input"]

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Use pre-built TensorRT engine for instant startup.
# Fall back to ONNX if engine doesn't exist (first run: 2-5 min build).
ENGINE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "nafnet_w32_mid4",
                           "nafnet_w32mid4_1088x1920.engine")
ONNX_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "nafnet_w32_mid4",
                          "nafnet_w32mid4_1088x1920.onnx")

if os.path.exists(ENGINE_PATH):
    MODEL_PATH = ENGINE_PATH
else:
    MODEL_PATH = ONNX_PATH

# Decode with BestSource (frame-accurate) or L-SMASH-Works
try:
    clip = core.bs.VideoSource(source=input_path)
except AttributeError:
    clip = core.lsmas.LWLibavSource(source=input_path)

# Store original dimensions
orig_w = clip.width
orig_h = clip.height

# Convert to RGBS (float32 planar RGB) with BT.709 matrix
# This is what NAFNet expects: [0.0, 1.0] float RGB
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

# Pad to model's fixed input size: 1088x1920
TARGET_H, TARGET_W = 1088, 1920
pad_h = TARGET_H - orig_h
pad_w = TARGET_W - orig_w
need_pad = (pad_h > 0 or pad_w > 0)

if need_pad:
    clip = core.std.AddBorders(clip, right=max(0, pad_w), bottom=max(0, pad_h))

# Run TensorRT inference via vs-mlrt
# num_streams=2: overlap inference on consecutive frames for higher throughput
import vsmlrt
clip = vsmlrt.inference(clip, network_path=MODEL_PATH,
                        backend=vsmlrt.Backend.TRT(
                            fp16=True,
                            num_streams=2,
                            device_id=0,
                        ))

# Crop padding
if need_pad:
    clip = core.std.Crop(clip, right=max(0, pad_w), bottom=max(0, pad_h))

# Convert to YUV420P10 for high-quality HEVC encoding (10-bit output)
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s="709")

clip.set_output()
```

**Step 3.2 — Write `tools/encode_episode.sh`**

Wrapper script for encoding full episodes:

```bash
#!/bin/bash
# Encode a video with NAFNet denoising via VapourSynth + TensorRT
#
# Usage: ./tools/encode_episode.sh input.mkv output.mkv [cq]
#
# cq: constant quality (default 18, lower = higher quality)

set -euo pipefail

INPUT="$1"
OUTPUT="$2"
CQ="${3:-18}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENCODE_VPY="$SCRIPT_DIR/playback/encode.vpy"
FFMPEG="$SCRIPT_DIR/bin/ffmpeg.exe"

echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "CQ:     $CQ"
echo ""

# Encode: vspipe → ffmpeg NVENC
# -p: show progress
vspipe "$ENCODE_VPY" -c y4m -a "input=$INPUT" -p - | \
  "$FFMPEG" -y \
    -i pipe: \
    -i "$INPUT" \
    -map 0:v:0 \
    -map 1:a? \
    -map 1:s? \
    -c:v hevc_nvenc \
    -preset p4 \
    -tune hq \
    -rc vbr \
    -cq "$CQ" \
    -pix_fmt p010le \
    -c:a copy \
    -c:s copy \
    "$OUTPUT"

echo ""
echo "Done: $OUTPUT"
```

### Phase 4: Playback Script Update

**Step 4.1 — Update `playback/enhance.vpy`**

Update the existing script to prefer pre-built engines and add `num_streams` config:

```python
"""
VapourSynth script for real-time NAFNet denoising during mpv playback.
Uses pre-built TensorRT engine for instant startup.
"""
import vapoursynth as vs
import os

core = vs.core

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Prefer pre-built engine (instant startup) over ONNX (2-5 min first build)
ENGINE_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "nafnet_w32_mid4",
                           "nafnet_w32mid4_1088x1920.engine")
ONNX_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "nafnet_w32_mid4",
                          "nafnet_w32mid4_1088x1920.onnx")
MODEL_PATH = ENGINE_PATH if os.path.exists(ENGINE_PATH) else ONNX_PATH

clip = video_in
orig_w = clip.width
orig_h = clip.height

clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

TARGET_H, TARGET_W = 1088, 1920
pad_h = TARGET_H - orig_h
pad_w = TARGET_W - orig_w
need_pad = (pad_h > 0 or pad_w > 0)

if need_pad:
    clip = core.std.AddBorders(clip, right=max(0, pad_w), bottom=max(0, pad_h))

import vsmlrt
clip = vsmlrt.inference(clip, network_path=MODEL_PATH,
                        backend=vsmlrt.Backend.TRT(
                            fp16=True,
                            num_streams=1,  # 1 stream for real-time (low latency)
                            device_id=0,
                        ))

if need_pad:
    clip = core.std.Crop(clip, right=max(0, pad_w), bottom=max(0, pad_h))

clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()
```

### Phase 5: Validation

**Step 5.1 — Smoke test with vspipe**

```bash
# Check that the script loads and can produce frames
vspipe playback/encode.vpy -a "input=data/clip_mid_1080p.mp4" -i
# Should print: Width: 1920, Height: 1080, Frames: 720, FPS: 24000/1001

# Output a single frame to verify correctness
vspipe playback/encode.vpy -a "input=data/clip_mid_1080p.mp4" -c y4m -e 0 -s 0 - > test_frame.y4m
```

**Step 5.2 — Benchmark throughput**

```bash
# Encode to null output (measure raw pipeline speed)
vspipe playback/encode.vpy -a "input=data/clip_mid_1080p.mp4" -p - > /dev/null

# Full pipeline with NVENC
vspipe playback/encode.vpy -c y4m -a "input=data/clip_mid_1080p.mp4" -p - | \
  bin/ffmpeg.exe -i pipe: -c:v hevc_nvenc -preset p4 -tune hq -rc vbr -cq 18 \
  -pix_fmt p010le data/clip_mid_1080p_vsmlrt.mkv
```

**Expected results:**

| Metric | Target | Reasoning |
|--------|--------|-----------|
| vspipe only (no encode) | 50-78 fps | TRT inference bound (~13ms/frame) |
| vspipe + ffmpeg NVENC | 40-60 fps | NVENC at P4 is ~150 fps, won't bottleneck |
| Startup time (pre-built engine) | <5 seconds | Engine deserialization, no compilation |
| Startup time (ONNX, first run) | 2-5 minutes | TRT compilation, cached after |
| VRAM usage | ~1.0-1.5 GB | TRT engine + I/O buffers (no PyTorch) |
| System RAM | <500 MB | VapourSynth + vs-mlrt + frame buffers |

**Step 5.3 — Quality comparison**

Compare output against existing Python pipeline output to verify numerical equivalence:

```bash
# Generate test clip with both pipelines
# Python pipeline (existing):
python pipelines/denoise_nafnet.py --input data/clip_mid_1080p.mp4 \
  --output data/clip_compare_python.mkv --width 32 --middle-blk-num 4

# VapourSynth pipeline (new):
vspipe playback/encode.vpy -c y4m -a "input=data/clip_mid_1080p.mp4" -p - | \
  bin/ffmpeg.exe -i pipe: -c:v hevc_nvenc -preset p4 -cq 18 -pix_fmt p010le \
  data/clip_compare_vsmlrt.mkv

# Compare with existing bench tool
python bench/compare.py --source data/clip_mid_1080p.mp4 \
  --a data/clip_compare_python.mkv --b data/clip_compare_vsmlrt.mkv
```

PSNR between the two outputs should be >50 dB (minor differences from TRT kernel fusion and y4m round-trip are expected).

**Step 5.4 — Playback test**

```bash
mpv "data/clip_mid_1080p.mp4"
# Ctrl+E to toggle enhancement
# Check: no stuttering, no color shift, no artifacts at scene cuts
```

### Phase 6: Production Encoding Script

**Step 6.1 — Write `tools/encode_batch.sh`**

For encoding multiple episodes:

```bash
#!/bin/bash
# Encode all .mkv files in a directory with NAFNet denoising
#
# Usage: ./tools/encode_batch.sh /path/to/input/ /path/to/output/ [cq]

set -euo pipefail

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CQ="${3:-18}"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.mkv; do
  base="$(basename "$f")"
  out="$OUTPUT_DIR/$base"

  if [ -f "$out" ]; then
    echo "SKIP (exists): $base"
    continue
  fi

  echo "ENCODE: $base"
  ./tools/encode_episode.sh "$f" "$out" "$CQ"
done

echo "Batch complete."
```

---

## VRAM Budget

| Component | VRAM | Notes |
|-----------|------|-------|
| TensorRT engine weights (FP16) | ~120 MB | Fused kernels + weights |
| Input buffer (1x3x1088x1920 fp16) | 12 MB | |
| Output buffer (1x3x1088x1920 fp16) | 12 MB | |
| TRT execution workspace | ~200-400 MB | Layer activations |
| CUDA/driver overhead | ~300 MB | Context, allocator |
| VapourSynth frame buffers | ~50-100 MB | Decode + convert buffers |
| **Total** | **~700 MB - 1.0 GB** | vs 2.3 GB PyTorch, vs 3.3 GB PyTorch w64 |

Leaves ~5 GB free on the 6 GB RTX 3060 — plenty of headroom for NVDEC/NVENC concurrent use.

---

## Startup Time Budget

| Phase | Time | Notes |
|-------|------|-------|
| VapourSynth init | <100 ms | Plugin loading |
| Source filter open + index | 0.5-2 s | BestSource builds lightweight index |
| TRT engine deserialize | 1-3 s | From pre-built .engine file |
| First frame warmup | ~100 ms | TRT context init, CUDA warmup |
| **Total** | **2-5 s** | vs 2-5 min with ONNX → TRT build |

Key: the `.engine` file eliminates the compilation step entirely. TRT deserialization is just reading the binary blob into GPU memory.

---

## Color Space Handling

Current Python pipeline does full-range RGB (0-255 → 0.0-1.0, no limited range). The VapourSynth pipeline must match:

1. **Decode**: Source filter outputs YUV (whatever the file has, typically YUV420P8 or YUV420P10 BT.709)
2. **Convert to RGBS**: `core.resize.Bicubic(format=vs.RGBS, matrix_in_s="709")` — this converts from BT.709 YUV to full-range float RGB [0.0, 1.0]
3. **Model inference**: Operates on [0.0, 1.0] float RGB (same as Python pipeline)
4. **Convert to output YUV**: `core.resize.Bicubic(format=vs.YUV420P10, matrix_s="709")` — back to BT.709 YUV

The `matrix_in_s="709"` and `matrix_s="709"` parameters handle the color matrix correctly. VapourSynth's resizer (based on zimg) handles limited/full range conversion automatically based on the source metadata.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| vs-mlrt TRT version != system TRT version | Medium | Use trtexec from vs-mlrt's bundled TRT, or let vs-mlrt build the engine itself |
| BestSource/lsmas can't decode some files | Low | Both support HEVC/H.264; fall back to the other |
| TRT engine slower than expected | Low | ONNX model is simple CNN, TRT optimizes well; benchmark with trtexec first |
| Color difference vs Python pipeline | Medium | Both use BT.709 + full-range RGB; validate with PSNR comparison |
| vs-mlrt doesn't accept .engine directly | Medium | Let it build from ONNX on first run, cache auto-reused; 2-5 min penalty once |

---

## vs-mlrt Engine Path Compatibility

vs-mlrt may not accept raw `.engine` files — it may require ONNX and build/cache engines internally. If so:

**Option A**: Point vs-mlrt at the ONNX file, let it build the engine on first run. It caches the result. First run is 2-5 min, all subsequent runs are instant. This is the path of least resistance.

**Option B**: Find vs-mlrt's engine cache directory and place the pre-built engine there with the expected filename. Check the vs-mlrt source/docs for the cache naming convention (typically includes model hash + GPU name + TRT version).

**Option C**: Use vs-mlrt's ORT (ONNX Runtime) backend with the TensorRT execution provider. This also builds/caches TRT engines but through a different code path.

Recommendation: **Start with Option A.** The 2-5 minute first-run cost is acceptable for a one-time event. After that, startup is instant. Only pursue Option B if the auto-cached engine is slower than a manually-built one (unlikely).

---

## File Inventory

After implementation, the project will have:

```
playback/
  enhance.vpy          — mpv real-time playback script (updated)
  encode.vpy           — batch encoding script (new)
  README.md            — updated with engine build + batch encoding docs

tools/
  encode_episode.sh    — single episode encoding wrapper (new)
  encode_batch.sh      — batch directory encoding (new)

checkpoints/nafnet_w32_mid4/
  nafnet_w32mid4_1088x1920.onnx     — 57 MB, source of truth (existing)
  nafnet_w32mid4_1088x1920.engine   — ~120 MB, pre-built TRT FP16 (new)

docs/
  prd-vapoursynth-pipeline.md       — this document
```

---

## Success Criteria

1. `vspipe encode.vpy -p` shows **40+ fps** on RTX 3060
2. Full pipeline (vspipe + ffmpeg NVENC) achieves **30+ fps** sustained
3. Startup to first frame output in **<5 seconds** (with pre-built engine)
4. VRAM usage under **1.5 GB** during encoding
5. Output quality within **0.5 dB PSNR** of Python pipeline output
6. A single command encodes a full episode with audio/subtitle passthrough
7. mpv real-time playback at 24fps with no dropped frames
