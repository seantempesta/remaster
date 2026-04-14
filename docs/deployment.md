# Deployment Guide

All deployment options for the remaster pipeline, from fastest to most portable.

## Prerequisites

All pipelines require the student model checkpoint at `checkpoints/drunet_student/final.pth`. TensorRT-based pipelines additionally need an ONNX export and a built engine.

### ONNX Export

```bash
python tools/export_onnx.py
# Produces: checkpoints/drunet_student/drunet_student.onnx (FP16)
# For INT8 calibration: python tools/export_onnx.py --fp32
```

**Important:** PyTorch 2.11 defaults `torch.onnx.export()` to `dynamo=True` which produces opset 20 IR that TensorRT miscompiles (14.5 dB output). The export script uses `dynamo=False` (opset 17, TorchScript exporter).

### TensorRT Engine Build

```bash
# FP16 engine (one-time, ~2 min)
trtexec --onnx=checkpoints/drunet_student/drunet_student.onnx \
    --shapes=input:1x3x1080x1920 --fp16 --useCudaGraph \
    --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
    --saveEngine=checkpoints/drunet_student/drunet_student_1080p_fp16.engine
```

Engines are GPU-specific -- rebuild when changing GPUs or driver versions.

### INT8 Mixed-Precision Engine

INT8 gives ~3% higher throughput (57 vs 56 fps) with minimal quality loss (67.2 dB vs 68.0 dB).

```bash
# Build calibration data from source frames
python tools/build_int8_calibration.py

# Build INT8 engine with sensitive layers forced to FP16
python tools/build_int8_engine.py
```

Skip-connection Add ops, head/tail convolutions, and transposed convolutions are forced to FP16 to preserve quality. Pure INT8 on all layers produces 26 dB (unusable) without QAT.

---

## C++ Zero-Copy Pipeline (57 fps)

The fastest option. NVDEC decode, TensorRT inference, and NVENC encode all run on the GPU with zero CPU round-trips.

```bash
pipeline_cpp/build/remaster_pipeline.exe \
    -i input.mkv -o output.mkv \
    -e checkpoints/drunet_student/drunet_student_1080p_fp16.engine --cq 20
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-i` | required | Input video file |
| `-o` | required | Output MKV file |
| `-e` | required | TensorRT engine file |
| `--cq` | 24 | Constant quality (lower = higher quality, 18-24 recommended) |
| `--preset` | p4 | NVENC preset (p1=fastest, p7=best quality) |
| `--no-audio` | false | Skip audio/subtitle passthrough |
| `--8bit` | false | 8-bit output (default is 10-bit HEVC) |

**Output:** 10-bit HEVC MKV with audio/subtitle passthrough. BT.709 color metadata. Keyframes every 5 seconds for fast seeking.

**Building from source:** See the [C++ build section](#building-the-c-pipeline) below.

---

## NVEncC + VapourSynth (39 fps)

VapourSynth runs in-process with NVEncC -- no pipe overhead.

```bash
python remaster/encode_nvencc.py input.mkv output.mkv
```

Requires [NVEncC](https://github.com/rigaya/NVEnc) and [VapourSynth](https://www.vapoursynth.com/) with [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) installed.

---

## VapourSynth + ffmpeg (20 fps)

Standard pipe-based encoding. Wider compatibility.

```bash
python remaster/encode.py input.mkv output.mkv
```

---

## Python Streaming (24 fps)

Pure Python with `torch.compile`. No TensorRT or VapourSynth needed.

```bash
python pipelines/remaster.py -i input.mkv \
    -c checkpoints/drunet_student/final.pth \
    --nc-list 16,32,64,128 --nb 2 \
    --encoder hevc_nvenc --mux-audio --compile
```

---

## Real-Time Playback

Configure mpv with VapourSynth for live enhancement during playback:

```
# mpv.conf
hwdec=auto-copy
vf=vapoursynth="/path/to/remaster/play.vpy"
```

Uses TensorRT via vs-mlrt for real-time inference.

---

## Benchmarks

Measured on RTX 3060 Laptop GPU (6GB), 1920x1080 10-bit HEVC, NVIDIA driver 595.97.

### End-to-End Pipeline Speed

| Pipeline | FPS | Per-Frame | Notes |
|----------|-----|-----------|-------|
| C++ FP16 | 55.7 | 17.9 ms | Zero-copy GPU, CUDA graphs |
| C++ INT8 | 57.2 | 17.5 ms | Mixed precision (Add ops FP16) |
| NVEncC | 39 | 25.6 ms | VapourSynth in-process |
| Python | 24 | 41.7 ms | torch.compile, no TRT |
| VS+ffmpeg | 20 | 50.0 ms | Pipe-based |

### TensorRT Inference Only (trtexec)

| Engine | GPU Compute | Throughput | Quality |
|--------|-------------|------------|---------|
| FP16 | 15.8 ms | 50.7 qps | 68.0 dB vs FP32 |
| INT8 mixed | 15.5 ms | 51.4 qps | 67.2 dB vs FP32 |

### Pipeline Breakdown (C++ FP16)

| Stage | Time/Frame | % of Wall |
|-------|-----------|-----------|
| TRT Inference | 17.1 ms | 95.3% |
| CSC (NV12<->RGB) | 0.2 ms | 1.2% |
| Encode (NVENC) | 0.3 ms | 1.5% |
| Sync + overhead | 0.3 ms | 2.0% |
| **Total** | **17.9 ms** | **55.7 fps** |

NVDEC decode and NVENC encode are fully hidden on async CUDA streams. TRT inference is the sole bottleneck.

---

## Building the C++ Pipeline

The C++ pipeline must be built from a **Visual Studio Developer Command Prompt** (not bash).

### Prerequisites

1. **Visual Studio Build Tools 2022** with C++ desktop workload
   - Includes MSVC compiler, `lib.exe`, `dumpbin.exe` (used to generate import libs from TRT DLLs)

2. **CUDA Toolkit 13.0+** ([download](https://developer.nvidia.com/cuda-toolkit))
   - CMake finds it via `find_package(CUDAToolkit)`

3. **CMake 3.18+** ([download](https://cmake.org/download/))

4. **Git submodules initialized** (NVIDIA Video Codec SDK for NVDEC decoder, plus KAIR for model architecture)
   ```bash
   git submodule update --init --recursive
   ```

### Download Dependencies

The build needs TensorRT headers and FFmpeg shared libraries. The CMakeLists.txt looks for them in `tools/_downloads/`:

**TensorRT headers** (CMake will error with download instructions if missing):
```bash
# Download TRT 10.16 source (headers only, not the full SDK)
cd tools/_downloads
wget https://github.com/NVIDIA/TensorRT/archive/refs/tags/v10.16.zip
unzip v10.16.zip
# Unzips to TensorRT-10.16/ -- rename to match CMake expected path
mv TensorRT-10.16 TensorRT-10.16.0
# Verify: tools/_downloads/TensorRT-10.16.0/include/NvInfer.h should exist
```

TRT runtime DLLs come from the [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) release bundle. Download the CUDA variant and extract to `tools/vs/vs-plugins/vsmlrt-cuda/`. The key files needed are `nvinfer_10.dll` and `nvonnxparser_10.dll`. Import `.lib` files are auto-generated from the DLLs by CMake at configure time using `dumpbin` and `lib`.

**FFmpeg shared build**:
```bash
# Download from gyan.dev (GPL shared build)
cd tools/_downloads
wget https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z
7z x ffmpeg-release-full-shared.7z
# Rename to match CMake search path
mv ffmpeg-*-win64-gpl-shared* ffmpeg-n7.1-latest-win64-gpl-shared-7.1
```

### Build

Open a **VS Developer Command Prompt** (not bash, not PowerShell):

```cmd
cd pipeline_cpp
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Runtime DLLs

The executable needs TRT and FFmpeg DLLs on PATH at runtime. Either:
- Add `tools/vs/vs-plugins/vsmlrt-cuda/` and `tools/_downloads/ffmpeg-.../bin/` to PATH, or
- Copy the required DLLs next to the executable

### Verify

```cmd
pipeline_cpp\build\Release\remaster_pipeline.exe --help
```

### Troubleshooting

- **"NvInfer.h not found"** -- TRT headers not at expected path. Check `tools/_downloads/TensorRT-10.16.0/include/`
- **"Could not find FFmpeg"** -- Set `FFMPEG_ROOT` env var to your FFmpeg install, or place it in `tools/_downloads/`
- **Link errors with nvcuvid** -- Make sure the video-sdk-samples submodule is initialized
- **CUDA arch mismatch** -- CMakeLists.txt targets sm_86 (RTX 3060) and sm_89 (RTX 40xx). Edit `CUDA_ARCHITECTURES` for other GPUs
