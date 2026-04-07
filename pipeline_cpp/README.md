# remaster_pipeline -- GPU-Only Video Enhancement

Native C++ pipeline: NVDEC decode -> CUDA color convert -> TensorRT inference -> CUDA color convert -> NVENC encode. All frames stay on the GPU -- no CPU round-trips, no Python, no VapourSynth, no stdio pipes.

## Architecture

```
Input Video (MKV/MP4)
    |
    v
FFmpegDemuxer (container parsing, CPU)
    |
    v  compressed packets
NvDecoder (NVDEC hardware decode -> GPU NV12/P010)
    |
    v  GPU surface
CUDA kernel: NV12/P010 -> Planar RGB FP16 [0,1] (BT.709)
    |
    v  GPU tensor (1x3xHxW fp16)
TensorRT inference (DRUNet student model)
    |
    v  GPU tensor (1x3xHxW fp16)
CUDA kernel: Planar RGB FP16 -> NV12/P010 (BT.709)
    |
    v  GPU surface
NvEncoderCuda (NVENC hardware encode -> HEVC)
    |
    v  compressed packets
Output file (raw HEVC bitstream)
```

## Prerequisites

- **Windows 11**, MSVC 2022 (Visual Studio 2022 with C++ workload)
- **CUDA Toolkit 12.x** (tested with 12.6)
- **NVIDIA GPU** with NVDEC + NVENC (RTX 3060 or newer)
- **TensorRT 10.x** -- headers and libraries
  - Can use the bundled runtime in `tools/vs/vs-plugins/vsmlrt-cuda/`
  - Or install standalone from NVIDIA
- **FFmpeg** shared libraries (avcodec, avformat, avutil)
  - Download from https://github.com/BtbN/FFmpeg-Builds/releases (shared build)
  - Or use gyan.dev builds
- **NVIDIA Video Codec SDK** -- already included in `reference-code/video-sdk-samples/`

## Build

```powershell
# Set environment variables pointing to your installations
$env:TENSORRT_ROOT = "C:\TensorRT-10.x"
$env:FFMPEG_ROOT = "C:\ffmpeg"

# Configure and build
cd pipeline_cpp
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The executable will be at `build/Release/remaster_pipeline.exe`.

## Usage

```bash
# Basic usage
remaster_pipeline.exe \
    --input "D:/video/episode.mkv" \
    --output "D:/video/episode_enhanced.hevc" \
    --engine "checkpoints/drunet_student/drunet_student_1080p_fp16.engine"

# With custom quality and preset
remaster_pipeline.exe \
    --input "D:/video/episode.mkv" \
    --output "D:/video/episode_enhanced.hevc" \
    --engine "checkpoints/drunet_student/drunet_student_1080p_fp16.engine" \
    --cq 20 --preset p5

# 10-bit output
remaster_pipeline.exe \
    --input "D:/video/episode.mkv" \
    --output "D:/video/episode_enhanced.hevc" \
    --engine "checkpoints/drunet_student/drunet_student_1080p_fp16.engine" \
    --10bit
```

### Mux audio afterward

The pipeline outputs a raw HEVC bitstream. To combine with audio from the original:

```bash
ffmpeg -i episode_enhanced.hevc -i episode.mkv -map 0:v -map 1:a -c copy output.mkv
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` / `-i` | (required) | Input video file |
| `--output` / `-o` | (required) | Output HEVC bitstream |
| `--engine` / `-e` | (required) | TensorRT engine file |
| `--gpu` | 0 | GPU device index |
| `--cq` | 24 | NVENC constant quality (lower = better, 0-51) |
| `--preset` | p4 | NVENC preset (p1=fastest .. p7=best quality) |
| `--10bit` | off | Output 10-bit HEVC |

## Performance Notes

- Expected throughput: 50-60+ fps on RTX 3060 at 1080p
- The model (DRUNet student, 1.06M params) runs at ~55 fps in TRT FP16/INT8
- Bottleneck shifts to NVENC at high quality presets (p6/p7)
- All GPU memory usage is under 1 GB (model ~100 MB, decode/encode buffers ~200 MB)

## File Layout

```
pipeline_cpp/
  CMakeLists.txt        Build system
  main.cpp              Pipeline orchestration, CLI, progress reporting
  trt_inference.h/cpp   TensorRT engine loader and inference wrapper
  color_kernels.h/cu    CUDA kernels for NV12/P010 <-> RGB FP16 conversion
  README.md             This file
```
