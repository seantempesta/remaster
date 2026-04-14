# remaster_pipeline -- GPU Video Enhancement

Native C++ pipeline: Decode -> CUDA color convert -> TensorRT inference -> CUDA color convert -> NVENC encode. All processed frames stay on the GPU -- no Python, no VapourSynth, no stdio pipes.

Decoding uses NVDEC hardware by default, with automatic fallback to FFmpeg software decode for codecs/profiles NVDEC doesn't support (e.g., H264 High 10-bit on RTX 3060, AV1 on older GPUs, or any format FFmpeg can handle).

## Architecture

```
Input Video (MKV/MP4)
    |
    v
SimpleDemuxer (FFmpeg container parsing, CPU)
    |
    v  compressed packets
NvDecoder (NVDEC hardware)  -or-  SwDecoder (FFmpeg software + GPU upload)
    |                               |
    +-------------------------------+
    |
    v  GPU surface (NV12 or P010)
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
MkvMuxer (MKV output with audio/subtitle passthrough)
```

The decoder selection is automatic:
1. Check `cuvidGetDecoderCaps` for NVDEC support of the codec + bit depth
2. If supported, use NVDEC (zero-copy GPU decode, highest throughput)
3. If not, fall back to FFmpeg software decode (CPU decode + pinned memory upload to GPU)
4. Use `--sw-decode` to force software decode for testing/debugging

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
| `--no-audio` | off | Skip audio/subtitle passthrough |
| `--sw-decode` | off | Force FFmpeg software decode (skip NVDEC) |

## Performance Notes

- Expected throughput: 50-60+ fps on RTX 3060 at 1080p
- The model (DRUNet student, 1.06M params) runs at ~55 fps in TRT FP16/INT8
- Bottleneck shifts to NVENC at high quality presets (p6/p7)
- All GPU memory usage is under 1 GB (model ~100 MB, decode/encode buffers ~200 MB)

## File Layout

```
pipeline_cpp/
  CMakeLists.txt        Build system
  main.cpp              Pipeline orchestration, CLI, decoder selection, progress reporting
  sw_decoder.h/cpp      FFmpeg software decode fallback (planar YUV -> NV12/P010 -> GPU upload)
  simple_demuxer.h      FFmpeg demuxer (container parsing, BSF filtering)
  trt_inference.h/cpp   TensorRT engine loader and inference wrapper
  color_kernels.h/cu    CUDA kernels for NV12/P010 <-> RGB FP16 conversion
  mkv_muxer.h           MKV container muxer with audio/subtitle passthrough
  async_writer.h        Thread-safe async packet writer for MKV muxing
  README.md             This file
```
