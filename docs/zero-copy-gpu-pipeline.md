# Zero-Copy GPU Video Pipeline Research

Research into building an NVDEC decode -> PyTorch inference -> NVENC encode pipeline
on Windows 11, RTX 3060 6GB, with no CPU round-trips.

**Date:** 2026-04-02

## Current Pipeline (Baseline)

```
File -> PyAV (CPU decode) -> numpy -> torch.from_numpy() -> .cuda() -> model -> .cpu() -> numpy -> pipe to ffmpeg -> NVENC encode -> File
```

CPU-GPU transfers happen twice: upload after decode, download before encode. With NAFNet
at 78 fps raw inference, these transfers are the bottleneck.

---

## Approach 1: PyNvVideoCodec (NVIDIA Official) -- RECOMMENDED

**What it is:** NVIDIA's official Python library for NVDEC/NVENC, successor to VPF
(Video Processing Framework). MIT license, actively maintained, latest release January
2026 (v2.0).

**Repo:** https://github.com/NVIDIA/VideoProcessingFramework

**Install:** `pip install PyNvVideoCodec` (Windows wheels available for Python 3.10, 3.11)

### Capabilities

- **Decode to GPU tensor (zero-copy):** Yes. `SimpleDecoder` with `OutputColorType.RGBP`
  produces planar CHW frames directly on GPU. Convert to PyTorch via
  `torch.from_dlpack(frame)` with zero-copy -- the tensor shares GPU memory.
- **Encode from GPU tensor:** Yes. Encoder supports GPU buffer mode
  (`usecpuinputbuffer=False`). Accepts NV12, YUV444, ARGB, ABGR formats.
- **Color space handling:** Decodes to RGBP (planar RGB, CHW). For encoding, NVENC
  requires NV12/YUV input. GPU-side RGB->NV12 conversion needed (see CV-CUDA below).

### Decode Pipeline (pseudocode)
```python
import PyNvVideoCodec as nvc
import torch

decoder = nvc.SimpleDecoder(
    "input.mp4",
    device_id=0,
    out_color_type=nvc.OutputColorType.RGBP  # CHW planar RGB
)

for frame in decoder:
    # Zero-copy to PyTorch tensor on GPU
    tensor = torch.from_dlpack(frame)        # shape: [3, H, W], on cuda:0
    tensor = tensor.unsqueeze(0).half() / 255.0  # [1, 3, H, W] fp16
    output = model(tensor)
    # ... encode output
```

### Encode Pipeline (pseudocode)
```python
encoder = nvc.Encoder(
    width=1920, height=1080,
    codec="h264",  # or "hevc"
    device_id=0,
    usecpuinputbuffer=False  # GPU buffer mode
)

# Model output is RGB float16 [1, 3, H, W]
# Need to convert to NV12 for NVENC
# Option A: Use CV-CUDA for GPU-side color conversion
# Option B: Use custom CUDA kernel
# Option C: Encode as YUV444 (less compression efficiency)
```

### Encode Color Conversion Problem

This is the main complexity. NVENC wants NV12 (YUV420). Our model outputs RGB float16.
Options:
1. **CV-CUDA** (`pip install cvcuda`): Has `cvtcolor_into()` with RGB2YUV_NV12.
   GPU-side, zero-copy. NVIDIA's recommended approach. **But: Linux-only as of 2025.**
2. **Custom CUDA kernel via cupy:** Write RGB->NV12 conversion. ~20 lines of CUDA code.
3. **PyNvVideoCodec ARGB input:** Encoder accepts ARGB/ABGR directly. Reformat RGB
   tensor to ARGB (add alpha channel) and encode. Simplest path.
4. **Encode YUV444:** Less common but avoids chroma subsampling conversion entirely.

### Assessment

| Criterion | Rating |
|-----------|--------|
| Windows compatibility | Yes (pip wheels for Win x64, Python 3.10/3.11) |
| Python 3.10 + PyTorch 2.11 | Yes (DLPack interop, framework-agnostic) |
| Maturity | High -- NVIDIA official, Video Codec SDK underneath, v2.0 Jan 2026 |
| Zero-copy decode | Yes -- DLPack, shares GPU memory |
| Zero-copy encode | Yes -- GPU buffer mode, but needs RGB->NV12 conversion |
| Integration effort | Medium -- need to handle color conversion for encode |

**Verdict: Best option. Most mature, Windows-native, true zero-copy on both ends.**

---

## Approach 2: TorchCodec (PyTorch Official)

**What it is:** Meta's official PyTorch library for video decoding/encoding. Wraps FFmpeg
internally. Has CUDA backend using NVDEC since v0.7.

**Repo:** https://github.com/meta-pytorch/torchcodec

**Install:** `pip install torchcodec` (CPU). For CUDA: `conda install torchcodec-cuda126`

### Capabilities

- **Decode to GPU tensor:** Yes (Beta). Uses NVDEC via FFmpeg's CUVID decoder. Returns
  PyTorch CUDA tensors directly. Up to 3x speedup over CPU, up to 90% NVDEC utilization.
- **Encode from GPU tensor:** CPU encoding only as of v0.10 (2026). No NVENC support yet.
- **Directly integrated with PyTorch:** Returns `torch.Tensor` natively.

### Windows Support

**Problematic.** Windows support was added in v0.7 (Beta) but has ongoing DLL loading
issues. Multiple GitHub issues report failures:
- Issue #640: Windows Support tracking issue
- Issue #1233: DLL loading failures on Windows with CUDA
- Issue #1147: Import errors on Windows
- Requires FFmpeg "full-shared" build with matching DLLs

CUDA decoding on Windows is particularly fragile -- needs FFmpeg built with NVDEC/CUVID
support, plus libnpp and libnvrtc from CUDA toolkit.

### Assessment

| Criterion | Rating |
|-----------|--------|
| Windows compatibility | Poor -- Beta, many DLL issues, CUDA on Windows fragile |
| Python 3.10 + PyTorch 2.11 | Likely, but untested combo (version alignment issues) |
| Maturity | Medium -- actively developed but still Beta for GPU features |
| Zero-copy decode | Yes (CUDA backend) |
| Zero-copy encode | No -- CPU-only encoding, defeats the purpose |
| Integration effort | Low for decode (native PyTorch tensors), high for full pipeline |

**Verdict: Not viable for our use case. No GPU encoding, and Windows CUDA support is
unreliable. Worth monitoring for future versions.**

---

## Approach 3: NVIDIA DALI

**What it is:** GPU-accelerated data loading library for deep learning training.
Designed for batch training pipelines, not streaming video processing.

**Repo:** https://github.com/NVIDIA/DALI

### Capabilities

- **Video decoding:** Yes, uses NVDEC. Supports H.264, HEVC, VP9, MPEG4, etc.
- **PyTorch integration:** Yes -- outputs can be consumed as PyTorch tensors.
- **GPU preprocessing:** Resize, crop, color conversion, normalization all on GPU.

### Windows Support

**No native Windows support.** Linux-only. Can be used through WSL, but that adds
complexity and potential performance overhead (WSL GPU passthrough).

### Assessment

| Criterion | Rating |
|-----------|--------|
| Windows compatibility | No (Linux-only, WSL workaround) |
| Maturity | High -- production-grade, used in NVIDIA's own training pipelines |
| Zero-copy | Yes on Linux |
| Fit for streaming inference | Poor -- designed for training data loading, not video encoding |

**Verdict: Not suitable. Linux-only, and designed for training not streaming inference.**

---

## Approach 4: FFmpeg with Shared CUDA Context

**What it is:** Using FFmpeg's `-hwaccel cuda` decoding with a shared CUDA context that
PyTorch can access, avoiding the CPU round-trip.

### PyAV with Hardware Acceleration

PyAV has merged hardware decoding support (PR #1685). You can create an `HWAccel` context
with `device_type='cuda'`:

```python
import av
hwaccel = av.codec.hwaccel.HWAccel(device_type='cuda')
container = av.open("input.mp4", hwaccel=hwaccel)
```

**However:** The standard PyAV pip wheels do NOT include CUDA-enabled FFmpeg. You'd need
to build PyAV from source against a CUDA-enabled FFmpeg -- significant effort on Windows.

### PyAV-CUDA (Third-Party Extension)

**Repo:** https://github.com/materight/PyAV-CUDA

Extension of PyAV with hardware encoding and decoding support. Compatible with PyTorch
and NVIDIA codecs. Provides CUDA-accelerated kernels for color space conversion.

**Status:** Small project, likely not production-ready. Would need evaluation.

### The Core Problem

Even with CUDA-accelerated FFmpeg decoding, getting the decoded frame from FFmpeg's CUDA
context into PyTorch's CUDA context requires either:
1. A shared CUDA context (complex, fragile)
2. CUDA IPC (inter-process, overhead)
3. Copying through CPU (defeats the purpose)

FFmpeg's hardware frame surfaces are opaque -- they're not easily exported as raw CUDA
pointers that PyTorch can wrap. PyNvVideoCodec solves this with DLPack; FFmpeg does not
expose DLPack.

### Assessment

| Criterion | Rating |
|-----------|--------|
| Windows compatibility | Poor -- needs custom FFmpeg build |
| Zero-copy | Questionable -- CUDA context sharing is the unsolved problem |
| Integration effort | Very high |

**Verdict: Not practical. The CUDA context sharing problem is unsolved in the FFmpeg
ecosystem. PyNvVideoCodec exists precisely to solve this.**

---

## Approach 5: GStreamer + DeepStream

**What it is:** NVIDIA's full video analytics pipeline framework. Handles decode,
inference (via TensorRT), encode as a GStreamer pipeline.

**Repo:** https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

### Windows Support

**No.** DeepStream SDK 9.0 only supports Ubuntu 24.04 with Python 3.12. No Windows
support at all.

### Assessment

**Verdict: Non-starter. Linux-only, overkill for single-stream inference, and requires
TensorRT models (no raw PyTorch).**

---

## Approach 6: Custom CuPy + Raw NVDEC API

**What it is:** Using ctypes/cffi to call NVIDIA's cuviddec.h API directly, or wrapping
it through CuPy.

### Feasibility

CuPy itself does not wrap NVDEC. You'd be calling the Video Codec SDK C API through
ctypes, which means:
- Parsing video bitstreams yourself (or using a demuxer)
- Managing NVDEC decoder contexts
- Handling async decode callbacks
- Mapping decoded surfaces to CUDA memory

This is essentially reimplementing what PyNvVideoCodec already does, but worse.

**Verdict: Don't do this. PyNvVideoCodec is the official wrapper for exactly this API.**

---

## Recommended Architecture

Based on this research, the optimal pipeline is:

```
File -> PyNvVideoCodec NVDEC decode (GPU, RGBP)
     -> torch.from_dlpack() (zero-copy)
     -> normalize to fp16 (GPU)
     -> NAFNet inference (GPU)
     -> denormalize to uint8 (GPU)
     -> RGB->ARGB or RGB->NV12 conversion (GPU)
     -> PyNvVideoCodec NVENC encode (GPU)
     -> File
```

**Every step stays on GPU. Zero CPU round-trips.**

### Color Conversion Strategy

The encode-side color conversion (RGB -> NV12) is the one tricky part. Options ranked:

1. **ARGB direct encode** -- PyNvVideoCodec encoder accepts ARGB input format. Just add
   an alpha channel (trivial GPU op: `torch.cat([tensor, alpha], dim=0)`). NVENC handles
   the YUV conversion internally. Simplest approach, try this first.

2. **Custom CUDA kernel via CuPy** -- Write a simple RGB->NV12 kernel (~20 lines). Full
   control, no extra dependencies.

3. **CV-CUDA** -- NVIDIA's official GPU image processing library has `cvtcolor_into()`
   with RGB2YUV_NV12. Professional solution but **Linux-only** currently.

### Audio Passthrough

PyNvVideoCodec handles video only. Audio still needs to be handled separately:
- Demux audio with PyAV or ffmpeg subprocess (CPU, but audio is tiny)
- Mux final video + original audio with ffmpeg

### Implementation Plan

1. `pip install PyNvVideoCodec` in the upscale conda env
2. Write a proof-of-concept: decode one frame -> DLPack -> PyTorch tensor -> verify on GPU
3. Benchmark decode speed vs current PyAV CPU decode
4. Add NAFNet inference in the middle
5. Implement encode side (try ARGB input first)
6. Benchmark full pipeline vs current pipeline
7. Handle audio remux

---

## Reference Repos Worth Cloning

### 1. VideoJaNai
- **URL:** `https://github.com/the-database/VideoJaNai.git`
- **What:** Windows GUI for GPU-accelerated video upscaling using ONNX models with
  TensorRT and VapourSynth. The closest existing project to what we're building.
- **Relevance:** Shows how to build a complete Windows video enhancement pipeline with
  GPU acceleration. Uses TensorRT for inference, VapourSynth for frame handling.

### 2. vs-mlrt (VapourSynth ML Runtime)
- **URL:** `https://github.com/AmusementClub/vs-mlrt.git`
- **What:** Efficient CPU/GPU ML runtimes for VapourSynth with built-in support for
  SCUNet, waifu2x, DPIR, Real-ESRGAN, RIFE, and more.
- **Relevance:** Has SCUNet integration already. Shows how to wrap ML models for
  efficient video processing. Supports TensorRT, ONNX Runtime, and OpenVINO backends.

### 3. NVIDIA Video Processing Framework (VPF)
- **URL:** `https://github.com/NVIDIA/VideoProcessingFramework.git`
- **What:** NVIDIA's reference implementation with sample code for NVDEC/NVENC with
  PyTorch integration. PyNvVideoCodec is built from this codebase.
- **Relevance:** Contains sample applications showing decode->PyTorch->encode pipelines.
  The samples directory has exactly what we need as starting points.

### 4. NVIDIA video-sdk-samples
- **URL:** `https://github.com/NVIDIA/video-sdk-samples.git`
- **What:** Official Video Codec SDK samples including NvDecoder and NvEncoder C++ code.
- **Relevance:** Reference for understanding the low-level API that PyNvVideoCodec wraps.

### 5. REAL-Video-Enhancer
- **URL:** `https://github.com/TNTwise/REAL-Video-Enhancer.git`
- **What:** Cross-platform video enhancement tool (upscale, interpolate, denoise) using
  TensorRT and NCNN. Supports Windows/Linux/macOS.
- **Relevance:** Another complete video enhancement pipeline. Uses TensorRT for inference
  with scene change detection and preview capabilities.

### 6. CV-CUDA
- **URL:** `https://github.com/CVCUDA/CV-CUDA.git`
- **What:** NVIDIA's GPU-accelerated computer vision library. Has sample pipelines
  showing PyNvVideoCodec decode -> CV-CUDA preprocessing -> inference -> encode.
- **Relevance:** The samples directory contains complete video pipeline examples that
  demonstrate the exact architecture we want. Linux-only for runtime, but the code
  patterns are directly applicable.

### 7. TorchCodec
- **URL:** `https://github.com/meta-pytorch/torchcodec.git`
- **What:** PyTorch's official video codec library. Worth watching for future Windows
  CUDA maturity.
- **Relevance:** Monitor for when GPU encoding support ships and Windows stabilizes.

### 8. PyAV-CUDA
- **URL:** `https://github.com/materight/PyAV-CUDA.git`
- **What:** Extension of PyAV with hardware encoding/decoding and PyTorch integration.
- **Relevance:** Small project but demonstrates CUDA-accelerated color space conversion
  in Python, which is relevant for the encode side.

---

## Summary Comparison

| Approach | Windows | Zero-Copy Decode | Zero-Copy Encode | Maturity | Recommendation |
|----------|---------|------------------|------------------|----------|---------------|
| **PyNvVideoCodec** | Yes | Yes (DLPack) | Yes (GPU buffer) | High | **Use this** |
| TorchCodec | Fragile | Yes (Beta) | No (CPU only) | Medium | Monitor |
| DALI | No (WSL) | Yes | N/A (training) | High | Skip |
| FFmpeg/PyAV CUDA | Build from source | Partial | Partial | Low | Skip |
| DeepStream | No | Yes | Yes | High | Skip (Linux) |
| Raw NVDEC/CuPy | Yes | Manual | Manual | DIY | Skip |

## Key Takeaway

**PyNvVideoCodec is the clear winner.** It's NVIDIA's official solution, has Windows pip
wheels, supports Python 3.10, provides true zero-copy GPU tensors via DLPack for both
decode and encode, and is actively maintained (v2.0 released January 2026). The only
complexity is RGB->NV12 color conversion for the encode side, which can be solved by
encoding in ARGB format directly or writing a simple CuPy kernel.

Expected speedup: eliminating the CPU round-trips should let us approach the raw 78 fps
inference speed of NAFNet, minus NVDEC/NVENC overhead (~5-10%). Realistic target: 60-70
fps full pipeline, up from the current ~14 fps with CPU transfers.
