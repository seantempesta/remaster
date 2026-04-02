# Real-Time Video Enhancement: Encoding Pipeline & Playback Research

Research into using NAFNet (14M params, 78 fps raw inference, 51MB VRAM on RTX 3060) for real-time video enhancement during encoding or playback.

**Current bottleneck:** Raw GPU inference runs at 78 fps, but the full pipeline (decode + inference + x265 encode) runs at only 5.2 fps because x265 CPU encoding is the bottleneck.

---

## 1. Encoding Pipeline Optimization

### 1.1 Encoder Speed Comparison at 1080p

| Encoder | Type | Approx. 1080p Speed | Quality | Notes |
|---------|------|---------------------|---------|-------|
| **h264_nvenc P1** | GPU (NVENC) | 300-480 fps | Low | Fastest option, dedicated HW encoder |
| **h264_nvenc P4** | GPU (NVENC) | 150-250 fps | Medium | Good speed/quality balance |
| **hevc_nvenc P1** | GPU (NVENC) | 200-350 fps | Low | H.265 compression gains |
| **hevc_nvenc P4** | GPU (NVENC) | 100-180 fps | Medium | **Best candidate for our pipeline** |
| **hevc_nvenc P7** | GPU (NVENC) | 40-80 fps | High | May bottleneck inference |
| **x264 ultrafast** | CPU | 40-80 fps | Low | Depends on CPU cores |
| **x264 fast** | CPU | 15-30 fps | Medium | Reasonable quality |
| **x265 fast** | CPU | 5-10 fps | Medium-High | Current bottleneck at 5.2 fps |
| **SVT-AV1 preset 10** | CPU | 50-70 fps | Medium | Best CPU option for speed |
| **SVT-AV1 preset 8** | CPU | 15-30 fps | Medium-High | ~x265 medium quality |

**Key insight:** NVENC at P1-P4 can encode at 150-350+ fps at 1080p, far exceeding our 78 fps inference rate. The encoder would never be the bottleneck.

**RTX 3060 limitation:** No AV1 hardware encoding (NVENC AV1 requires RTX 4000+). AV1 encoding is CPU-only via SVT-AV1.

### 1.2 NVENC + PyTorch Inference on the Same GPU

**This works well.** NVENC is a dedicated hardware encoder physically separate from the CUDA cores:

- NVENC runs on a fixed-function ASIC on the GPU die, completely independent of CUDA/shader cores
- CUDA compute (PyTorch inference) and NVENC encoding run in parallel without competing for resources
- NVDEC (hardware decoder) is also separate -- decode, inference, and encode can all run simultaneously
- The only shared resource is VRAM bandwidth, but NVENC uses very little bandwidth compared to CUDA compute
- Exception: if input is RGBA, NVENC uses a small CUDA kernel for color space conversion. With NV12/YUV input, zero CUDA usage

**VRAM budget on RTX 3060 (6GB):**
- NAFNet fp16 inference: ~51MB
- NVENC encoder state: ~50-100MB
- NVDEC decoder state: ~30-50MB
- Frame buffers (a few 1080p frames): ~30MB
- Total: ~200MB -- leaves 5.8GB headroom

**Conclusion:** Running NAFNet + NVENC + NVDEC simultaneously on the RTX 3060 is entirely feasible. The 78 fps model inference is the bottleneck, not encoding.

### 1.3 AV1 Encoding Options

Since RTX 3060 lacks NVENC AV1:

- **SVT-AV1 preset 10-12:** 50-70+ fps at 1080p on modern CPUs. Could keep up with inference if CPU is fast enough. Better compression than HEVC at similar quality.
- **SVT-AV1 preset 8:** ~x265 medium quality, 15-30 fps. Would bottleneck.
- **Recommendation:** For archival quality, use hevc_nvenc P4 (keeps up with inference, decent quality). For maximum compression, run SVT-AV1 preset 8 as a second pass on the already-enhanced output.

### 1.4 Zero-Copy GPU Pipeline (Decode -> Inference -> Encode)

The ideal pipeline avoids CPU<->GPU data transfers entirely:

**Option A: TorchAudio StreamWriter with NVENC**

TorchAudio's `StreamWriter` supports `hw_accel="cuda:0"` to encode directly from CUDA tensors via NVENC without copying to CPU. Combined with NVDEC decoding, this enables a full GPU pipeline:

```
NVDEC decode (GPU) -> CUDA tensor -> NAFNet inference (GPU) -> CUDA tensor -> NVENC encode (GPU)
```

Requires torchaudio built with NVENC-enabled FFmpeg. This eliminates the ~6ms per frame CPU round-trip overhead.

**Option B: NVIDIA GMAT / FFmpeg-GPU-Demo**

NVIDIA's GMAT project (github.com/NVIDIA/GMAT) demonstrates a full GPU pipeline using a custom FFmpeg build:

```
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i input.mkv \
  -vf scale_npp=1920:1080,format_cuda=rgbpf32le,tensorrt=model.engine,format_cuda=nv12 \
  -c:v hevc_nvenc -preset p4 output.mkv
```

This keeps frames on the GPU throughout decode -> filter -> encode. The `tensorrt` filter loads ONNX or TensorRT engine files directly. **We already have NAFNet exported to ONNX**, so this could work with our existing model.

**Option C: Custom PyAV + NVENC subprocess (current approach, improved)**

Keep the current architecture but switch from x265 to hevc_nvenc:

```
PyAV decode (CPU) -> numpy -> CUDA tensor -> NAFNet -> numpy -> pipe -> hevc_nvenc (GPU)
```

Still has CPU<->GPU copies but the encoding no longer bottlenecks. With hevc_nvenc P4, expected throughput: **~40-60 fps** (limited by CPU<->GPU transfer + inference, not encoding).

### 1.5 Recommended Encoding Strategy

**For batch processing the library (immediate win):**
1. Switch encoder from `libx265` to `hevc_nvenc` preset P4 in `denoise_nafnet.py`
2. Expected speedup: 5.2 fps -> 40-60 fps (10x improvement)
3. File sizes will be ~20-30% larger than x265 at equivalent visual quality
4. Use `-cq 20` for quality comparable to x265 CRF 18

**For maximum throughput (future optimization):**
1. Use TorchAudio StreamWriter with `hw_accel="cuda:0"` for zero-copy encode
2. Or build NVIDIA GMAT's FFmpeg fork with TensorRT filter for full GPU pipeline
3. Target: approach the raw 78 fps inference speed

---

## 2. Real-Time Playback

### 2.1 VapourSynth + mpv (Most Promising Path)

**How it works:** mpv has built-in VapourSynth support. VapourSynth scripts can call neural network inference via the `vs-mlrt` plugin. mpv decodes video, passes frames through VapourSynth, and displays the result.

**vs-mlrt** (github.com/AmusementClub/vs-mlrt):
- Mature VapourSynth plugin for ML inference with multiple backends
- Supports **TensorRT**, ONNX Runtime, NCNN (Vulkan), and OpenVINO
- Already has built-in support for SCUNet and many other models
- Generic API for custom ONNX models (v12.2+):
  ```python
  import vsmlrt
  output = vsmlrt.inference(clip, "nafnet.onnx", backend=vsmlrt.Backend.TRT(fp16=True))
  ```
- TensorRT backend can be 2x faster than NCNN/Vulkan
- Pre-built Windows releases available

**mpv configuration for VapourSynth:**
```
# mpv.conf
hwdec=auto-copy    # Required: copy-back HW decoder for VapourSynth compatibility
vf=vapoursynth=~~/enhance.vpy
```

```python
# enhance.vpy (VapourSynth script)
import vapoursynth as vs
import vsmlrt
core = vs.core
clip = video_in  # mpv provides this
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
clip = vsmlrt.inference(clip, "path/to/nafnet.onnx", backend=vsmlrt.Backend.TRT(fp16=True))
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()
```

**Performance expectation:** With TensorRT backend and our 14M param NAFNet at 78 fps raw inference, real-time 1080p24 playback should be achievable with significant headroom. Even 1080p60 might work.

**Existing reference project: mpv-upscale-2x_animejanai** (github.com/the-database/mpv-upscale-2x_animejanai):
- Real-time 2x upscaling in mpv using TensorRT + VapourSynth
- Proves the concept works for real-time playback on consumer GPUs
- Uses compact Real-ESRGAN models, similar parameter count to our NAFNet

### 2.2 VideoJaNai (Standalone GUI)

**VideoJaNai** (github.com/the-database/VideoJaNai):
- Windows GUI for video upscaling with ONNX models via TensorRT + VapourSynth
- Supports batch processing and real-time playback
- DirectML backend available for non-NVIDIA GPUs
- Could potentially load our NAFNet ONNX model directly
- Good reference implementation for the VapourSynth + TensorRT pipeline

### 2.3 VLC Custom Filters

**Verdict: Not practical for neural network filters.**

- VLC has a C plugin API for video filters (`video filter2`), but filters must be written in C99
- No GPU compute integration -- filters operate on CPU pixel buffers
- VLC does not have native VapourSynth support
- Building a VLC plugin that calls TensorRT/ONNX Runtime in C would be a large undertaking
- VLC 4.0 (beta) may improve GPU filter support, but nothing concrete yet

**Workaround:** Pipe-based approach is theoretically possible (VLC decodes -> pipe -> our model -> pipe -> display) but adds latency and complexity. mpv + VapourSynth is far superior.

### 2.4 mpv GLSL Shaders (Not Applicable)

mpv supports custom GLSL shaders (e.g., Anime4K) for real-time enhancement. However:
- GLSL shaders cannot implement CNNs with learned weights efficiently
- Anime4K uses hand-crafted mathematical filters, not trained neural networks
- Our NAFNet has 14M learned parameters -- impossible to express as GLSL
- The VapourSynth path (Section 2.1) is the correct approach for CNN models in mpv

---

## 3. Plex Integration

### 3.1 Plex Transcoding Architecture

Plex uses a custom fork of FFmpeg (plex-ffmpeg) for all transcoding. It supports hardware acceleration via NVENC/NVDEC, Intel QSV, and AMD VCE. However:

- Plex does **not** expose a plugin API for custom video filters
- The custom FFmpeg fork is closed-source and cannot be easily modified
- There is no way to inject a neural network filter into Plex's transcoding pipeline
- Community requests for RIFE/TensorRT integration exist but are not supported

### 3.2 Pre-Processing (Best Approach for Plex)

**Plex Optimized Versions:**

The most practical approach is to pre-process videos and let Plex serve the enhanced versions:

1. Run NAFNet on the original files using the fast NVENC pipeline (Section 1)
2. Replace originals or add as "Optimized Versions" in Plex
3. Plex serves enhanced video with no transcoding needed (direct play)

**Workflow:**
```
Original MKV -> NAFNet + hevc_nvenc -> Enhanced MKV -> Plex library
```

At 40-60 fps with hevc_nvenc, a 24-minute episode (~34,500 frames at 23.976 fps) takes ~10-15 minutes. An overnight batch could process 30-50 episodes.

### 3.3 On-Demand Enhancement (Experimental)

A more advanced approach: run a local proxy server that enhances video on-the-fly:

1. Plex requests video from its library
2. A proxy intercepts the request, decodes, enhances, re-encodes in real-time
3. Streams the enhanced video to the Plex client

This would require significant engineering (HTTP range request handling, seeking, buffering) and is not recommended unless pre-processing is impractical.

---

## 4. Other Approaches

### 4.1 ONNX Runtime with TensorRT Execution Provider

For a serving/always-on scenario, ONNX Runtime with TensorRT EP may be faster than torch.compile:

- Eliminates PyTorch overhead entirely
- TensorRT optimizes the graph at load time (slow startup, fast steady-state)
- We already have NAFNet exported to ONNX (`checkpoints/nafnet_distill/nafnet_w64_1088x1920.onnx`)
- On Windows: `pip install onnxruntime-gpu` includes TensorRT EP
- Expected improvement over torch.compile: 10-30% for pure CNN models
- vs-mlrt uses this exact approach for VapourSynth integration

### 4.2 DirectML / DXVA Integration

**DirectML** (Microsoft's ML acceleration layer for DirectX 12):
- ONNX Runtime supports DirectML execution provider on Windows
- Works with any DirectX 12 GPU (NVIDIA, AMD, Intel)
- Lower performance than TensorRT on NVIDIA GPUs, but more portable
- Could enable enhancement on non-NVIDIA systems
- Not useful for our RTX 3060 setup (TensorRT is faster), but good for distribution

**DXVA** (DirectX Video Acceleration):
- Hardware decode/encode API, not relevant for ML inference
- Already used by media players for HW decode

### 4.3 FFmpeg DNN Filters

FFmpeg has two paths for neural network inference:

**Built-in `dnn_processing` filter:**
- Supports TensorFlow, OpenVINO, and Libtorch backends
- Limited: designed for simple models (denoise, SR)
- No TensorRT backend in mainline FFmpeg
- OpenVINO backend works but is CPU/Intel-GPU only

**NVIDIA GMAT TensorRT filter** (github.com/NVIDIA/GMAT):
- Custom FFmpeg fork with `tensorrt` video filter
- Full GPU pipeline: `hwaccel cuda decode -> tensorrt filter -> nvenc encode`
- Accepts ONNX models or pre-built TensorRT engines
- Requires building FFmpeg from source with TensorRT support
- **This is the most performant option for batch encoding** -- zero CPU copies
- Command: `ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i in.mkv -vf format_cuda=rgbpf32le,tensorrt=nafnet.onnx,format_cuda=nv12 -c:v hevc_nvenc out.mkv`

### 4.4 GStreamer + DeepStream

NVIDIA DeepStream SDK uses GStreamer with `nvinfer` element for TensorRT inference:
- Designed for video analytics (object detection, tracking)
- Supports video enhancement use cases
- Full GPU pipeline with NVDEC/NVENC integration
- Overkill for single-model enhancement -- more suited for multi-model analytics
- Linux-focused; Windows support is limited
- Not recommended for this use case

### 4.5 VSGAN-tensorrt-docker

**VSGAN-tensorrt-docker** (github.com/styler00dollar/VSGAN-tensorrt-docker):
- Docker container with VapourSynth + TensorRT for video enhancement
- Supports many models: Real-ESRGAN, RIFE, DPIR, SCUNet, etc.
- Linux/Docker only (not native Windows)
- Good reference for VapourSynth + TensorRT integration patterns
- Could be adapted for cloud processing via Modal

---

## 5. Recommendations

### Immediate Win: Switch to NVENC Encoding (1 hour of work)

Change `denoise_nafnet.py` to use `hevc_nvenc` instead of `libx265`:
- Expected: 5.2 fps -> 40-60 fps (10x speedup)
- A 24-min episode goes from 1.8 hours to ~10-15 minutes
- Full library processing becomes practical

### Medium Term: Real-Time Playback via mpv + VapourSynth + vs-mlrt

1. Install VapourSynth (Windows, standalone or via Python)
2. Install vs-mlrt with TensorRT backend (pre-built Windows release)
3. Export NAFNet to ONNX at 1080p (already done)
4. Write a VapourSynth script that loads the ONNX model
5. Configure mpv to use the VapourSynth filter
6. Result: real-time 1080p enhancement during playback

### Long Term: Full GPU Pipeline

Build NVIDIA GMAT's FFmpeg fork with TensorRT filter for maximum batch processing throughput. This eliminates all CPU<->GPU copies and could approach the raw 78 fps inference speed.

### Plex: Pre-Process Library

Use the fast NVENC pipeline to pre-process videos. Store enhanced versions in the Plex library for direct play. This is simpler and more reliable than trying to hook into Plex's transcoding.

---

## Summary Table

| Approach | Use Case | Expected FPS | Effort | Recommended? |
|----------|----------|-------------|--------|--------------|
| hevc_nvenc P4 in current pipeline | Batch encoding | 40-60 fps | Low | **Yes -- do this first** |
| TorchAudio NVENC zero-copy | Batch encoding | 60-78 fps | Medium | Yes, after NVENC switch |
| GMAT FFmpeg TensorRT filter | Batch encoding | ~78 fps | High (build from source) | Future optimization |
| mpv + VapourSynth + vs-mlrt | Real-time playback | 78 fps (>24 needed) | Medium | **Yes -- best playback option** |
| VideoJaNai | Real-time playback | Varies | Low (GUI app) | Worth trying |
| VLC plugin | Real-time playback | N/A | Very High | No |
| Plex transcoding hook | Streaming | N/A | Not possible | No |
| Plex pre-processing | Streaming | 40-60 fps | Low | **Yes -- pre-process library** |
| ONNX Runtime TensorRT EP | Serving | 80-100 fps | Medium | Yes, for dedicated server |
| GStreamer DeepStream | Batch/Serving | ~78 fps | High | No (overkill, Linux-only) |

---

## Sources

- [NVIDIA NVENC Application Note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html)
- [NVIDIA Video Codec SDK 10 Presets](https://developer.nvidia.com/blog/introducing-video-codec-sdk-10-presets/)
- [NVIDIA FFmpeg Transcoding Guide](https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/)
- [NVIDIA GMAT (FFmpeg GPU Demo)](https://github.com/NVIDIA/GMAT)
- [TorchAudio NVENC Tutorial](https://docs.pytorch.org/audio/2.7.0/tutorials/nvenc_tutorial.html)
- [vs-mlrt: ML Runtimes for VapourSynth](https://github.com/AmusementClub/vs-mlrt)
- [VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)
- [VideoJaNai](https://github.com/the-database/VideoJaNai)
- [mpv-upscale-2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai)
- [SVT-AV1 3.0 Release](https://www.phoronix.com/news/SVT-AV1-3.0-Released)
- [SVT-AV1 Presets Analysis](https://ottverse.com/analysis-of-svt-av1-presets-and-crf-values/)
- [FFmpeg dnn_processing Filter](https://ffmpeg.org/ffmpeg-filters.html)
- [NVIDIA DeepStream nvinfer Plugin](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html)
- [DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [VLC Hacker Guide: Video Filters](https://wiki.videolan.org/Hacker_Guide/Video_Filters/)
- [NVENC Wikipedia](https://en.wikipedia.org/wiki/Nvidia_NVENC)
- [Anime4K](https://github.com/bloc97/Anime4K)
- [vs-NNVISR](https://github.com/tongyuantongyu/vs-NNVISR)
