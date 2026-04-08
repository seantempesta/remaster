# Real-Time Video Enhancement: Pipeline & Playback Research

Research into using NAFNet (14M params, 78 fps raw inference on RTX 3060) for both
fast batch encoding and real-time playback enhancement.

**Date:** 2026-04-02 (updated with measured results)

---

## TL;DR — Python Can't Do This

We built and benchmarked three Python pipeline architectures. All hit the same wall:
**Python's GIL serializes CUDA operations across threads**, capping throughput at
5-7 fps despite the model running at 78 fps.

**Recommended path forward:** Use the ONNX model we exported with a C++ pipeline —
either **mpv + VapourSynth + vs-mlrt** (ready-made, for playback and batch) or a
**custom C++ app** with TensorRT + NVIDIA Video Codec SDK (maximum control).

---

## Measured Results

| Pipeline | FPS | Bottleneck | Notes |
|----------|-----|-----------|-------|
| Raw model inference | **78** | — | NAFNet w32_mid4 fp16 torch.compile |
| `denoise_fast.py` (ffmpeg pipes) | **3.5** | CPU↔GPU copies | Original pipe-based pipeline |
| `denoise_gpu.py` (PyNvVideoCodec) | **5.4** | Default stream serialization | All ops on stream 0 |
| `denoise_gpu_v2.py` (CUDA streams) | **6.8** | GIL contention | Stream-isolated but GIL-bound |
| `denoise_gpu_v3.py` (ffmpeg encode) | **5.0** | GPU→CPU sync | Encode outside GIL but sync blocks |
| `denoise_gpu_v2.py` (eager, no compile) | **1.2** | No CUDA graphs | 825ms/frame inference |
| Target (batch) | **40-60** | — | ~20 min for 42-min episode |
| Target (playback) | **24** | — | Real-time 1080p23.976 |

---

## 1. What We Tried and What We Learned

### 1.1 The Hardware (Independent, Can Overlap)

```
NVDEC  --- dedicated decoder ASIC --- 500+ fps at 1080p HEVC
CUDA   --- shader cores ------------- NAFNet at 78 fps (13ms/frame)
NVENC  --- dedicated encoder ASIC --- 150+ fps at P4 1080p HEVC
```

Task Manager confirmed: 3D at 97%, Video Encode at 2%, Video Decode at 1%.
The hardware CAN run concurrently — Python prevents it.

### 1.2 v1: Default Stream Serialization (5.4 fps)

`denoise_gpu.py` — 3 Python threads, PyNvVideoCodec for decode/encode, but everything
on the default CUDA stream. NVDEC's `cuStreamSynchronize` and NVENC's
`nvEncLockBitstream` drain ALL pending GPU work each frame.

### 1.3 v2: CUDA Stream Isolation (6.8 fps)

`denoise_gpu_v2.py` — Key discovery: **PyNvVideoCodec's SimpleDecoder and
CreateEncoder accept `cuda_stream` parameters** (undocumented in examples):

```python
decoder = nvc.SimpleDecoder(path, gpu_id=0,
    cuda_stream=decode_stream.cuda_stream)  # isolate NVDEC syncs

encoder = nvc.CreateEncoder(w, h, "ARGB", False,
    cudastream=encode_stream.cuda_stream)   # isolate NVENC syncs
```

Source confirmed in:
- `PyNvVideoCodec/decoders/SimpleDecoder.py:54` — `cuda_stream` param
- `PyNvVideoCodec/__init__.py:260` — `cudastream` kwarg for encoder
- `video-sdk-samples NvDecoder.cpp:504` — `cuStreamSynchronize(m_cuvidStream)`
- `VPF NvEncoder.cpp:476` — `nvEncLockBitstream` blocks on encoder's stream

**This fixed decode/encode isolation.** Per-frame timing:

```
decode:  0.4ms  (was 78ms — stream isolation works)
infer:   2.2ms  (was 165ms — async kernel launch)
encode: 183ms   (THE NEW BOTTLENECK)
```

### 1.4 The CUDA Graph Stream Bug

torch.compile `mode="reduce-overhead"` captures CUDA graphs that replay on the
**default stream**, regardless of `with torch.cuda.stream(infer_stream)` context.

This meant `infer_done[slot].record(infer_stream)` fired immediately (no work pending
on infer_stream), and the encoder waited for inference via its own internal sync instead.

**Fix:** Record the event on `torch.cuda.default_stream()`:

```python
infer_done[slot].record(torch.cuda.default_stream())  # where the graph actually runs
```

This brought encode from 183ms to 24ms/frame. But inference rose to 147ms because...

### 1.5 The GIL Wall (The Real Problem)

With stream isolation + correct event recording:

```
decode:  1.3ms   (fast, NVDEC async)
infer: 147ms     (13ms GPU + 134ms waiting for GIL)
encode: 24ms     (NVENC fast, but holds GIL during C extension call)
```

**The GIL serializes everything.** The 3 Python threads take turns:
1. Encode thread calls `encoder.Encode()` — C extension holds GIL for ~24ms
2. Decode thread calls `next(decoder)` — C extension holds GIL for ~2ms
3. Infer thread runs `model(input)` — launches CUDA graph, returns fast
4. Repeat... but each thread waits for the others' GIL releases

Total per-frame: ~24ms (encode GIL) + ~2ms (decode GIL) + ~13ms (GPU inference) +
Python overhead = ~180ms → **5-7 fps**

### 1.6 v3: ffmpeg Pipe Encode (5.0 fps)

`denoise_gpu_v3.py` — Moved encode to a separate ffmpeg process (no GIL).
But the main thread now does inference + `tensor.cpu()` which synchronizes the
default stream for 191ms/frame. Moving the GPU→CPU copy outside the infer thread
just shifted the sync to a different place. Still GIL-bound.

### 1.7 Why More Python Won't Help

Every approach hits the same wall: Python's GIL means only one thread executes Python
code at a time. CUDA operations release the GIL while running on GPU, but:

- `encoder.Encode()` is a C extension that holds GIL during `nvEncLockBitstream`
- `decoder.__next__()` holds GIL during `cuStreamSynchronize`
- `tensor.cpu()` holds GIL during DMA transfer
- Queue operations, tensor creation, and all Python code hold the GIL

No amount of threading, stream management, or buffering fixes this. The GIL is a
language-level constraint.

---

## 2. Recommended Path Forward: C++ Pipeline

### 2.1 Why C++

The NVIDIA Video Codec SDK, TensorRT, and CUDA are all native C++ APIs. A C++ pipeline
eliminates the GIL entirely. With proper CUDA streams, the three hardware units
(NVDEC, CUDA, NVENC) run concurrently as designed.

Expected throughput: **60-78 fps** (inference-bound).

### 2.2 Option A: mpv + VapourSynth + vs-mlrt (Recommended First Step)

This is a ready-made C++ pipeline. No code to write.

**vs-mlrt** (github.com/AmusementClub/vs-mlrt):
- VapourSynth plugin for TensorRT/ONNX inference, written in C++
- Handles CUDA stream management, frame batching, engine caching
- Pre-built Windows releases (includes TensorRT runtime)
- Already proven for real-time 1080p enhancement on RTX 3060

**For playback:** mpv + VapourSynth + vs-mlrt with our ONNX model.
**For batch encoding:** `vspipe` + ffmpeg with the same VapourSynth script.

Setup:
```python
# playback/enhance.vpy (already written)
clip = vsmlrt.inference(clip, "nafnet_w32mid4.onnx",
                        backend=vsmlrt.Backend.TRT(fp16=True))
```

```
# mpv.conf
hwdec=auto-copy
vf=vapoursynth=enhance.vpy
```

We already have:
- ONNX model: `checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx` (57MB)
- VapourSynth script: `playback/enhance.vpy`
- Install guide: `playback/README.md`

Remaining: install VapourSynth + vs-mlrt + mpv, test playback.

### 2.3 Option B: Custom C++ Pipeline (Maximum Control)

For batch encoding with maximum throughput, build a dedicated C++ app:

```
Architecture:
  NVDEC (decode_stream) → TensorRT (infer_stream) → NVENC (encode_stream)
  
Components:
  - NVIDIA Video Codec SDK: NvDecoder + NvEncoder (C++ API)
  - TensorRT: load ONNX, build engine, execute_async_v3(stream)
  - CUDA streams + events for inter-stage sync
  - Ring buffers in GPU memory (triple buffered)
  
Reference code:
  - reference-code/video-sdk-samples/Samples/AppTranscode/AppTransPerf/
  - reference-code/VPF/src/TC/ (decode + encode C++ implementation)
```

Effort: ~1-2 weeks of C++ development. Reward: 60-78 fps, fully pipelined.

### 2.4 Option C: Rust Pipeline (Possible but Less Mature)

Rust can call all the same C APIs via FFI:
- `cudarc` crate for CUDA
- TensorRT C API via bindgen
- NVDEC/NVENC via Video Codec SDK C API

The safety guarantees are nice for the ring buffer / stream synchronization logic.
But: no existing examples of this exact pipeline in Rust, thinner ecosystem for GPU
video processing compared to C++. The FFI wrapping overhead is real development cost.

**Verdict:** C++ is more pragmatic for this specific problem. If you already preferred
Rust for other reasons, it's viable but expect more plumbing work.

### 2.5 Language Comparison for This Pipeline

| Language | GIL? | NVDEC/NVENC | TensorRT | Ecosystem | Effort |
|----------|------|-------------|----------|-----------|--------|
| **C++** | No | Native SDK | Native API | vs-mlrt, VPF, samples | Low-Med |
| **Rust** | No | FFI | FFI | Thin | Medium |
| **Python** | Yes | PyNvVideoCodec | torch/onnxrt | Rich | Done (but 7 fps max) |
| **Go** | No | cgo/FFI | FFI | Very thin | High |

---

## 3. ONNX Model: The Bridge

The ONNX export is the key asset — it makes the model portable across runtimes:

- **File:** `checkpoints/nafnet_w32_mid4/nafnet_w32mid4_1088x1920.onnx` (57MB)
- **Opset:** 18 (PyTorch 2.11 dynamo export)
- **Input:** `[1, 3, 1088, 1920]` float32 (padded 1080p)
- **Output:** `[1, 3, 1088, 1920]` float32
- **Validated:** ONNX checker + ORT CPU inference sanity check
- **Export script:** `cloud/modal_export_onnx_w32.py` (runs on Modal)

Any TensorRT-based pipeline (vs-mlrt, custom C++, Rust) loads this ONNX file.
First run builds a GPU-specific TensorRT engine (~2-5 min), cached for reuse.

---

## 4. VLC Plugin: Not Viable

VLC's video filter API (`video filter2`, C99) only handles CPU pixel buffers. Even
with hardware decoding, frames are copied to CPU before filters see them. No existing
CUDA/TensorRT VLC plugins exist. RTX Video Super Resolution works at the driver level
with fixed NVIDIA models — no custom model support.

**Verdict:** VLC's architecture predates GPU compute. mpv + VapourSynth is correct.

---

## 5. What the Python Pipelines ARE Good For

Despite the GIL ceiling, the Python pipelines are valuable for:

- **Prototyping:** Fast iteration on model architectures, preprocessing, quality eval
- **Quality testing:** Process short clips to compare model outputs
- **Cloud inference:** Modal GPU instances (H100 at 27.9 fps) — GIL matters less when
  GPU is much faster and Python overhead is proportionally smaller
- **Training:** The GIL doesn't affect single-GPU training

The 5-7 fps Python pipelines process a 42-min episode in ~2.5 hours. Usable but slow.
For real production throughput (20 min per episode), use the C++ path.

---

## 6. Implementation Plan

### Done
- [x] ONNX export: `nafnet_w32mid4_1088x1920.onnx` (57MB, validated)
- [x] VapourSynth script: `playback/enhance.vpy`
- [x] Confirmed Python/GIL is the ceiling (not CUDA, not NVENC, not streams)
- [x] Confirmed PyNvVideoCodec stream isolation works (`cuda_stream=` params)
- [x] Confirmed CUDA graph replay stream behavior (default stream, not context)

### Next: Install and Test vs-mlrt Playback
1. Download VapourSynth portable + vs-mlrt Windows release + mpv
2. Copy ONNX model to vs-mlrt search path
3. Test playback: `mpv --vf=vapoursynth=enhance.vpy episode.mkv`
4. First run builds TensorRT engine (~2-5 min)
5. Measure playback fps — expect real-time 1080p24 with headroom

### Next: vs-mlrt Batch Encoding
1. Use `vspipe` to pipe through VapourSynth + vs-mlrt + ffmpeg:
   `vspipe enhance.vpy -c y4m - | ffmpeg -i - -c:v hevc_nvenc -preset p4 output.mkv`
2. This is the fastest batch path without custom C++ — C++ TensorRT inference
   with ffmpeg encoding
3. Measure throughput — expect 20-40 fps

### Future: Custom C++ Pipeline (If vs-mlrt Is Insufficient)
1. Build from Video Codec SDK samples + TensorRT
2. Full NVDEC → TensorRT → NVENC in C++, three CUDA streams
3. Target: 60-78 fps
4. Only worth doing if vs-mlrt batch encoding isn't fast enough

---

## Sources

- [vs-mlrt (VapourSynth ML Runtime)](https://github.com/AmusementClub/vs-mlrt)
- [VideoJaNai](https://github.com/the-database/VideoJaNai)
- [mpv-upscale-2x_animejanai](https://github.com/the-database/mpv-upscale-2x_animejanai)
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk)
- [NVIDIA Video Processing Framework](https://github.com/NVIDIA/VideoProcessingFramework)
- [NVIDIA video-sdk-samples](https://github.com/NVIDIA/video-sdk-samples)
- [NVIDIA GMAT (FFmpeg GPU Demo)](https://github.com/NVIDIA/GMAT)
- [TorchCodec](https://github.com/meta-pytorch/torchcodec) — monitor for future
- [VLC Hacker Guide](https://wiki.videolan.org/Hacker_Guide/Video_Filters/)
- [VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)
- [NVIDIA NVENC Application Note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html)
