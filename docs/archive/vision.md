# remaster: Vision

## What It Is

remaster removes compression artifacts from video at native resolution. Not upscaling, not colorizing, not adding fake detail -- just undoing the damage that H.264 and H.265 encoders inflict on every frame. Blocking, banding, ringing, mosquito noise, smeared textures. The stuff that makes a perfectly good BluRay rip look like it was streamed over hotel wifi.

The model that does this runs at 78 fps on an RTX 3060. The bottleneck has never been inference -- it's everything around it. remaster is the system that eliminates that bottleneck.

## Two Modes

### Real-time playback

Open a video in mpv. It plays back enhanced, live, with no pre-processing. VapourSynth sits between the decoder and the renderer, running every frame through a TensorRT engine before it hits the screen. You see the clean version. That's it.

```
mpv --vf=vapoursynth=enhance.vpy input.mkv
```

This is for watching. You don't commit to processing your whole library before you can enjoy it. Pick a file, hit play, see the difference immediately.

### Batch processing

Upgrade your entire library overnight. vspipe feeds frames through the same TensorRT engine, ffmpeg encodes the output to HEVC, and you wake up to a shelf of remastered files that look better than the originals ever did on disc.

```
vspipe --y4m enhance.vpy - | ffmpeg -i - -c:v libx265 -crf 18 output.mkv
```

This is for permanence. Process once, store forever. The remastered files replace the originals in your library and every future playback is the clean version -- no GPU needed at watch time.

## Architecture

The entire hot path is C++. That's the whole point.

```
ffmpeg/NVDEC (decode) --> VapourSynth (frame routing) --> vs-mlrt/TensorRT (inference) --> NVENC/ffmpeg (encode)
```

The NAFNet model runs at 78 fps raw. But every Python pipeline we built caps out at 5-7 fps because Python's GIL serializes the decode/infer/encode threads regardless of CUDA stream separation. We tried everything -- CUDA stream isolation, ring buffers, pipe-based architectures, zero-copy GPU paths. The GIL is the ceiling.

VapourSynth eliminates Python from the hot path entirely. It's a C++ frame server. vs-mlrt runs TensorRT inference in C++. The decode and encode happen in C++ (or on dedicated NVDEC/NVENC hardware). No GIL, no Python overhead, no 7 fps cap.

Three independent hardware units on the GPU work simultaneously:

- **NVDEC** -- dedicated decode ASIC, handles any input codec
- **CUDA cores** -- run the TensorRT inference engine
- **NVENC** -- dedicated encode ASIC, writes HEVC output

These are physically separate silicon. They don't compete for the same resources. A properly pipelined system saturates all three.

## Input Flexibility

remaster accepts anything ffmpeg can decode. H.264, HEVC, AV1, VP9, MPEG-2, whatever legacy codec your old rips use. The input format doesn't matter because decoding happens before the model ever sees a frame. If ffmpeg can read it, remaster can enhance it.

## Output

Always HEVC. It's the right trade-off for remastered content: excellent compression efficiency, universal hardware decode support, mature encoder ecosystem. The remastered frames have less noise and fewer high-frequency artifacts than the originals, which means HEVC compresses them more efficiently -- you often get smaller files at higher visual quality.

AV1 encoding is too slow for batch processing entire libraries today. When hardware AV1 encoders mature, that changes. HEVC is the pragmatic choice now.

## Hardware Saturation

The system auto-detects GPU capabilities and builds TensorRT engines tuned to the specific card. An RTX 3060 gets a different engine than an RTX 4090 -- different tensor core generations, different memory bandwidth, different optimal batch sizes.

The build step happens once per GPU per model:

1. Load ONNX model (exported from PyTorch training)
2. TensorRT optimizes for the detected GPU -- layer fusion, precision calibration, memory planning
3. Engine file is cached to disk
4. All subsequent runs load the cached engine in milliseconds

This means remaster works on any NVIDIA GPU with tensor cores, not just the one it was developed on. The engine adapts to the hardware.

## Model Pipeline

The model doesn't appear from nowhere. There's a deliberate pipeline from slow-and-good to fast-and-good-enough:

```
SCUNet (teacher, 1 fps)
  |
  v
Generate targets: SCUNet_GAN(frame) + 0.15 * high_pass(frame)
  |
  v
Distill to NAFNet (student, 78 fps)
  |
  v
Export ONNX (portable, framework-independent)
  |
  v
Build TensorRT engine (GPU-specific, maximum speed)
  |
  v
Run in VapourSynth via vs-mlrt
```

**Teacher (SCUNet):** Transformer-based denoiser. Produces excellent results at ~1 fps. Too slow for production but perfect for generating training data. The GAN variant produces sharper output than the PSNR-only variant.

**Detail transfer:** The teacher's GAN output is clean but sometimes over-smooths fine texture. Blending 15% of the original's high-frequency content back into the targets recovers hair detail, fabric texture, and film grain without reintroducing compression artifacts. Zero hallucination -- every pixel of recovered detail comes from the source.

**Student (NAFNet):** Pure CNN, no attention layers. torch.compile friendly, TensorRT friendly, stupidly fast. The w32-mid4 variant (width 32, 4 middle blocks) is 14.3M parameters, 55 MB on disk, 2.3 GB VRAM, 78 fps at 1080p.

**ONNX export:** Decouples the model from PyTorch. The ONNX file is the portable artifact that TensorRT consumes. Export once, build engines on any GPU.

**TensorRT engine:** The final form. FP16 inference with CUDA graph replay, layer fusion, and memory optimization specific to the target GPU.

## Future

**WebUI for status and control.** A local web interface for monitoring batch jobs, comparing before/after on specific frames, and configuring processing parameters. Not a cloud service -- runs on the same machine as the pipeline.

**AI-agent-friendly operation.** CLAUDE.md and AGENTS.md describe the system well enough that an AI coding agent can navigate the codebase, run experiments, and propose improvements without hand-holding. The codebase is the documentation.

**Configurable models.** Different content needs different treatment. Animation has different artifacts than live action. Dark scenes need different handling than bright ones. The system should support swapping models per-content-type without rebuilding the pipeline.

## Design Principles

**Every component testable independently with synthetic data.** The TensorRT engine should process a solid-color frame correctly before you hook it up to a video decoder. The encoder should produce valid HEVC from a test pattern before you feed it real inference output. Don't debug the pipeline -- debug each stage, then connect them.

**Saturate hardware before connecting components.** If TensorRT inference only hits 30 fps in isolation, pipelining it with decode and encode won't magically make it faster. Get each stage to its theoretical maximum throughput alone, then worry about connecting them without introducing bottlenecks.

**Portable -- works on any NVIDIA GPU, not just one specific card.** TensorRT engines are GPU-specific by design, but the build process is automatic. Clone the repo, run the setup, get an engine optimized for whatever GPU you have. No manual tuning, no "works on my machine."

**Maintain forks of critical dependencies.** vs-mlrt is the bridge between VapourSynth and TensorRT. If upstream makes a breaking change or goes unmaintained, we need to keep going. Fork it, pin it, insulate from API drift. Same for any dependency where a broken update would halt the project.
