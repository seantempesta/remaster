# PRD: C++ Pipeline v2 — End-to-End MKV Muxing + Performance Optimization

## Problem Statement

The C++ pipeline (`pipeline_cpp/remaster_pipeline.exe`) runs at 40 fps — the fastest path we have. But it's not production-ready:

1. **No container output** — writes raw HEVC bitstream, no timestamps, no seeking
2. **No audio** — requires a separate ffmpeg mux step to add audio
3. **No frame rate metadata** — output plays at wrong speed without manual -r flag
4. **Sequential processing** — decode/infer/encode run serially per frame (40 fps vs 52 fps theoretical)

Goal: single-command end-to-end `input.mkv -> output.mkv` with audio passthrough at 45+ fps.

## Current Architecture

```
SimpleDemuxer (libavformat)
  -> reads video packets from MKV
  -> applies HEVC Annex B bitstream filter
  
NvDecoder (NVDEC hardware)
  -> decodes HEVC to NV12/P010 on GPU

CUDA kernel: NV12 -> RGB FP16
TRT inference: RGB FP16 -> RGB FP16  
CUDA kernel: RGB FP16 -> NV12

NvEncoderCuda (NVENC hardware)
  -> encodes NV12 to HEVC packets

std::ofstream
  -> writes raw packets to .hevc file (NO CONTAINER)
```

All stages run on a single CUDA stream, with `cudaStreamSynchronize` between frames.

## Changes Required

### 1. MKV Muxer with Audio Passthrough

Replace `std::ofstream` with libavformat muxer:

```cpp
// In main.cpp setup (after opening input):
AVFormatContext* ofmtCtx = nullptr;
avformat_alloc_output_context2(&ofmtCtx, nullptr, "matroska", outputPath);

// Add video stream - copy params from encoder
AVStream* outVideoStream = avformat_new_stream(ofmtCtx, nullptr);
// Set codec_id, width, height, time_base, color metadata (BT.709)

// Add audio stream(s) - copy from input
for each audio stream in input:
    AVStream* outAudioStream = avformat_new_stream(ofmtCtx, nullptr);
    avcodec_parameters_copy(outAudioStream->codecpar, inputAudioStream->codecpar);

// Open output file
avio_open(&ofmtCtx->pb, outputPath, AVIO_FLAG_WRITE);
avformat_write_header(ofmtCtx, nullptr);

// In frame loop: write video packets with timestamps
AVPacket pkt;
pkt.data = encodedData;
pkt.size = encodedSize;
pkt.pts = frameCount * timeBaseDen / fps;  // proper timestamps
pkt.dts = pkt.pts;
pkt.stream_index = videoStreamIndex;
av_interleaved_write_frame(ofmtCtx, &pkt);

// In demux loop: forward audio packets to output
if (packet is audio) {
    av_packet_rescale_ts(&pkt, inputTimeBase, outputTimeBase);
    pkt.stream_index = outputAudioIndex;
    av_interleaved_write_frame(ofmtCtx, &pkt);
}

// At end:
av_write_trailer(ofmtCtx);
```

**Key details:**
- `av_interleaved_write_frame` handles interleaving video + audio packets in correct order
- Audio packets are copied verbatim (no re-encoding)
- Video timestamps derived from frame count * frame duration
- BT.709 color metadata set on output video stream
- Subtitle streams can also be passed through (same as audio)

### 2. Audio Passthrough in SimpleDemuxer

Currently `simple_demuxer.h` skips non-video packets (line 135: `if (pkt_->stream_index == videoStreamIdx_) break`). 

Need to either:
- **Option A:** Return audio packets separately via a second method `DemuxAudio()`
- **Option B:** Return all packets and let main.cpp route them (video to decoder, audio to muxer)

Option B is simpler:

```cpp
// New method: read next packet of any type
bool DemuxAny(uint8_t** ppData, int* pnBytes, int* pStreamIndex) {
    av_packet_unref(pkt_);
    int ret = av_read_frame(fmtCtx_, pkt_);
    if (ret < 0) return false;
    
    *pStreamIndex = pkt_->stream_index;
    
    if (pkt_->stream_index == videoStreamIdx_ && bsfCtx_) {
        // Apply bitstream filter for video
        av_bsf_send_packet(bsfCtx_, pkt_);
        av_bsf_receive_packet(bsfCtx_, pktFiltered_);
        *ppData = pktFiltered_->data;
        *pnBytes = pktFiltered_->size;
    } else {
        *ppData = pkt_->data;
        *pnBytes = pkt_->size;
    }
    return true;
}

// Expose stream info for audio passthrough
int GetNumStreams() const { return fmtCtx_->nb_streams; }
AVStream* GetStream(int idx) const { return fmtCtx_->streams[idx]; }
int GetVideoStreamIndex() const { return videoStreamIdx_; }
AVRational GetTimeBase() const { return fmtCtx_->streams[videoStreamIdx_]->time_base; }
double GetFrameRate() const { /* from stream avg_frame_rate */ }
```

### 3. Main Loop Restructure

Current loop:
```cpp
while (demuxer.Demux(&pVideo, &nVideoBytes)) {
    dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
    processDecodedFrames(ppFrame, nFrameReturned);
}
```

New loop:
```cpp
int streamIdx;
while (demuxer.DemuxAny(&pData, &nBytes, &streamIdx)) {
    if (streamIdx == demuxer.GetVideoStreamIndex()) {
        // Video: decode -> infer -> encode -> mux
        dec.Decode(pData, nBytes, &ppFrame, &nFrameReturned);
        for (int i = 0; i < nFrameReturned; i++) {
            processFrame(ppFrame[i]);  // color convert + TRT + color convert
            encodeAndMux(ppFrame[i]);  // NVENC + write to muxer
        }
    } else {
        // Audio/subtitle: passthrough to muxer
        muxPassthroughPacket(demuxer.GetCurrentPacket(), streamIdx);
    }
}
```

### 4. BT.709 Color Metadata

Set on the output video stream:
```cpp
outVideoStream->codecpar->color_range = AVCOL_RANGE_MPEG;
outVideoStream->codecpar->color_primaries = AVCOL_PRI_BT709;
outVideoStream->codecpar->color_trc = AVCOL_TRC_BT709;
outVideoStream->codecpar->color_space = AVCOL_SPC_BT709;
```

### 5. Performance Optimization (PENDING PROFILING DATA)

**Current:** Sequential per-frame processing on one CUDA stream.

**Target:** Overlapped decode/infer/encode using separate CUDA streams.

```
Stream 1 (decode):  [D1][D2][D3][D4][D5]...
Stream 2 (infer):      [I1][I2][I3][I4]...
Stream 3 (encode):         [E1][E2][E3]...
```

Frame N+1 decodes while frame N infers while frame N-1 encodes. All three use separate hardware (NVDEC/CUDA cores/NVENC) that can run concurrently.

**Implementation:**
- Triple buffer: 3 FrameBuffer objects (already allocated in current code)
- 3 CUDA streams (already created in current code as `streams[3]`)
- CUDA events for synchronization between stages:
  - After decode N completes -> signal infer N can start
  - After infer N completes -> signal encode N can start
- Remove the per-frame `cudaStreamSynchronize` (replace with event-based sync)

**Expected speedup:** From 40 fps to 48-52 fps (approaching TRT-bound ceiling).

**Profiling results (780 frames, 43.3 fps measured):**

| Stage | ms/frame | % of wall | Notes |
|-------|----------|-----------|-------|
| **TRT Inference** | **19.40** | **84.0%** | Matches trtexec (19ms = 52 fps) |
| File Write | 2.24 | 9.7% | Synchronous ofstream, exFAT drive |
| NVENC Encode | 0.64 | 2.8% | Hardware encoder, fast |
| NVDEC Decode | 0.37 | 1.6% | Hardware decoder, fast |
| CSC NV12->RGB | 0.10 | 0.4% | CUDA kernel, trivial |
| CSC RGB->NV12 | 0.12 | 0.5% | CUDA kernel, trivial |
| Demux (CPU) | 0.05 | 0.2% | FFmpeg packet reading |
| **Total wall** | **23.1** | | **43.3 fps** |

**Key findings:**
- TRT is 84% of time -- the ceiling. Color conversion is free (0.22ms combined).
- File I/O is 10% -- writing raw HEVC to exFAT. MKV muxer on SSD + async thread fixes this.
- Sequential serialization loses ~5 fps from CPU waiting between stages.

**Optimization plan (priority order):**
1. **MKV muxer writes to local SSD** (not exFAT) -- recovers ~4 fps
2. **Async writer thread** -- file I/O off critical path -- recovers ~4 fps
3. **Double-buffer pipeline** -- overlap encode N with infer N+1 -- recovers ~3 fps
4. **Expected result: 48-50 fps** (TRT-bound ceiling is 52 fps)

**Batch inference (experimental):**

TRT supports batch>1. Build engine with `--shapes=input:2x3x1080x1920` to process 2 frames per inference call. Benefits:
- Better GPU utilization (more data per kernel launch, better occupancy)
- Amortizes kernel launch overhead across 2 frames
- Could drop effective per-frame time from 19ms to ~12ms (6ms/frame)

Costs:
- ~600MB I/O buffers (vs 300MB for batch 1) -- fits in 6GB
- Requires 2 decoded frames ready before inference starts (adds 1 frame latency)
- NVENC still encodes one frame at a time

Worth testing with trtexec first:
```bash
trtexec --onnx=drunet_student.onnx --shapes=input:2x3x1080x1920 --fp16
```
If batch-2 throughput is significantly better than 2x batch-1, integrate into pipeline.

### 6. CLI Updates

```
remaster_pipeline [options]
  --input   / -i    Input video file (MKV, MP4, etc.)
  --output  / -o    Output MKV file (with audio passthrough)
  --engine  / -e    TensorRT engine file
  --gpu             GPU ordinal (default: 0)
  --cq              Constant quality (default: 20, lower=better)
  --preset          NVENC preset p1-p7 (default: p4)
  --10bit           Output 10-bit HEVC
  --no-audio        Skip audio passthrough
  --help   / -h     Show help
```

## Files to Modify

| File | Changes |
|------|---------|
| `pipeline_cpp/main.cpp` | Add muxer init/write/close, restructure main loop, add audio routing |
| `pipeline_cpp/simple_demuxer.h` | Add DemuxAny(), expose stream info, frame rate, time base |
| `pipeline_cpp/CMakeLists.txt` | May need to link `swresample` if audio needs resampling (unlikely for passthrough) |

## Build & Test

```bash
# Build
cmd /C pipeline_cpp\build.bat

# Test (should produce MKV with audio)
pipeline_cpp/build/remaster_pipeline.exe \
    -i data/archive/firefly_s01e08_30s.mkv \
    -e checkpoints/drunet_student/drunet_student_1080p_fp16.engine \
    -o data/archive/test_cpp_muxed.mkv --cq 20

# Verify
bin/ffmpeg.exe -i data/archive/test_cpp_muxed.mkv
# Should show: video stream (HEVC) + audio stream (AAC 5.1)
```

## Success Criteria

1. **End-to-end MKV output** with video + audio in a single command
2. **Audio plays correctly** (no sync issues, no corruption)
3. **Proper timestamps** (seeking works, correct duration reported)
4. **BT.709 color metadata** tagged in output
5. **Speed >= 40 fps** (muxing overhead should be negligible)
6. **With pipelining: >= 45 fps** (stretch goal)
