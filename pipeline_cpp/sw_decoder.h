// sw_decoder.h -- FFmpeg software decode fallback for formats NVDEC cannot handle
//
// Provides the same output interface as NvDecoder: GPU device pointers in
// NV12 (8-bit) or P010 (10-bit) semi-planar format, so downstream code
// (color kernels, TRT inference, NVENC) requires zero changes.
//
// Use case: H264 High 10-bit on RTX 3060 (NVDEC hardware limitation),
// or any codec FFmpeg supports but NVDEC does not.
#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

class SwDecoder {
public:
    // Construct from the demuxer's codec parameters.
    // Allocates the FFmpeg decoder context, pinned host buffer, and GPU device buffers.
    SwDecoder(const AVCodecParameters* codecpar, cudaStream_t stream = 0);
    ~SwDecoder();

    // Non-copyable
    SwDecoder(const SwDecoder&) = delete;
    SwDecoder& operator=(const SwDecoder&) = delete;

    // Decode a compressed packet. Returns pointers to decoded frames via pppFrame
    // and frame count via pnFrameReturned, matching NvDecoder's interface.
    //
    // Each returned frame is a GPU device pointer in NV12 (8-bit) or P010 (10-bit)
    // semi-planar layout, identical to what NVDEC produces.
    //
    // Pass pData=nullptr, nSize=0 to flush remaining buffered frames.
    //
    // The returned pointers are valid until the next Decode() call.
    bool Decode(const uint8_t* pData, int nSize,
                uint8_t*** pppFrame, int* pnFrameReturned);

    // Pitch (bytes per row) of the device frame buffer.
    // For NV12: width bytes. For P010: width * 2 bytes.
    int GetDeviceFramePitch() const { return devicePitch_; }

    // Whether the decoded content is 10-bit (P010 output) or 8-bit (NV12 output).
    bool Is10Bit() const { return is10bit_; }

    // Decoded frame dimensions.
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }

    bool IsValid() const { return valid_; }

private:
    // Convert a decoded AVFrame (planar YUV) to semi-planar NV12 or P010 in a
    // pinned host buffer, then upload to a GPU device buffer from the ring.
    // Returns the device pointer on success, nullptr on failure.
    uint8_t* convertAndUpload(AVFrame* frame);

    // Interleave separate U and V planes into a single UV plane (NV12/P010 format).
    void interleavePlanes8(const uint8_t* uPlane, int uLinesize,
                           const uint8_t* vPlane, int vLinesize,
                           uint8_t* uvDst, int uvDstPitch,
                           int chromaWidth, int chromaHeight);

    void interleavePlanes16(const uint16_t* uPlane, int uLinesize,
                            const uint16_t* vPlane, int vLinesize,
                            uint16_t* uvDst, int uvDstPitch,
                            int chromaWidth, int chromaHeight);

    AVCodecContext* codecCtx_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* pkt_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    bool is10bit_ = false;
    bool valid_ = false;

    // Ring of device frame buffers (GPU memory, NV12 or P010 layout).
    // Multiple buffers are needed because Decode() can return multiple frames
    // per packet (B-frame reordering), and the caller processes them via async
    // GPU kernels that may not have consumed earlier frames yet.
    static constexpr int kNumDeviceBuffers = 8;  // match NvDecoder's typical surface count
    uint8_t* deviceBuffers_[kNumDeviceBuffers] = {};
    int deviceBufIdx_ = 0;  // next slot to write into

    int devicePitch_ = 0;        // bytes per row
    size_t perBufferSize_ = 0;   // size of one NV12/P010 frame buffer

    // Pinned host buffer for CPU-side conversion before upload
    uint8_t* hostBuffer_ = nullptr;
    size_t hostBufferSize_ = 0;

    // CUDA stream for async memcpy
    cudaStream_t stream_ = 0;

    // Decoded frame pointers returned to caller
    std::vector<uint8_t*> framePointers_;
};
