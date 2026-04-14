// sw_decoder.cpp -- FFmpeg software decode fallback implementation
//
// Decodes video with FFmpeg's libavcodec, converts planar YUV to semi-planar
// NV12/P010 on the CPU, then uploads to GPU via pinned memory + cudaMemcpyAsync.
// Produces the exact same GPU buffer format as NvDecoder.

#include "sw_decoder.h"
#include <cstdio>
#include <cstring>

SwDecoder::SwDecoder(const AVCodecParameters* codecpar, cudaStream_t stream)
    : stream_(stream)
{
    // Find the software decoder for this codec
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "SwDecoder: no FFmpeg decoder for codec ID %d\n", codecpar->codec_id);
        return;
    }

    codecCtx_ = avcodec_alloc_context3(codec);
    if (!codecCtx_) {
        fprintf(stderr, "SwDecoder: failed to allocate codec context\n");
        return;
    }

    int ret = avcodec_parameters_to_context(codecCtx_, codecpar);
    if (ret < 0) {
        fprintf(stderr, "SwDecoder: failed to copy codec parameters\n");
        return;
    }

    // Use all available CPU threads for software decode
    codecCtx_->thread_count = 0;  // auto-detect
    codecCtx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    ret = avcodec_open2(codecCtx_, codec, nullptr);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "SwDecoder: failed to open codec: %s\n", errbuf);
        return;
    }

    width_ = codecCtx_->width;
    height_ = codecCtx_->height;

    // Determine bit depth from pixel format
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codecCtx_->pix_fmt);
    int bitsPerComponent = desc ? desc->comp[0].depth : 8;
    is10bit_ = (bitsPerComponent > 8);

    fprintf(stderr, "SwDecoder: %s %dx%d %d-bit (pix_fmt=%s)\n",
            codec->name, width_, height_, bitsPerComponent,
            desc ? desc->name : "unknown");

    // Compute pitch and buffer sizes.
    // NV12: 1 byte/sample, P010: 2 bytes/sample.
    // Layout: luma (height rows) + chroma (height/2 rows), same pitch.
    if (is10bit_) {
        devicePitch_ = width_ * 2;  // 2 bytes per sample for P010 (uint16)
    } else {
        devicePitch_ = width_;      // 1 byte per sample for NV12 (uint8)
    }
    perBufferSize_ = (size_t)devicePitch_ * height_ * 3 / 2;

    // Allocate ring of device buffers
    for (int i = 0; i < kNumDeviceBuffers; i++) {
        cudaError_t err = cudaMalloc(&deviceBuffers_[i], perBufferSize_);
        if (err != cudaSuccess) {
            fprintf(stderr, "SwDecoder: cudaMalloc failed for buffer %d (%zu bytes): %s\n",
                    i, perBufferSize_, cudaGetErrorString(err));
            // Free any already-allocated buffers
            for (int j = 0; j < i; j++) {
                cudaFree(deviceBuffers_[j]);
                deviceBuffers_[j] = nullptr;
            }
            return;
        }
    }

    // Allocate pinned host buffer (one frame, reused for each upload)
    hostBufferSize_ = perBufferSize_;
    cudaError_t err = cudaMallocHost(&hostBuffer_, hostBufferSize_);
    if (err != cudaSuccess) {
        fprintf(stderr, "SwDecoder: cudaMallocHost failed (%zu bytes): %s\n",
                hostBufferSize_, cudaGetErrorString(err));
        for (int i = 0; i < kNumDeviceBuffers; i++) {
            cudaFree(deviceBuffers_[i]);
            deviceBuffers_[i] = nullptr;
        }
        return;
    }

    // Allocate reusable AVFrame and AVPacket
    frame_ = av_frame_alloc();
    pkt_ = av_packet_alloc();
    if (!frame_ || !pkt_) {
        fprintf(stderr, "SwDecoder: failed to allocate AVFrame/AVPacket\n");
        return;
    }

    valid_ = true;
}

SwDecoder::~SwDecoder() {
    if (frame_) av_frame_free(&frame_);
    if (pkt_) av_packet_free(&pkt_);
    if (codecCtx_) avcodec_free_context(&codecCtx_);
    for (int i = 0; i < kNumDeviceBuffers; i++) {
        if (deviceBuffers_[i]) cudaFree(deviceBuffers_[i]);
    }
    if (hostBuffer_) cudaFreeHost(hostBuffer_);
}

bool SwDecoder::Decode(const uint8_t* pData, int nSize,
                       uint8_t*** pppFrame, int* pnFrameReturned)
{
    framePointers_.clear();
    *pppFrame = nullptr;
    *pnFrameReturned = 0;

    if (!valid_) return false;

    // Send packet to decoder (nullptr for flush)
    if (pData && nSize > 0) {
        av_packet_unref(pkt_);
        // Wrap the data without copying -- avcodec_send_packet copies internally
        pkt_->data = const_cast<uint8_t*>(pData);
        pkt_->size = nSize;

        int ret = avcodec_send_packet(codecCtx_, pkt_);
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            // EAGAIN means the internal buffer is full, we'll drain it below
            if (ret != AVERROR_EOF) {
                char errbuf[128];
                av_strerror(ret, errbuf, sizeof(errbuf));
                fprintf(stderr, "SwDecoder: send_packet error: %s\n", errbuf);
            }
            return false;
        }
    } else {
        // Flush: send NULL packet
        avcodec_send_packet(codecCtx_, nullptr);
    }

    // Receive all available frames
    while (true) {
        int ret = avcodec_receive_frame(codecCtx_, frame_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            char errbuf[128];
            av_strerror(ret, errbuf, sizeof(errbuf));
            fprintf(stderr, "SwDecoder: receive_frame error: %s\n", errbuf);
            break;
        }

        // Convert planar YUV to NV12/P010 and upload to a ring buffer slot
        uint8_t* devPtr = convertAndUpload(frame_);
        av_frame_unref(frame_);

        if (!devPtr) {
            fprintf(stderr, "SwDecoder: convert+upload failed\n");
            break;
        }

        framePointers_.push_back(devPtr);
    }

    if (!framePointers_.empty()) {
        *pppFrame = framePointers_.data();
        *pnFrameReturned = (int)framePointers_.size();
    }

    return true;
}

uint8_t* SwDecoder::convertAndUpload(AVFrame* frame) {
    int w = frame->width;
    int h = frame->height;
    int chromaW = w / 2;
    int chromaH = h / 2;

    if (is10bit_) {
        // ---- P010 output: 10-bit in upper bits of uint16 ----

        // Copy Y plane, converting to P010 format (shift left by 6).
        // FFmpeg yuv420p10le stores 10-bit values in the lower 10 bits of uint16.
        // P010 stores them in the upper 10 bits.
        uint16_t* hostY = (uint16_t*)hostBuffer_;
        int hostYPitch = devicePitch_;  // bytes

        const uint16_t* srcY = (const uint16_t*)frame->data[0];
        int srcYLinesize = frame->linesize[0] / 2;  // linesize in bytes -> uint16 count

        for (int y = 0; y < h; y++) {
            uint16_t* dst = (uint16_t*)((uint8_t*)hostY + y * hostYPitch);
            const uint16_t* src = srcY + y * srcYLinesize;
            for (int x = 0; x < w; x++) {
                dst[x] = src[x] << 6;
            }
        }

        // Interleave U and V planes into UV plane (P010 format)
        uint16_t* hostUV = (uint16_t*)(hostBuffer_ + h * hostYPitch);
        int uvDstPitch = devicePitch_;  // bytes

        const uint16_t* srcU = (const uint16_t*)frame->data[1];
        const uint16_t* srcV = (const uint16_t*)frame->data[2];
        int srcULinesize = frame->linesize[1] / 2;
        int srcVLinesize = frame->linesize[2] / 2;

        interleavePlanes16(srcU, srcULinesize, srcV, srcVLinesize,
                           hostUV, uvDstPitch, chromaW, chromaH);
    } else {
        // ---- NV12 output: 8-bit ----

        // Copy Y plane
        uint8_t* hostY = hostBuffer_;
        int hostYPitch = devicePitch_;  // bytes

        const uint8_t* srcY = frame->data[0];
        int srcYLinesize = frame->linesize[0];

        for (int y = 0; y < h; y++) {
            memcpy(hostY + y * hostYPitch, srcY + y * srcYLinesize, w);
        }

        // Interleave U and V planes into UV plane (NV12 format)
        uint8_t* hostUV = hostBuffer_ + h * hostYPitch;
        int uvDstPitch = devicePitch_;  // bytes

        const uint8_t* srcU = frame->data[1];
        const uint8_t* srcV = frame->data[2];
        int srcULinesize = frame->linesize[1];
        int srcVLinesize = frame->linesize[2];

        interleavePlanes8(srcU, srcULinesize, srcV, srcVLinesize,
                          hostUV, uvDstPitch, chromaW, chromaH);
    }

    // Pick the next device buffer from the ring
    int bufIdx = deviceBufIdx_;
    deviceBufIdx_ = (deviceBufIdx_ + 1) % kNumDeviceBuffers;
    uint8_t* devBuf = deviceBuffers_[bufIdx];

    // Upload the NV12/P010 frame to GPU.
    // Using cudaMemcpyAsync with pinned host memory for best throughput.
    cudaError_t err = cudaMemcpyAsync(
        devBuf, hostBuffer_, perBufferSize_,
        cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        fprintf(stderr, "SwDecoder: cudaMemcpyAsync failed: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }

    // Synchronize to ensure the upload completes before the caller launches
    // GPU kernels on the inference stream. Since we reuse the host buffer
    // for the next frame, we must wait for the DMA transfer to finish.
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        fprintf(stderr, "SwDecoder: cudaStreamSynchronize failed: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }

    return devBuf;
}

void SwDecoder::interleavePlanes8(
    const uint8_t* uPlane, int uLinesize,
    const uint8_t* vPlane, int vLinesize,
    uint8_t* uvDst, int uvDstPitch,
    int chromaWidth, int chromaHeight)
{
    for (int y = 0; y < chromaHeight; y++) {
        const uint8_t* uRow = uPlane + y * uLinesize;
        const uint8_t* vRow = vPlane + y * vLinesize;
        uint8_t* dstRow = uvDst + y * uvDstPitch;
        for (int x = 0; x < chromaWidth; x++) {
            dstRow[x * 2]     = uRow[x];
            dstRow[x * 2 + 1] = vRow[x];
        }
    }
}

void SwDecoder::interleavePlanes16(
    const uint16_t* uPlane, int uLinesize,
    const uint16_t* vPlane, int vLinesize,
    uint16_t* uvDst, int uvDstPitch,
    int chromaWidth, int chromaHeight)
{
    // uvDstPitch is in bytes, convert to uint16 count for indexing
    int uvDstStride = uvDstPitch / 2;

    for (int y = 0; y < chromaHeight; y++) {
        const uint16_t* uRow = uPlane + y * uLinesize;
        const uint16_t* vRow = vPlane + y * vLinesize;
        uint16_t* dstRow = uvDst + y * uvDstStride;
        for (int x = 0; x < chromaWidth; x++) {
            // P010 format: U and V in upper 10 bits, shifted left by 6
            dstRow[x * 2]     = uRow[x] << 6;
            dstRow[x * 2 + 1] = vRow[x] << 6;
        }
    }
}
