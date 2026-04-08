// mkv_muxer.h -- MKV container muxer with audio/subtitle passthrough
//
// Wraps libavformat to produce a proper MKV file with:
// - HEVC video stream with BT.709 color metadata
// - Audio passthrough (any codec, no re-encoding)
// - Subtitle passthrough
// - Proper PTS/DTS timestamps
// - Interleaved packet writing
//
// Header writing is deferred until the first video packet arrives,
// because NVENC's first packet contains VPS/SPS/PPS extradata that
// the MKV muxer needs.
#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
}

#include <cstdint>
#include <string>
#include <vector>
#include <iostream>

class MkvMuxer {
public:
    MkvMuxer() = default;

    ~MkvMuxer() {
        close();
    }

    // Non-copyable
    MkvMuxer(const MkvMuxer&) = delete;
    MkvMuxer& operator=(const MkvMuxer&) = delete;

    // Initialize the muxer. Creates output context and streams but does NOT
    // write the header yet (deferred until first video packet).
    //
    // inputCtx: the demuxer's format context (for stream info)
    // videoStreamIdx: which input stream is video
    // outputPath: path to write the .mkv file
    // width, height: video dimensions
    // fps: frame rate for video timestamps
    // is10bit: whether the NVENC output is 10-bit HEVC
    // passAudio: whether to include audio/subtitle streams
    bool open(AVFormatContext* inputCtx, int videoStreamIdx,
              const char* outputPath,
              int width, int height, double fps,
              bool is10bit, bool passAudio)
    {
        inputCtx_ = inputCtx;
        inputVideoIdx_ = videoStreamIdx;
        is10bit_ = is10bit;
        width_ = width;
        height_ = height;

        int ret = avformat_alloc_output_context2(&ofmtCtx_, nullptr, "matroska", outputPath);
        if (ret < 0 || !ofmtCtx_) {
            char errbuf[128];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "Failed to allocate output context: " << errbuf << std::endl;
            return false;
        }

        // --- Video output stream ---
        AVStream* outVideo = avformat_new_stream(ofmtCtx_, nullptr);
        if (!outVideo) {
            std::cerr << "Failed to create output video stream" << std::endl;
            return false;
        }

        outVideoIdx_ = outVideo->index;

        outVideo->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        outVideo->codecpar->codec_id = AV_CODEC_ID_HEVC;
        outVideo->codecpar->width = width;
        outVideo->codecpar->height = height;
        outVideo->codecpar->format = is10bit ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;

        // BT.709 color metadata
        outVideo->codecpar->color_range = AVCOL_RANGE_MPEG;
        outVideo->codecpar->color_primaries = AVCOL_PRI_BT709;
        outVideo->codecpar->color_trc = AVCOL_TRC_BT709;
        outVideo->codecpar->color_space = AVCOL_SPC_BT709;

        // Time base: 1/fps for simple frame-counting PTS
        videoFps_ = fps;
        outVideo->time_base = av_d2q(1.0 / fps, 1000000);
        videoTimeBase_ = outVideo->time_base;

        outVideo->avg_frame_rate = av_d2q(fps, 1000000);
        outVideo->r_frame_rate = outVideo->avg_frame_rate;

        // --- Audio/subtitle passthrough streams ---
        if (passAudio) {
            for (unsigned i = 0; i < inputCtx->nb_streams; i++) {
                AVStream* inStream = inputCtx->streams[i];
                if ((int)i == videoStreamIdx)
                    continue;

                AVMediaType type = inStream->codecpar->codec_type;
                if (type != AVMEDIA_TYPE_AUDIO && type != AVMEDIA_TYPE_SUBTITLE)
                    continue;

                AVStream* outStream = avformat_new_stream(ofmtCtx_, nullptr);
                if (!outStream) {
                    std::cerr << "Failed to create output stream for input stream " << i << std::endl;
                    continue;
                }

                // Copy codec parameters verbatim
                ret = avcodec_parameters_copy(outStream->codecpar, inStream->codecpar);
                if (ret < 0) {
                    std::cerr << "Failed to copy codec params for stream " << i << std::endl;
                    continue;
                }
                outStream->codecpar->codec_tag = 0;  // let muxer choose

                // Copy time base from input
                outStream->time_base = inStream->time_base;

                // Map input stream index -> output stream index
                streamMap_.push_back({(int)i, outStream->index});
            }
        }

        // --- Open output file (but don't write header yet) ---
        if (!(ofmtCtx_->oformat->flags & AVFMT_NOFILE)) {
            ret = avio_open(&ofmtCtx_->pb, outputPath, AVIO_FLAG_WRITE);
            if (ret < 0) {
                char errbuf[128];
                av_strerror(ret, errbuf, sizeof(errbuf));
                std::cerr << "Failed to open output file: " << errbuf << std::endl;
                return false;
            }
        }

        open_ = true;
        return true;
    }

    // Write an encoded video packet (raw HEVC bitstream from NVENC).
    // frameIndex is the encode-order frame number (0-based).
    // The first call triggers header writing (extradata extraction).
    bool writeVideoPacket(const uint8_t* data, int size, int64_t frameIndex) {
        if (!open_) return false;

        // On first video packet, extract HEVC extradata and write header
        if (!headerWritten_) {
            if (!extractExtradataAndWriteHeader(data, size)) {
                return false;
            }
            // After header write, the muxer may change the time base
            // (Matroska uses 1/1000). Cache the actual output time base.
            actualVideoTimeBase_ = ofmtCtx_->streams[outVideoIdx_]->time_base;
        }

        AVPacket* pkt = av_packet_alloc();
        if (!pkt) return false;

        // Must copy data since av_interleaved_write_frame may buffer it
        int ret = av_new_packet(pkt, size);
        if (ret < 0) {
            av_packet_free(&pkt);
            return false;
        }
        memcpy(pkt->data, data, size);

        pkt->stream_index = outVideoIdx_;

        // Convert from "frame index" time base (1/fps) to actual output time base
        // PTS in frame-counting units: frameIndex ticks at videoTimeBase_ (1/fps)
        // Rescale to actual output time base (e.g., 1/1000 for Matroska)
        pkt->pts = av_rescale_q(frameIndex, videoTimeBase_, actualVideoTimeBase_);
        pkt->dts = pkt->pts;
        pkt->duration = av_rescale_q(1, videoTimeBase_, actualVideoTimeBase_);

        // Check for IDR frame (HEVC NALU type 19 or 20) and set keyframe flag
        if (isHevcKeyframe(data, size)) {
            pkt->flags |= AV_PKT_FLAG_KEY;
        }

        ret = av_interleaved_write_frame(ofmtCtx_, pkt);
        av_packet_free(&pkt);

        if (ret < 0) {
            char errbuf[128];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "Failed to write video packet (frame " << frameIndex << "): " << errbuf << std::endl;
            return false;
        }
        return true;
    }

    // Pass through a non-video packet (audio/subtitle) from the demuxer.
    // If the header hasn't been written yet, queues the packet internally.
    bool writePassthroughPacket(AVPacket* inPkt) {
        if (!open_) return false;

        // If header not written yet, queue the packet for later
        if (!headerWritten_) {
            AVPacket* clone = av_packet_clone(inPkt);
            if (clone) pendingPassthrough_.push_back(clone);
            return true;
        }

        return writePassthroughPacketInternal(inPkt);
    }

    // Finalize: write trailer and close file.
    void close() {
        if (!open_) return;

        if (headerWritten_) {
            av_write_trailer(ofmtCtx_);
        }

        if (ofmtCtx_ && !(ofmtCtx_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&ofmtCtx_->pb);
        }

        // Free any pending passthrough packets
        for (auto* pkt : pendingPassthrough_) {
            av_packet_free(&pkt);
        }
        pendingPassthrough_.clear();

        if (ofmtCtx_) {
            avformat_free_context(ofmtCtx_);
            ofmtCtx_ = nullptr;
        }
        open_ = false;
        headerWritten_ = false;
    }

    int getOutputVideoIndex() const { return outVideoIdx_; }
    bool isOpen() const { return open_; }

private:
    struct StreamMapping {
        int inputIdx;
        int outputIdx;
    };

    // Check if an Annex B HEVC packet contains an IDR frame
    static bool isHevcKeyframe(const uint8_t* data, int size) {
        const uint8_t* p = data;
        const uint8_t* end = data + size;
        while (p < end - 4) {
            if (p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 1) {
                int naluType = (p[4] >> 1) & 0x3F;
                // IDR_W_RADL=19, IDR_N_LP=20, CRA=21
                if (naluType == 19 || naluType == 20 || naluType == 21)
                    return true;
            }
            p++;
        }
        return false;
    }

    // Extract VPS/SPS/PPS from the first NVENC packet and set as extradata,
    // then write the MKV header. Also flushes any pending passthrough packets.
    bool extractExtradataAndWriteHeader(const uint8_t* data, int size) {
        // NVENC outputs Annex B format. The first packet typically contains
        // VPS, SPS, PPS NALUs followed by the first IDR slice.
        // We need to extract the parameter sets as extradata for the muxer.
        //
        // For MKV (Matroska), the muxer's bitstream filter will convert
        // from Annex B to length-prefixed if needed. We can either:
        // (a) Set extradata to the full Annex B parameter set bytes
        // (b) Let the muxer handle it via AVFMT_FLAG_AUTO_BSF
        //
        // Approach: find all parameter set NALUs (VPS=32, SPS=33, PPS=34)
        // and set them as extradata.

        std::vector<uint8_t> extradata;
        const uint8_t* p = data;
        const uint8_t* end = data + size;

        // Scan for Annex B start codes and extract parameter set NALUs
        while (p < end - 4) {
            // Find next start code (00 00 00 01 or 00 00 01)
            const uint8_t* naluStart = nullptr;
            int startCodeLen = 0;

            if (p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 1) {
                naluStart = p + 4;
                startCodeLen = 4;
            } else if (p[0] == 0 && p[1] == 0 && p[2] == 1) {
                naluStart = p + 3;
                startCodeLen = 3;
            } else {
                p++;
                continue;
            }

            if (naluStart >= end) break;

            // HEVC NALU type is in bits 1-6 of the first byte
            int naluType = (naluStart[0] >> 1) & 0x3F;

            // VPS=32, SPS=33, PPS=34
            if (naluType == 32 || naluType == 33 || naluType == 34) {
                // Find the end of this NALU (next start code or end of data)
                const uint8_t* naluEnd = naluStart;
                while (naluEnd < end - 3) {
                    if (naluEnd[0] == 0 && naluEnd[1] == 0 &&
                        (naluEnd[2] == 1 || (naluEnd[2] == 0 && naluEnd + 3 < end && naluEnd[3] == 1)))
                        break;
                    naluEnd++;
                }
                if (naluEnd >= end - 3) naluEnd = end;

                // Include the start code in extradata
                extradata.insert(extradata.end(), p, naluEnd);

                p = naluEnd;
            } else {
                // Non-parameter-set NALU: stop scanning
                break;
            }
        }

        // Set extradata on the video stream
        AVStream* outVideo = ofmtCtx_->streams[outVideoIdx_];
        if (!extradata.empty()) {
            // Allocate with av_malloc for proper alignment
            uint8_t* ed = (uint8_t*)av_malloc(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE);
            if (ed) {
                memcpy(ed, extradata.data(), extradata.size());
                memset(ed + extradata.size(), 0, AV_INPUT_BUFFER_PADDING_SIZE);
                outVideo->codecpar->extradata = ed;
                outVideo->codecpar->extradata_size = (int)extradata.size();
            }
        }

        // Write header
        AVDictionary* opts = nullptr;
        int ret = avformat_write_header(ofmtCtx_, &opts);
        av_dict_free(&opts);
        if (ret < 0) {
            char errbuf[128];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "Failed to write MKV header: " << errbuf << std::endl;
            return false;
        }

        headerWritten_ = true;

        // Flush any pending passthrough packets
        for (auto* pkt : pendingPassthrough_) {
            writePassthroughPacketInternal(pkt);
            av_packet_free(&pkt);
        }
        pendingPassthrough_.clear();

        return true;
    }

    bool writePassthroughPacketInternal(AVPacket* inPkt) {
        // Find output stream index for this input stream
        int outIdx = -1;
        for (const auto& mapping : streamMap_) {
            if (mapping.inputIdx == inPkt->stream_index) {
                outIdx = mapping.outputIdx;
                break;
            }
        }
        if (outIdx < 0) return false;

        AVPacket* pkt = av_packet_clone(inPkt);
        if (!pkt) return false;

        // Rescale timestamps from input time base to output time base
        AVStream* inStream = inputCtx_->streams[inPkt->stream_index];
        AVStream* outStream = ofmtCtx_->streams[outIdx];
        pkt->stream_index = outIdx;
        av_packet_rescale_ts(pkt, inStream->time_base, outStream->time_base);

        int ret = av_interleaved_write_frame(ofmtCtx_, pkt);
        av_packet_free(&pkt);

        return ret >= 0;
    }

    AVFormatContext* ofmtCtx_ = nullptr;
    AVFormatContext* inputCtx_ = nullptr;  // borrowed, not owned
    int inputVideoIdx_ = -1;
    int outVideoIdx_ = -1;
    bool is10bit_ = false;
    int width_ = 0;
    int height_ = 0;

    double videoFps_ = 0.0;
    AVRational videoTimeBase_ = {0, 1};       // desired time base (1/fps)
    AVRational actualVideoTimeBase_ = {0, 1}; // actual time base after header write

    std::vector<StreamMapping> streamMap_;
    std::vector<AVPacket*> pendingPassthrough_;  // queued until header is written
    bool open_ = false;
    bool headerWritten_ = false;
};
