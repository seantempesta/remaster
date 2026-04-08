// simple_demuxer.h -- Minimal FFmpeg 7.x demuxer for feeding NVDEC
//
// Replaces the Video SDK's FFmpegDemuxer.h which uses removed FFmpeg APIs
// (av_register_all, stack-allocated AVPacket, ck() overload for int).
// This implementation uses only modern FFmpeg 7.x API.
//
// Supports two modes:
// - Demux(): returns only video packets (original behavior)
// - DemuxAny(): returns ALL packets (video + audio + subtitle) for muxing
#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
}

#include <cstdint>
#include <iostream>
#include <string>

// Map FFmpeg codec ID to NVDEC codec ID (same as Video SDK's FFmpeg2NvCodecId)
#include <cuviddec.h>
inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
    switch (id) {
    case AV_CODEC_ID_MPEG1VIDEO: return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO: return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4:      return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_VC1:        return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264:       return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC:       return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8:        return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9:        return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG:      return cudaVideoCodec_JPEG;
    default:                     return cudaVideoCodec_NumCodecs;
    }
}

class SimpleDemuxer {
public:
    SimpleDemuxer(const char* filePath) {
        avformat_network_init();

        int ret = avformat_open_input(&fmtCtx_, filePath, nullptr, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to open input: " << filePath << std::endl;
            return;
        }

        ret = avformat_find_stream_info(fmtCtx_, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to find stream info" << std::endl;
            return;
        }

        videoStreamIdx_ = av_find_best_stream(fmtCtx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (videoStreamIdx_ < 0) {
            std::cerr << "No video stream found" << std::endl;
            return;
        }

        AVCodecParameters* par = fmtCtx_->streams[videoStreamIdx_]->codecpar;
        codecId_ = par->codec_id;
        width_   = par->width;
        height_  = par->height;

        // Determine bit depth from pixel format
        bitDepth_ = 8;
        if (par->format == AV_PIX_FMT_YUV420P10LE || par->format == AV_PIX_FMT_YUV420P10BE)
            bitDepth_ = 10;
        if (par->format == AV_PIX_FMT_YUV420P12LE || par->format == AV_PIX_FMT_YUV420P12BE)
            bitDepth_ = 12;

        // Allocate packets
        pkt_ = av_packet_alloc();
        pktFiltered_ = av_packet_alloc();

        // Set up bitstream filter for H.264/HEVC in MP4/MKV/FLV containers
        // (converts from length-prefixed NALUs to Annex B start codes for NVDEC)
        bool isContainer = fmtCtx_->iformat && (
            !strcmp(fmtCtx_->iformat->name, "mov,mp4,m4a,3gp,3g2,mj2") ||
            !strcmp(fmtCtx_->iformat->name, "flv") ||
            !strcmp(fmtCtx_->iformat->name, "matroska,webm")
        );
        const char* bsfName = nullptr;
        if (isContainer && codecId_ == AV_CODEC_ID_H264)
            bsfName = "h264_mp4toannexb";
        else if (isContainer && codecId_ == AV_CODEC_ID_HEVC)
            bsfName = "hevc_mp4toannexb";

        if (bsfName) {
            const AVBitStreamFilter* bsf = av_bsf_get_by_name(bsfName);
            if (bsf) {
                ret = av_bsf_alloc(bsf, &bsfCtx_);
                if (ret >= 0) {
                    avcodec_parameters_copy(bsfCtx_->par_in,
                        fmtCtx_->streams[videoStreamIdx_]->codecpar);
                    ret = av_bsf_init(bsfCtx_);
                    if (ret < 0) {
                        av_bsf_free(&bsfCtx_);
                        bsfCtx_ = nullptr;
                        std::cerr << "Warning: BSF init failed, Annex B conversion disabled" << std::endl;
                    }
                }
            }
        }

        valid_ = true;
    }

    ~SimpleDemuxer() {
        if (pkt_)        av_packet_free(&pkt_);
        if (pktFiltered_) av_packet_free(&pktFiltered_);
        if (bsfCtx_)     av_bsf_free(&bsfCtx_);
        if (fmtCtx_)     avformat_close_input(&fmtCtx_);
    }

    // Non-copyable
    SimpleDemuxer(const SimpleDemuxer&) = delete;
    SimpleDemuxer& operator=(const SimpleDemuxer&) = delete;

    // ---- Basic info ----
    bool IsValid() const { return valid_; }
    AVCodecID GetVideoCodec() const { return codecId_; }
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    int GetBitDepth() const { return bitDepth_; }

    // ---- Stream info (for muxer setup) ----
    int GetVideoStreamIndex() const { return videoStreamIdx_; }
    int GetNumStreams() const { return fmtCtx_ ? (int)fmtCtx_->nb_streams : 0; }
    AVStream* GetStream(int idx) const {
        if (!fmtCtx_ || idx < 0 || idx >= (int)fmtCtx_->nb_streams) return nullptr;
        return fmtCtx_->streams[idx];
    }
    AVFormatContext* GetFormatContext() const { return fmtCtx_; }

    AVRational GetVideoTimeBase() const {
        if (!fmtCtx_ || videoStreamIdx_ < 0) return {1, 1};
        return fmtCtx_->streams[videoStreamIdx_]->time_base;
    }

    double GetFrameRate() const {
        if (!fmtCtx_ || videoStreamIdx_ < 0) return 0.0;
        AVRational fr = fmtCtx_->streams[videoStreamIdx_]->avg_frame_rate;
        if (fr.num == 0 || fr.den == 0) {
            fr = fmtCtx_->streams[videoStreamIdx_]->r_frame_rate;
        }
        if (fr.num == 0 || fr.den == 0) return 0.0;
        return av_q2d(fr);
    }

    // ---- Original video-only demux ----
    // Read the next video packet. Returns false at EOF.
    // Sets *ppVideo to the compressed data and *pnVideoBytes to its size.
    // The returned pointer is valid until the next Demux() call.
    bool Demux(uint8_t** ppVideo, int* pnVideoBytes) {
        if (!valid_ || !fmtCtx_) return false;

        *ppVideo = nullptr;
        *pnVideoBytes = 0;

        // Read frames, skipping non-video streams
        while (true) {
            av_packet_unref(pkt_);
            int ret = av_read_frame(fmtCtx_, pkt_);
            if (ret < 0) return false;  // EOF or error

            if (pkt_->stream_index == videoStreamIdx_)
                break;
        }

        return filterVideoPacket(ppVideo, pnVideoBytes);
    }

    // ---- Multi-stream demux (for muxing with audio passthrough) ----
    // Read the next packet from any stream. Returns false at EOF.
    //
    // For video packets: *ppData and *pnBytes contain the (BSF-filtered) data
    //   for feeding NVDEC, and *pStreamIndex == GetVideoStreamIndex().
    //
    // For audio/subtitle packets: the raw AVPacket is accessible via
    //   GetCurrentPacket() for passing to the muxer.
    bool DemuxAny(uint8_t** ppData, int* pnBytes, int* pStreamIndex) {
        if (!valid_ || !fmtCtx_) return false;

        *ppData = nullptr;
        *pnBytes = 0;

        av_packet_unref(pkt_);
        int ret = av_read_frame(fmtCtx_, pkt_);
        if (ret < 0) return false;  // EOF or error

        *pStreamIndex = pkt_->stream_index;

        if (pkt_->stream_index == videoStreamIdx_) {
            return filterVideoPacket(ppData, pnBytes);
        } else {
            // Non-video: data accessible via GetCurrentPacket()
            *ppData = pkt_->data;
            *pnBytes = pkt_->size;
            return true;
        }
    }

    // Get the current raw AVPacket (for audio/subtitle passthrough to muxer).
    // Valid until the next Demux/DemuxAny call.
    AVPacket* GetCurrentPacket() const { return pkt_; }

private:
    // Apply bitstream filter to the current video packet (pkt_).
    // Returns the filtered data via ppVideo/pnVideoBytes.
    bool filterVideoPacket(uint8_t** ppVideo, int* pnVideoBytes) {
        if (bsfCtx_) {
            av_packet_unref(pktFiltered_);
            int ret = av_bsf_send_packet(bsfCtx_, pkt_);
            if (ret < 0) return false;
            ret = av_bsf_receive_packet(bsfCtx_, pktFiltered_);
            if (ret < 0) return false;
            *ppVideo = pktFiltered_->data;
            *pnVideoBytes = pktFiltered_->size;
        } else {
            *ppVideo = pkt_->data;
            *pnVideoBytes = pkt_->size;
        }
        return true;
    }

    AVFormatContext* fmtCtx_ = nullptr;
    AVBSFContext*    bsfCtx_ = nullptr;
    AVPacket*        pkt_ = nullptr;
    AVPacket*        pktFiltered_ = nullptr;

    int videoStreamIdx_ = -1;
    AVCodecID codecId_ = AV_CODEC_ID_NONE;
    int width_   = 0;
    int height_  = 0;
    int bitDepth_ = 8;
    bool valid_  = false;
};
