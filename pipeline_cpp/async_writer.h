// async_writer.h -- Thread-safe async packet writer for MKV muxing
//
// Decouples the GPU pipeline from disk I/O. The main thread pushes
// packets into a queue; a background thread pulls them and writes
// to the muxer. This prevents slow disk writes (especially on exFAT)
// from stalling the GPU pipeline.
#pragma once

#include "mkv_muxer.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <atomic>
#include <cstdint>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
}

class AsyncWriter {
public:
    // Packet types for the write queue
    enum class PacketType {
        Video,
        Passthrough,  // audio/subtitle
        Flush         // sentinel: signals the writer thread to exit
    };

    struct WriteItem {
        PacketType type;

        // Video packet data (owned copy)
        std::vector<uint8_t> videoData;
        int64_t frameIndex = 0;

        // Passthrough packet (cloned AVPacket, owned)
        AVPacket* avPkt = nullptr;

        ~WriteItem() {
            if (avPkt) {
                av_packet_free(&avPkt);
            }
        }

        // Move-only
        WriteItem() = default;
        WriteItem(WriteItem&& o) noexcept
            : type(o.type), videoData(std::move(o.videoData)),
              frameIndex(o.frameIndex), avPkt(o.avPkt)
        {
            o.avPkt = nullptr;
        }
        WriteItem& operator=(WriteItem&& o) noexcept {
            if (this != &o) {
                if (avPkt) av_packet_free(&avPkt);
                type = o.type;
                videoData = std::move(o.videoData);
                frameIndex = o.frameIndex;
                avPkt = o.avPkt;
                o.avPkt = nullptr;
            }
            return *this;
        }
        WriteItem(const WriteItem&) = delete;
        WriteItem& operator=(const WriteItem&) = delete;
    };

    explicit AsyncWriter(MkvMuxer& muxer)
        : muxer_(muxer)
    {}

    ~AsyncWriter() {
        stop();
    }

    // Start the background writer thread.
    void start() {
        if (running_) return;
        running_ = true;
        thread_ = std::thread(&AsyncWriter::writerLoop, this);
    }

    // Signal the writer to finish and wait for all queued packets to be written.
    void stop() {
        if (!running_) return;

        // Push a flush sentinel
        {
            std::lock_guard<std::mutex> lock(mutex_);
            WriteItem item;
            item.type = PacketType::Flush;
            queue_.push(std::move(item));
        }
        cv_.notify_one();

        if (thread_.joinable()) {
            thread_.join();
        }
        running_ = false;
    }

    // Queue a video packet for writing. Takes ownership of the data (copies it).
    void pushVideoPacket(const uint8_t* data, int size, int64_t frameIndex) {
        WriteItem item;
        item.type = PacketType::Video;
        item.videoData.assign(data, data + size);
        item.frameIndex = frameIndex;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    // Queue an audio/subtitle packet for passthrough. Clones the AVPacket.
    void pushPassthroughPacket(AVPacket* pkt) {
        WriteItem item;
        item.type = PacketType::Passthrough;
        item.avPkt = av_packet_clone(pkt);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    // Number of packets written so far (video only, for progress tracking).
    int64_t videoPacketsWritten() const { return videoPacketsWritten_.load(); }

    // Current queue depth (for monitoring backpressure).
    size_t queueDepth() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    void writerLoop() {
        while (true) {
            WriteItem item;

            // Wait for a packet
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]{ return !queue_.empty(); });
                item = std::move(queue_.front());
                queue_.pop();
            }

            if (item.type == PacketType::Flush) {
                break;  // All done
            }

            if (item.type == PacketType::Video) {
                muxer_.writeVideoPacket(item.videoData.data(),
                                        (int)item.videoData.size(),
                                        item.frameIndex);
                videoPacketsWritten_++;
            }
            else if (item.type == PacketType::Passthrough) {
                if (item.avPkt) {
                    muxer_.writePassthroughPacket(item.avPkt);
                }
            }
        }
    }

    MkvMuxer& muxer_;
    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<WriteItem> queue_;
    std::atomic<bool> running_{false};
    std::atomic<int64_t> videoPacketsWritten_{0};
};
