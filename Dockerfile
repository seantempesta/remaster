# Remaster -- GPU video enhancement pipeline
#
# Removes compression artifacts and recovers detail from 1080p video at 50+ fps.
# Uses TensorRT inference via a C++ zero-copy pipeline (NVDEC -> TRT -> NVENC).
#
# Build:  docker build -t remaster .
# Run:    docker run --gpus all -v /path/to/videos:/data remaster /data/input.mkv /data/output.mkv
# Shell:  docker run --gpus all -it --entrypoint bash remaster
#
# Requires: NVIDIA Container Toolkit (--gpus all), Linux host with NVIDIA driver 550+

# =============================================================================
# Stage 1: Build the C++ pipeline
# =============================================================================
FROM nvcr.io/nvidia/tensorrt:26.03-py3 AS builder

# FFmpeg dev libs (muxer/demuxer) + ninja for fast builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        libavformat-dev libavcodec-dev libavutil-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy only what's needed for the C++ build (ordered by change frequency)
COPY reference-code/video-sdk-samples/Samples/NvCodec/ /src/reference-code/video-sdk-samples/Samples/NvCodec/
COPY reference-code/video-sdk-samples/Samples/Utils/   /src/reference-code/video-sdk-samples/Samples/Utils/
COPY pipeline_cpp/ /src/pipeline_cpp/

# Build with broad GPU architecture support
RUN cmake -G Ninja -B /src/pipeline_cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -S /src/pipeline_cpp \
    && cmake --build /src/pipeline_cpp/build --config Release -j$(nproc)

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM nvcr.io/nvidia/tensorrt:26.03-py3

# Request NVDEC + NVENC driver libraries from the NVIDIA Container Toolkit
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# FFmpeg runtime libs (muxer/demuxer for the C++ pipeline) + ffprobe for resolution detection
RUN apt-get update && apt-get install -y --no-install-recommends \
        libavformat60 libavcodec60 libavutil58 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary
COPY --from=builder /src/pipeline_cpp/build/remaster_pipeline /app/bin/remaster_pipeline

# Copy the pre-exported ONNX model (2 MB, resolution-agnostic)
COPY checkpoints/drunet_student/drunet_student.onnx /app/model/drunet_student.onnx

# Entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Engine cache directory (mount a volume here for persistence across runs)
RUN mkdir -p /app/engines

# trtexec is pre-installed in the NGC container
ENV PATH="/opt/tensorrt/bin:${PATH}"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
