# Remaster — GPU video enhancement pipeline
#
# Runs the full pipeline in a Linux container with native CUDA + Triton support:
#   VapourSynth + libtorch (AOT Inductor) + ffmpeg NVENC
#
# Build:  docker compose build
# Run:    docker compose run remaster encode input.mkv output.mkv
# Shell:  docker compose run remaster bash

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git cmake ninja-build pkg-config \
    ffmpeg \
    libxxhash-dev zlib1g-dev libbz2-dev liblzma-dev libssl-dev \
    curl wget \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# PyTorch + Triton (for torch.compile / AOT Inductor)
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Build tools for AOT Inductor and custom plugins
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# Working directory
WORKDIR /app

# Copy project code (lib/ and remaster/ scripts)
COPY lib/ /app/lib/
COPY remaster/ /app/remaster/
COPY reference-code/vs-mlrt/scripts/vsmlrt.py /app/reference-code/vs-mlrt/scripts/vsmlrt.py

# Copy checkpoints (model weights + ONNX)
# These are large — use .dockerignore to exclude unnecessary files
COPY checkpoints/nafnet_w32_mid4/nafnet_best.pth /app/checkpoints/nafnet_w32_mid4/nafnet_best.pth

# Verify CUDA + PyTorch
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Entry point
COPY remaster/encode.py /app/remaster/encode.py
ENTRYPOINT ["python", "remaster/encode.py"]
