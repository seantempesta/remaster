# Video Enhancement Experiment Platform

A research platform for improving video quality using ML models. The primary goal is removing compression artifacts from existing video libraries (e.g., BluRay rips) at native 1080p resolution, running on consumer hardware.

> **Status: Active development.** The distillation pipeline is producing real results — NAFNet student models that run 50x faster than the SCUNet teacher while approaching its quality.

## The Problem

Compressed video (H.264/H.265) introduces blocking artifacts, ringing, and loss of fine detail. Traditional denoising filters are either too aggressive (destroying detail) or too conservative (leaving artifacts). Neural networks can learn the difference — but the best models are too slow for practical use.

## The Approach

**Distillation**: Train a fast student model (NAFNet, pure CNN) to replicate the output of a slow but high-quality teacher (SCUNet, transformer-based). The student runs at real-time speeds while the teacher took 32+ hours per episode.

```
Compressed Frame ──> NAFNet (student) ──> Clean Frame
                        ^
                        │ trained to match
                        │
Compressed Frame ──> SCUNet (teacher) ──> Clean Frame (slow but excellent)
```

### Loss Functions

The training uses three complementary loss signals:

- **Charbonnier** (pixel loss) — smooth L1 for overall fidelity
- **DISTS** (perceptual loss) — [Deep Image Structure and Texture Similarity](https://github.com/dingkeyan93/DISTS), specifically designed for compression artifact assessment. Better calibrated to human perception than VGG feature matching
- **Focal Frequency Loss** — operates in the frequency domain to preserve high-frequency detail (edges, texture) that pixel and perceptual losses tend to smooth away

## Results So Far

| Metric | Value |
|--------|-------|
| Teacher (SCUNet) speed | 0.52 fps |
| Student (NAFNet w64) on H100 | 27.9 fps |
| Student (NAFNet w64) local RTX 3060 | 1.94 fps |
| Quality improvement over original | +2.39 dB PSNR |
| VRAM usage (local fp16) | 96 MB |

Current experiment: shrinking the model further (width32, fewer middle blocks) targeting 30 fps on a laptop GPU.

## Architecture

```
Local (Windows, RTX 3060 6GB)     Cloud (Modal, H100 80GB)
├── Inference pipelines            ├── Training (distillation)
├── TensorRT export/run            ├── Pair generation (SCUNet teacher)
├── Benchmarking                   └── Cloud inference pipeline
└── Quality evaluation
```

- **Local inference** — PyTorch with torch.compile, or TensorRT FP16
- **Cloud training** — [Modal](https://modal.com) for on-demand H100 GPU time
- **Streaming pipeline** — reads video, processes frames, writes video (no intermediate files)

## Project Structure

```
lib/           Shared code: NAFNet architecture, paths, ffmpeg utils, metrics
training/      Distillation training: losses, dataset, visualization, training loop
cloud/         Modal scripts for remote GPU training and inference
pipelines/     Production streaming denoisers (SCUNet, NAFNet, episode)
bench/         Benchmarking, quality comparison, checkpoint evaluation
tools/         Utilities: clip extraction, probing, training control
docs/          Research notes, setup guide, approach comparison
reference-code/ Git submodules: SCUNet, NAFNet, DISTS, RAFT, etc.
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `training/train_nafnet.py` | Distillation training loop with DISTS + FFT loss |
| `training/losses.py` | All loss functions: Charbonnier, DISTS, Focal Frequency |
| `cloud/modal_train.py` | Modal wrapper — upload data, train on H100, download checkpoints |
| `pipelines/denoise_nafnet.py` | Local NAFNet inference pipeline |
| `pipelines/denoise_batch.py` | Production SCUNet batch pipeline with threaded I/O |
| `bench/compare.py` | PSNR/SSIM metrics and side-by-side comparison |

## Quick Start

### Prerequisites

- Python 3.10, PyTorch 2.11+ with CUDA
- NVIDIA GPU (6GB+ VRAM for inference, training runs on cloud)
- [Modal](https://modal.com) account for cloud training (optional)

### Setup

```bash
# Create conda environment
conda create -n upscale python=3.10
conda activate upscale

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install opencv-python-headless numpy matplotlib

# Initialize submodules
git submodule update --init --recursive
```

### Run Inference (Local)

```bash
# Denoise a video with NAFNet
python pipelines/denoise_nafnet.py --input video.mp4 --checkpoint checkpoints/nafnet_best.pth
```

### Train (Cloud)

```bash
# Generate training pairs (SCUNet teacher output)
modal run cloud/modal_generate_pairs.py --input-dir data/clips

# Train NAFNet student with DISTS + FFT loss
modal run cloud/modal_train.py \
    --width 32 --middle-blk-num 4 \
    --perceptual-weight 0.1 --fft-weight 500 \
    --max-iters 25000
```

## Training Visualization

The training loop generates at each validation step:
- **Sample comparison images** (input | teacher target | model prediction)
- **Loss curve charts** (pixel, perceptual, FFT, total loss + validation PSNR)
- **JSON training log** for custom analysis

## Reference Code

This project builds on several excellent open-source implementations:

- [SCUNet](https://github.com/cszn/SCUNet) — Swin-Conv-UNet denoiser (teacher model)
- [NAFNet](https://github.com/megvii-research/NAFNet) — Nonlinear Activation Free Network (student architecture)
- [DISTS](https://github.com/dingkeyan93/DISTS) — Deep Image Structure and Texture Similarity (perceptual loss)
- [RAFT](https://github.com/princeton-vl/RAFT) — Optical flow estimation

## License

MIT
