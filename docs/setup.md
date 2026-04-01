# Setup Guide

## Prerequisites

- Windows 10/11 with NVIDIA GPU (tested on RTX 3060 6GB)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

## Conda Environment

```bash
conda create -n upscale python=3.10
conda activate upscale
```

## Dependencies

Install project dependencies first:

```bash
pip install -r requirements.txt
```

**Critical: Install PyTorch CUDA last** (or with `--no-deps`). Other packages sometimes pull in CPU-only PyTorch as a dependency, overwriting the CUDA build.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## Git Submodules

Reference code (SCUNet, RAFT, NAFNet, etc.) lives in `reference-code/` as git submodules:

```bash
git submodule update --init --recursive
```

## Model Weights

Model weights are git-ignored (*.pth pattern). Download and place them manually:

| Model | Path | Source |
|-------|------|--------|
| SCUNet (PSNR) | `reference-code/SCUNet/model_zoo/scunet_color_real_psnr.pth` | [GitHub releases](https://github.com/cszn/SCUNet) |
| SCUNet (GAN) | `reference-code/SCUNet/model_zoo/scunet_color_real_gan.pth` | Same |
| NAFNet-SIDD-w64 | `reference-code/NAFNet/experiments/pretrained_models/NAFNet-SIDD-width64.pth` | [Google Drive](https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR) |
| RAFT (things) | `reference-code/RAFT/models/raft-things.pth` | [RAFT repo](https://github.com/princeton-vl/RAFT) |
| RAFT (sintel) | `reference-code/RAFT/models/raft-sintel.pth` | Same |
| VDA (vits) | `reference-code/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth` | [VDA repo](https://github.com/DepthAnything/Video-Depth-Anything) |

## FFmpeg

The streaming pipelines need ffmpeg. Options:

1. **imageio-ffmpeg** (recommended): `pip install imageio-ffmpeg` — bundles a static ffmpeg binary
2. **Manual**: Place `ffmpeg.exe` in `bin/` or add to system PATH

For hardware encoding (hevc_nvenc), you need an ffmpeg build with NVENC support.

## Modal (Cloud Training)

For cloud GPU access via [Modal](https://modal.com):

```bash
pip install modal
modal token set
```

Requires a Modal account with billing configured. See `cloud/` scripts for usage.
