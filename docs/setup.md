# Setup Guide

## Prerequisites

- Windows 10/11 or Linux with NVIDIA GPU (tested on RTX 3060 Laptop, 6GB VRAM)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git
- CUDA Toolkit 13.0+ (for TensorRT and C++ pipeline)

## Python Environment

```bash
conda create -n remaster python=3.12
conda activate remaster
```

**Install PyTorch with CUDA last** -- other packages sometimes pull in CPU-only PyTorch:

```bash
pip install opencv-python-headless numpy matplotlib av timm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## Git Submodules

Reference code (KAIR, SCUNet, DISTS, etc.) lives in `reference-code/` as submodules:

```bash
git submodule update --init --recursive
```

## Model Weights

Checkpoints are git-ignored. Download pretrained weights:

Download from [HuggingFace](https://huggingface.co/seantempesta/remaster-drunet):

| Model | Path | Size |
|-------|------|------|
| Student | `checkpoints/drunet_student/final.pth` | 4 MB |
| Student ONNX | `checkpoints/drunet_student/drunet_student.onnx` | 2 MB |
| Teacher | `checkpoints/drunet_teacher/final.pth` | 125 MB |

For training target generation, you also need:

| Model | Path | Source |
|-------|------|--------|
| SCUNet GAN | `reference-code/SCUNet/model_zoo/scunet_color_real_gan.pth` | [SCUNet releases](https://github.com/cszn/SCUNet) |

## TensorRT (for optimized inference)

TensorRT provides the fastest inference path (63 fps at 1080p vs 24 fps with torch.compile).

1. Install [TensorRT 10.16+](https://developer.nvidia.com/tensorrt) or use the bundled trtexec from vs-mlrt
2. Export ONNX and build an engine:

```bash
python tools/export_onnx.py

trtexec --onnx=checkpoints/drunet_student/drunet_student.onnx \
    --shapes=input:1x3x1080x1920 --fp16 --useCudaGraph \
    --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw \
    --saveEngine=checkpoints/drunet_student/drunet_student_1080p_fp16.engine
```

Engines are GPU-specific and must be rebuilt when changing hardware or drivers.

## VapourSynth (for NVEncC pipeline)

1. Install [VapourSynth R68+](https://www.vapoursynth.com/)
2. Install [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) plugin (TensorRT backend)
3. Install [BestSource](https://github.com/vapoursynth/bestsource) for frame-accurate decoding
4. Install [NVEncC](https://github.com/rigaya/NVEnc) for hardware encoding with `--vpy` support

## C++ Pipeline Build

See the [deployment guide](deployment.md#building-the-c-pipeline) for build instructions. Requires Visual Studio Build Tools 2022, CUDA Toolkit, CMake, and TensorRT headers.

## FFmpeg

A modern ffmpeg (7.0+) with NVENC support is recommended. Place at `bin/ffmpeg.exe` or ensure it's on PATH. The `lib/ffmpeg_utils.py` module auto-detects the best available ffmpeg.

## Modal (Cloud Training)

For cloud GPU training via [Modal](https://modal.com):

```bash
pip install modal
modal token set
```

Create a W&B secret for experiment tracking:

```bash
modal secret create wandb-api-key WANDB_API_KEY=your_key_here
```

See the [training guide](training.md) for details.
