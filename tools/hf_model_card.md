---
library_name: pytorch
license: mit
tags:
  - image-restoration
  - video-enhancement
  - denoising
  - compression-artifact-removal
  - drunet
  - distillation
  - tensorrt
  - onnx
pipeline_tag: image-to-image
---

# Remaster DRUNet -- Video Enhancement Models

Tiny neural networks that remove compression artifacts AND recover detail from video at native resolution, faster than real-time on a laptop GPU.

Both models are **DRUNet (UNetRes)** -- pure Conv+ReLU residual U-Nets with no attention, no normalization layers, no dynamic operations. 100% compatible with TensorRT INT8 quantization and CUDA graph capture.

## Models

| | Student | Teacher |
|--|---------|---------|
| **File** | `drunet_student.pth` | `drunet_teacher.pth` |
| **Parameters** | 1.06M | 32.6M |
| **Architecture** | nc=[16,32,64,128] nb=2 | nc=[64,128,256,512] nb=4 |
| **Quality (PSNR)** | 49.98 dB | 53.27 dB |
| **Sharpness** | ~100% of original | 107% of original |
| **Speed (RTX 3060)** | 57 fps C++ pipeline / 63 fps TRT FP16 / 64 fps TRT INT8 | ~5 fps |
| **VRAM** | ~500 MB | ~2 GB |
| **Checkpoint size** | 4 MB | 125 MB |
| **Use case** | Deployment / real-time | Quality reference / training |

Also included: `drunet_student.onnx` -- ONNX export with dynamic spatial dimensions for TensorRT engine building.

## How It Works

A large **teacher** model learns to enhance video using perceptual and pixel-level losses against high-quality targets. A tiny **student** model then learns to replicate the teacher's output at 30x the speed through knowledge distillation with feature matching.

### Training

- **Targets**: [SCUNet GAN](https://github.com/cszn/SCUNet) (perceptual denoiser) + Unsharp Mask -- denoise AND sharpen in one pass
- **Losses**: Charbonnier pixel + [DISTS](https://github.com/dingkeyan93/DISTS) perceptual (teacher), Charbonnier + feature matching (student)
- **Optimizer**: Prodigy (auto-tuned learning rate)
- **Data**: ~7K paired frames from diverse 1080p content (live action, animation, film)
- **Fine-tuning**: Full 1920x1080 frames to handle edge artifacts and letterboxing

### Key Finding

Mixed training data (HEVC artifact removal + synthetic edge-aware blur) produces a model that generalizes beyond its training tasks -- it denoises AND sharpens, often exceeding the quality of the original Bluray source material.

## Usage

### Python Inference

```python
import torch
import sys
sys.path.insert(0, "/path/to/KAIR")  # github.com/cszn/KAIR
from models.network_unet import UNetRes

# Student model (fast, deployment)
model = UNetRes(in_nc=3, out_nc=3, nc=[16, 32, 64, 128], nb=2,
                act_mode='R', bias=False)
ckpt = torch.load("drunet_student.pth", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["params"])
model.eval().half().cuda()

# Inference: input is [0, 1] float tensor, NCHW
with torch.no_grad(), torch.cuda.amp.autocast():
    output = model(input_tensor)
output = output.clamp(0, 1)
```

### Teacher Model

```python
# Same architecture, larger config
model = UNetRes(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4,
                act_mode='R', bias=False)
ckpt = torch.load("drunet_teacher.pth", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["params"])
```

### ONNX / TensorRT

```bash
# Build TensorRT FP16 engine from ONNX (one-time, ~2 min)
trtexec --onnx=drunet_student.onnx \
    --shapes=input:1x3x1080x1920 --fp16 --useCudaGraph \
    --saveEngine=drunet_student_1080p_fp16.engine

# INT8 quantization (requires calibration data)
trtexec --onnx=drunet_student.onnx \
    --shapes=input:1x3x1080x1920 --int8 --fp16 --useCudaGraph \
    --calib=calibration_data.bin \
    --saveEngine=drunet_student_1080p_int8.engine
```

### VapourSynth Real-Time Playback

The ONNX model works with [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) for TensorRT inference inside VapourSynth:

```python
import vapoursynth as vs
core = vs.core
clip = core.bs.VideoSource("input.mkv")
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
clip = core.ort.Model(clip, network_path="drunet_student.onnx",
                      backend=core.ort.Backend.TRT(fp16=True))
clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s="709")
clip.set_output()
```

## Checkpoint Format

All `.pth` files use the format `{"params": state_dict}`. Load with:

```python
state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)["params"]
```

## Architecture Details

DRUNet (UNetRes) from [KAIR](https://github.com/cszn/KAIR):

- 4-level encoder-decoder U-Net with skip connections
- Conv 3x3 + ReLU activation (mode='R'), no bias
- Residual blocks at each level (nb=2 for student, nb=4 for teacher)
- Channel progression: student [16,32,64,128], teacher [64,128,256,512]
- Input: 3-channel RGB [0,1], Output: 3-channel RGB [0,1]
- Spatial dimensions must be divisible by 8 (3 downsampling levels)
- No batch normalization, layer normalization, or attention -- pure CNN

## Requirements

- PyTorch 2.0+
- [KAIR](https://github.com/cszn/KAIR) for `UNetRes` architecture definition
- NVIDIA GPU with 1GB+ VRAM (student) or 4GB+ VRAM (teacher)
- Optional: TensorRT 10+ for optimized inference, VapourSynth + vs-mlrt for video pipeline

## License

MIT

## Citation

```
@misc{remaster-drunet,
  title={Remaster DRUNet: Real-Time Video Enhancement via Teacher-Student Distillation},
  url={https://github.com/seantempesta/remaster},
  year={2026}
}
```

## Acknowledgments

- [KAIR](https://github.com/cszn/KAIR) -- DRUNet/UNetRes architecture (MIT license)
- [SCUNet](https://github.com/cszn/SCUNet) -- Training target generation
- [DISTS](https://github.com/dingkeyan93/DISTS) -- Perceptual loss function
- [vs-mlrt](https://github.com/AmusementClub/vs-mlrt) -- VapourSynth TensorRT inference
