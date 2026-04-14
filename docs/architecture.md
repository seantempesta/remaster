# Technical Architecture

This document covers the model design, training methodology, deployment pipeline, and research history behind Remaster. It is intended for ML engineers who want to understand the technical details beyond what the [README](../README.md) covers.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Training Approach](#training-approach)
3. [Training Data](#training-data)
4. [Loss Functions](#loss-functions)
5. [Key Finding: Generalization Beyond Artifact Removal](#key-finding-generalization-beyond-artifact-removal)
6. [Deployment Pipeline](#deployment-pipeline)
7. [INT8 Quantization](#int8-quantization)
8. [Experiments and Research History](#experiments-and-research-history)
9. [Future Directions](#future-directions)

---

## Model Architecture

Both the teacher and student models use **DRUNet** (UNetRes), a residual U-Net architecture from the [KAIR](https://github.com/cszn/KAIR) repository (MIT license). The architecture is a 4-level encoder-decoder with skip connections, using only Conv 3x3 + ReLU residual blocks. There is no batch normalization, no layer normalization, and no attention of any kind.

### Why DRUNet

The choice of architecture was driven by deployment constraints:

- **TensorRT compatibility.** Every operation is a static Conv2d or ReLU. No dynamic shapes, no data-dependent branching, no attention with variable sequence lengths. TensorRT can fuse and optimize the entire graph without fallback to slower plugins.
- **CUDA graph capture.** Because the model has no dynamic operations, the entire inference pass can be captured as a single CUDA graph. This eliminates kernel launch overhead entirely and is critical for reaching 55+ fps at 1080p.
- **INT8 quantization.** Conv+ReLU blocks quantize cleanly. The only INT8-sensitive operations are the skip-connection additions (discussed in [INT8 Quantization](#int8-quantization)).
- **No attention overhead.** Self-attention is O(n^2) in spatial tokens. At 1080p (2M pixels), even efficient attention implementations add significant latency. DRUNet processes all spatial locations in parallel through convolutions.

### Teacher Configuration

| Property | Value |
|----------|-------|
| Parameters | 32.6M |
| Channels per level | nc = [64, 128, 256, 512] |
| Residual blocks per level | nb = 4 |
| Activation | ReLU (act_mode='R') |
| Bias | False |
| Input / Output | 3-channel RGB [0, 1] |
| Spatial constraint | Height and width divisible by 8 |
| Checkpoint size | 125 MB |
| Inference speed (RTX 3060) | ~5 fps at 1080p |

The teacher serves as the quality ceiling. It was pretrained from `drunet_deblocking_color.pth` (Gaussian denoising weights from KAIR) and fine-tuned on our paired training data. At 53.27 dB PSNR and 107% sharpness relative to the original source, the teacher often exceeds Bluray quality.

### Student Configuration

| Property | Value |
|----------|-------|
| Parameters | 1.06M |
| Channels per level | nc = [16, 32, 64, 128] |
| Residual blocks per level | nb = 2 |
| Activation | ReLU (act_mode='R') |
| Bias | False |
| Input / Output | 3-channel RGB [0, 1] |
| Spatial constraint | Height and width divisible by 8 |
| Checkpoint size | 4 MB |
| Inference speed (RTX 3060) | 56 fps FP16, 57 fps INT8 mixed (C++ pipeline) |

The student is 30x smaller than the teacher and runs at real-time speeds. It is the deployment model. The channel widths are reduced 4x at each level and the number of residual blocks is halved, but the overall U-Net topology is identical.

### Architecture Diagram

```
Input (B, 3, H, W)
  |
  v
m_head: Conv2d(3, nc[0], 3x3)
  |
  v  [x1: (B, nc[0], H, W)]
m_down1: nb x ResBlock(nc[0]) + Downsample
  |
  v  [x2: (B, nc[1], H/2, W/2)]
m_down2: nb x ResBlock(nc[1]) + Downsample
  |
  v  [x3: (B, nc[2], H/4, W/4)]
m_down3: nb x ResBlock(nc[2]) + Downsample
  |
  v  [x4: (B, nc[3], H/8, W/8)]
m_body: nb x ResBlock(nc[3])  (bottleneck)
  |
  v
m_up3: Upsample + nb x ResBlock(nc[2])   <-- skip add x4
  |
  v
m_up2: Upsample + nb x ResBlock(nc[1])   <-- skip add x3
  |
  v
m_up1: Upsample + nb x ResBlock(nc[0])   <-- skip add x2
  |
  v
m_tail: Conv2d(nc[0], 3, 3x3)            <-- skip add x1
  |
  v
Output (B, 3, H, W) -- residual, added to input
```

Each ResBlock is `Conv 3x3 -> ReLU -> Conv 3x3` with a residual (identity) skip connection. Downsampling uses strided convolution; upsampling uses transposed convolution. The model learns a residual correction that is added to the input, so it only needs to learn the difference between compressed and clean.

---

## Training Approach

Training uses a two-phase **teacher-student distillation** approach.

### Phase 1: Train the Teacher

The teacher model learns directly from paired training data (compressed originals as input, SCUNet GAN + USM targets as ground truth). It is trained with:

- **Charbonnier pixel loss** for overall fidelity
- **DISTS perceptual loss** (weight 0.15) for structural and textural quality
- **Prodigy optimizer** with EMA weight averaging
- 13K+ iterations on Modal L40S (48GB VRAM)

The teacher reaches 53.27 dB PSNR -- a quality level that exceeds the original Bluray source material in sharpness.

### Phase 2: Distill to Student

The student learns to replicate the teacher's behavior through two mechanisms:

**Output matching.** The student sees the same compressed input frames as the teacher. The teacher runs inference (frozen, no gradients) to produce a target output for each input. The student is trained to match this output via Charbonnier pixel loss. This is "online distillation" -- the teacher generates targets live during training rather than from pre-computed files on disk.

**Feature matching.** In addition to matching the final output, the student is trained to match the teacher's intermediate encoder representations. At each of the 4 encoder levels, the student's feature maps are projected through a learned 1x1 adapter convolution to match the teacher's channel dimensions, and the L1 distance between the projected student features and the (detached) teacher features is minimized.

The adapter convolutions bridge the channel width gap between student and teacher:

```
Level 0:  student 16ch  --[1x1 Conv]--> teacher 64ch,  L1 loss
Level 1:  student 32ch  --[1x1 Conv]--> teacher 128ch, L1 loss
Level 2:  student 64ch  --[1x1 Conv]--> teacher 256ch, L1 loss
Level 3:  student 128ch --[1x1 Conv]--> teacher 512ch, L1 loss
```

The adapters are initialized with small-scale Kaiming normal weights (0.1x) to avoid disrupting the student's pre-existing representations at the start of training. They are trainable parameters included in the optimizer alongside the student model weights, but they are discarded after training -- only the student model weights are used for inference.

### Optimizer: Prodigy

Both teacher and student use the [Prodigy](https://github.com/konstmish/prodigy) optimizer, which automatically estimates the optimal learning rate during training. Key settings:

- `lr=1.0` (Prodigy treats this as a multiplier on its auto-tuned D estimate)
- `safeguard_warmup=True` (prevents the scheduler from corrupting the D estimate)
- `use_bias_correction=True` (recommended for fine-tuning from pretrained weights)
- Cosine annealing schedule with no warmup (warmup confuses Prodigy's D estimation)

Prodigy eliminates the need for learning rate sweeps, which is especially valuable when the loss landscape changes between runs (e.g., adding new data, changing loss weights).

### EMA Weights

Exponential moving average (decay=0.999) of model weights is maintained during training. The EMA weights are smoother and often generalize better than the raw training weights. The final checkpoint (`final.pth`) contains the EMA-averaged weights.

### Full-Frame Fine-Tuning

After initial training on 256x256 random crops, a final fine-tuning pass uses full 1920x1080 frames with batch size 8. This teaches the model how dark edges, letterboxing, and brightness transitions behave across the full spatial extent of the frame. Without this step, the model produces artifacts in dark areas near frame boundaries.

---

## Training Data

### Core Principle

The training uses **original compressed video frames** as input and **SCUNet GAN + USM(1.0) processed frames** as target. The HEVC compression itself is the degradation -- we do not synthesize degraded inputs. The model learns the mapping: `compressed original -> clean/sharp target`.

An earlier experiment with synthetic degradation (adding noise and blur to already-compressed inputs) was tried and failed to converge. The distribution of real HEVC compression artifacts is complex and varied enough that the original frames provide sufficient training signal on their own.

### Target Generation

Training targets are generated by [SCUNet GAN](https://github.com/cszn/SCUNet), a Swin-Conv-UNet denoiser trained with perceptual and adversarial losses. Unlike PSNR-optimized denoisers that soften detail, SCUNet GAN denoises while preserving and enhancing texture. A light unsharp mask (strength=1.0, sigma=1.5) is applied after SCUNet GAN to push detail recovery slightly further.

The choice of SCUNet GAN over other denoisers was deliberate:

- **DRUNet** (PSNR-optimized): removes noise but softens detail. Not suitable as a training target.
- **HYPIR** (diffusion-based): adds AI-generated texture artifacts, especially on faces. Rejected after evaluation ([findings](research/hypir/findings.md)).
- **Upscale-A-Video** (diffusion video SR): 100-1000x too slow for generating 7K training targets. Rejected ([findings](research/upscale-a-video/findings.md)).
- **SCUNet GAN** (adversarial): denoises while preserving texture. No sigma tuning needed. The right balance of quality and practicality.

### Data Sources

Approximately 7,000 paired frames extracted from diverse 1080p content, sampled proportionally at 1 frame per 500 source frames:

| Source | Content Type | Samples | Resolution |
|--------|-------------|---------|------------|
| Firefly S01 | Live action sci-fi (2002) | ~1,876 | 1920x1080 |
| The Expanse S02 | Live action sci-fi | ~1,635 | 1920x1080 |
| One Piece S01 | Anime (Netflix) | ~1,306 | 1920x1080 |
| Squid Game S02 | Live action drama | ~1,237 | 1920x960 |
| Dune Part Two | Film (2024) | ~476 | 1920x802 |
| Foundation S03 | Live action sci-fi | ~426 | 1920x800 |
| **Total** | | **~6,956** | |

The mix of live action, anime, and film at different aspect ratios (1080, 960, 802, 800 heights) ensures the model generalizes across content types and letterboxing configurations. The 90/10 train/val split is stratified by source with a fixed seed.

### Data Pipeline

The data build process is staged and resumable, implemented in `tools/build_training_data.py`:

1. **Extract originals** (`--extract-only`): Probe source videos for frame count, sample proportionally, extract raw PNGs with NVDEC hardware decode to `data/originals/`, compute per-frame noise metrics.
2. **Denoise** (`--denoise`): Run every original through SCUNet GAN + USM(1.0). Split 90/10 train/val (stratified by source). Save targets to `data/training/{train,val}/target/`.
3. **Build inputs** (`--build-inputs`): Copy original compressed frames as-is to `data/training/{train,val}/input/`.

---

## Loss Functions

### Charbonnier Loss (Pixel)

```
L_char = mean(sqrt((pred - target)^2 + eps^2))
```

A smooth approximation to L1 loss. Unlike MSE (L2), Charbonnier does not disproportionately penalize large errors, which prevents the model from averaging over sharp edges to minimize squared error. Unlike hard L1, the sqrt is differentiable everywhere, providing stable gradients near zero error. Used in both teacher and student training.

### DISTS Perceptual Loss

[Deep Image Structure and Texture Similarity](https://github.com/dingkeyan93/DISTS) (Ding et al., 2020). DISTS uses a VGG16 backbone to extract multi-scale feature representations, then computes separate structure (mean luminance) and texture (variance/covariance) similarity scores. Unlike raw VGG feature distance, DISTS is specifically calibrated for image quality assessment of structural distortions -- exactly the kind of damage HEVC compression introduces.

Implementation details:
- The VGG16 backbone is frozen (always in eval mode). Its parameters are not added to the optimizer.
- Gradients flow through the backbone to the denoiser via the prediction features; target features are computed with `torch.no_grad()` for a 26% speedup.
- Must run in FP32 -- the VGG backbone is numerically unstable in FP16. Inputs are cast to float32 before the DISTS forward pass.

Used in teacher training (weight 0.15) and optionally in student training (weight 0.05).

### Focal Frequency Loss

```
L_fft = mean(weight * diff)
  where diff = |abs(FFT(pred)) - abs(FFT(target))|
  and   weight = diff^alpha
```

Operates in the frequency domain via 2D real FFT. Computes L1 distance per frequency component, then applies focal weighting that upweights frequencies where the model struggles most. This preserves high-frequency detail (edges, fine texture) that pixel-level losses tend to smooth away. Available in the codebase but not used in the current best models -- Charbonnier + DISTS proved sufficient.

### Feature Matching Loss

For student distillation only. Compares intermediate encoder features between student and teacher at each of the 4 U-Net levels. Learned 1x1 adapter convolutions project student features (narrow channels) to teacher feature dimensions (wide channels), then L1 distance is computed:

```
L_feat = (1/4) * sum_k L1(adapter_k(student_feat_k), teacher_feat_k)
```

The feature matching loss ensures the student learns similar internal representations to the teacher, not just similar outputs. This is particularly important for a 30x compression ratio where the student's narrow channels cannot represent the same feature space as the teacher without explicit alignment pressure.

Weight: 0.1 in student training.

---

## Key Finding: Generalization Beyond Artifact Removal

The model generalizes significantly beyond simple compression artifact removal.

Training on mixed content (different shows, different compression levels, anime and live action) with SCUNet GAN + USM targets produces a model that simultaneously denoises AND sharpens. The teacher model reaches 107% of the original source sharpness -- it produces output that is measurably sharper than the Bluray source material it was trained on.

This was not an expected outcome. The original goal was pure artifact removal (banding, blocking, mosquito noise). The sharpening emerges because the SCUNet GAN + USM targets contain detail that the original HEVC compression discarded, and the model learns to reconstruct that detail from the compressed input. The model is effectively learning to invert part of the compression process and recover information that was quantized away.

The student preserves most of this generalization capability despite being 30x smaller, achieving 49.98 dB PSNR against the same targets.

---

## Deployment Pipeline

The path from trained model weights to real-time video processing involves several conversion steps.

### Step 1: ONNX Export

The student model is exported to ONNX format using `tools/export_onnx.py`:

```
PyTorch .pth --> ONNX (opset 17, FP16 weights, dynamic spatial dims)
```

Critical detail: PyTorch 2.11 defaults `torch.onnx.export()` to `dynamo=True`, which emits opset 20 IR that TensorRT 10.16 miscompiles (produces 14.5 dB output -- effectively garbage). The export script forces `dynamo=False` to use the legacy TorchScript exporter at opset 17, which works correctly.

The ONNX file uses dynamic height and width dimensions, so a single export works for any resolution. FP16 weights are used so that TensorRT engines automatically inherit FP16 I/O without additional format overrides.

### Step 2: TensorRT Engine Build

The ONNX model is compiled to a TensorRT engine specific to the target GPU and resolution:

```
ONNX --> trtexec --> .engine (GPU-specific, resolution-specific)
```

The engine is built with `--fp16 --useCudaGraph` for maximum throughput. FP16 inference on the student model produces 68.0 dB fidelity vs. FP32 PyTorch -- visually identical. Engine build takes approximately 2 minutes and only needs to happen once per GPU/resolution combination.

### Step 3: C++ Pipeline

The production pipeline (`pipeline_cpp/`) is a C++ application that chains three GPU-accelerated stages with zero CPU round-trips:

```
NVDEC (hardware decode) --> NV12-to-RGB CUDA kernel
  --> TensorRT inference (CUDA graph captured)
  --> RGB-to-NV12 CUDA kernel --> NVENC (hardware encode)
```

Each stage runs on a separate CUDA stream with async I/O:

- **NVDEC** decodes at 500+ fps -- never the bottleneck.
- **TensorRT** inference runs at ~63 fps (15.8ms/frame GPU compute) -- this is the pipeline bottleneck.
- **NVENC** encodes at 200+ fps -- never the bottleneck.

The pipeline achieves 55-57 fps end-to-end with approximately 0.6ms/frame of sync and CPU overhead. Output is 10-bit HEVC in MKV containers with full audio and subtitle passthrough. A 44-minute episode at 1080p processes in about 18 minutes.

### Alternative Pipelines

| Pipeline | FPS | Implementation |
|----------|-----|---------------|
| C++ (NVDEC + TRT + NVENC) | 55-57 | `pipeline_cpp/` |
| NVEncC + VapourSynth | 39 | `remaster/encode_nvencc.py` |
| Python + torch.compile | 24 | `pipelines/remaster.py` |
| VapourSynth + ffmpeg pipe | 20 | `remaster/encode.py` |

The Python pipeline exists for environments where TensorRT and a C++ toolchain are not available. It uses `torch.compile` with the inductor backend for optimized inference. The GIL limits Python pipelines to ~24 fps because PyNvVideoCodec's C extensions hold the GIL during NVDEC/NVENC calls, serializing all three pipeline stages on a single thread.

### Edge-Replicate Padding

All deployment paths handle sub-1080p content (e.g., 1920x802 letterboxed film) by padding with edge-replication rather than zero-fill. Zero padding introduces black borders at convolution boundaries that produce visible artifacts. Edge replication extends the frame content naturally. This applies to the C++ color conversion kernels, the Python test suite, and the INT8 calibrator.

---

## INT8 Quantization

### FP16 vs INT8

TensorRT FP16 is the default deployment precision. It achieves 68.0 dB fidelity vs. FP32 PyTorch (visually indistinguishable) at 50.7 fps inference-only, 55-57 fps in the full pipeline.

INT8 quantization was investigated to push throughput further. The results are mixed:

| Precision | Inference FPS | Pipeline FPS | Quality (dB vs FP32) |
|-----------|--------------|-------------|---------------------|
| FP16 | 50.7 | 55-57 | 68.0 |
| INT8 mixed | 51.4 | 57 | 67.2 |
| INT8 pure | 39 | -- | 26.1 |

### The Skip-Connection Problem

Pure INT8 (all layers quantized) produces catastrophically low quality (26 dB). The root cause is the skip-connection Add operations in the U-Net. The encoder path and decoder path produce feature maps with different value distributions at their addition points. INT8's 256 quantization levels cannot represent the fine residuals that the model relies on for its correction signal.

The current workaround excludes Add operations from INT8 quantization (`op_types_to_exclude=["Add"]`), forcing them to run in FP16. Head and tail convolutions and transposed convolutions are also kept in FP16. This "mixed precision" approach preserves quality (67.2 dB) but limits the speedup to ~3% over pure FP16, because the computation-heavy layers still run in higher precision.

### Engine Builders

Two engine builders exist for historical reasons:

- **`tools/build_engine.py`** -- Uses NVIDIA ModelOpt with Q/DQ nodes and strongly-typed TRT (forward-compatible with TRT 11). Best for FP16 engines. INT8 mixed-precision OOMs on 6GB VRAM during ONNX Runtime calibration at 1080p.
- **`tools/build_int8_engine.py`** -- Uses the legacy `IInt8EntropyCalibrator2` API with weak-typed TRT. Handles INT8 mixed-precision within 6GB VRAM by calibrating with batches of real frames and manually specifying FP16 layer exclusions.

### QAT (Planned)

The INT8 results above used post-training quantization (PTQ). Quantization-aware training (QAT) is planned but untested. The hypothesis is that QAT could teach the model INT8-friendly representations where the skip-connection additions remain accurate at 8-bit precision. The training script already has QAT infrastructure (`prepare_qat()` in `training/train.py`) using PyTorch's `torch.ao.quantization` with symmetric per-channel FakeQuantize modules that match TensorRT's INT8 scheme.

---

## Experiments and Research History

The current DRUNet teacher-student approach emerged after extensive experimentation. This section summarizes what was tried, what worked, and what was abandoned.

### NAFNet (Abandoned)

The project originally used [NAFNet](https://github.com/megvii-research/NAFNet), a non-attention FFT-based restoration network. NAFNet achieved good quality but had several deployment problems:

- LayerNorm2d required an FP32 cast workaround for FP16 inference.
- `torch.compile` worked well (78 fps model inference) but TensorRT was catastrophically slow (4-6 fps) due to poor fusion of SimpleGate and Simple Channel Attention operations.
- The Python GIL limited the full pipeline to ~7 fps regardless of model speed.

NAFNet was replaced by DRUNet when it became clear that a TRT-friendly architecture was essential for real-time deployment via the C++ pipeline.

### PlainNet and UNetDenoise (Explored)

Simple architectures (plain Conv stacks and small U-Nets without residual blocks) were tested as potential student models. They converged but could not match DRUNet's quality-to-speed ratio. PlainDenoise required more parameters to reach equivalent PSNR, and UNetDenoise without DRUNet's residual blocks trained less stably.

### NeRV: Neural Representations for Videos (Concluded)

[NeRV](https://github.com/haochen-rye/NeRV) and its variants (HNeRV, FFNeRV, Boosting-NeRV) encode an entire video as a neural network's weights. The network's spectral bias causes it to learn clean signal before noise, providing implicit denoising without paired training data.

**Finding:** NeRV memorizes individual videos but does not generalize across content. Each video requires training its own network from scratch, making it impractical as a general-purpose denoiser or training target generator. The temporal modeling is genuine but the per-video cost is prohibitive. See [research/nerv-denoising/](research/nerv-denoising/) for the full research log and scaling analysis.

### HYPIR: Diffusion Restoration (Rejected)

[HYPIR](https://arxiv.org/abs/2401.00110) is a single-forward-pass diffusion restoration model using Stable Diffusion 2.1 with LoRA weights. It was evaluated as a potential source of higher-quality training targets.

**Finding:** Even at the lowest tested timestep (t=25), HYPIR distorts facial features and adds an "AI-generated" texture quality to skin, hair, and fine detail. The diffusion prior has a strong bias toward its web-image training distribution, which warps faces and invents textures that were not in the source. These artifacts survive downscaling and would propagate into the student model during distillation. See [research/hypir/findings.md](research/hypir/findings.md).

### Upscale-A-Video: Diffusion Video SR (Rejected)

A diffusion-based video super-resolution model evaluated for training target generation.

**Finding:** Produces temporally consistent output with genuine quality improvements, but the per-frame cost makes it impractical. Processing 7K training frames at 1080p would cost approximately $2,100 and require ~850 GPU-hours on A100s. For comparison, SCUNet GAN processes the same frames in ~1 hour for ~$1. The temporal attention is valuable, but not at 1000x the cost of simpler alternatives. See [research/upscale-a-video/findings.md](research/upscale-a-video/findings.md).

### RAFT Optical Flow Alignment (Deprioritized)

RAFT optical flow was explored for temporal alignment between adjacent frames to improve temporal consistency of the single-frame model. The approach was deprioritized in favor of the simpler recurrent temporal context architecture, which lets the model learn implicit alignment through concatenated frames rather than explicit flow computation. See [research/raft-alignment/](research/raft-alignment/).

---

## Future Directions

### Recurrent Temporal Context

The most impactful planned improvement. The student model's input is widened from 3 to 9 channels: `[prev_cleaned | current_noisy | next_noisy]`. This gives the model two noisy observations of each pixel (from adjacent frames) and a feedback signal from its own previous output, enabling temporal consistency without optical flow computation.

The architecture change is minimal: only the head convolution changes from `Conv2d(3, 16, 3x3)` to `Conv2d(9, 16, 3x3)`, adding 864 parameters (from 1,063,776 to 1,064,640). All other weights transfer directly from the existing single-frame model, and the head weights for the current-frame channels (3-5) are initialized from the pretrained head. Speed impact is negligible.

A detailed PRD with file-by-file implementation plan, progressive unfreezing strategy, and cost estimates is at [research/temporal-context/prd.md](research/temporal-context/prd.md). Expected gain: +0.3-0.8 dB PSNR plus significantly reduced temporal flicker.

### Quantization-Aware Training

Train with FakeQuantize modules inserted at every Conv2d to learn INT8-friendly weight and activation distributions. Could enable pure INT8 on all layers including skip-connection Adds, potentially unlocking meaningful speedups beyond the current 3% INT8-mixed improvement. The training infrastructure exists in `training/train.py` (`prepare_qat()`) but has not been tested end-to-end.

### dynamo=True ONNX Export

The current ONNX export uses the legacy TorchScript exporter due to TRT miscompilation with dynamo-generated opset 20 IR. TensorRT's opset 20 support continues to improve, and a future version may handle the dynamo output correctly, eliminating the dependency on the deprecated TorchScript export path.
