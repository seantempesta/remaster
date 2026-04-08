# NAFNet Quantization Research

Research into quantization options for NAFNet-width64 (~116M params) to enable local inference on RTX 3060 (6GB VRAM) at 1080p, ideally batch 2+.

**Current baseline:** fp16 inference on cloud H100 at 27.9 fps via torch.compile + CUDA graphs.

## Summary: Approaches Ranked by Effort vs Reward

| Rank | Approach | Effort | Expected Speedup | PSNR Impact | Retraining? |
|------|----------|--------|-------------------|-------------|-------------|
| 1 | TensorRT FP16+INT8 | Medium | 1.5-2x over fp16 | <0.1 dB | No |
| 2 | TorchAO INT8 weight-only | Low | 1.2-1.4x | <0.1 dB | No |
| 3 | TorchAO INT8 dynamic (W8A8) | Low-Med | 1.3-1.5x | 0.1-0.3 dB | No |
| 4 | TensorRT INT8 (full) | Medium | 1.6-2x over fp16 | 0.2-0.6 dB | No |
| 5 | Mixed-precision PTQ | Medium | 1.3-1.6x | 0.1-0.2 dB | No |
| 6 | QAT INT8 | High | 1.5-2x | <0.1 dB | Yes |
| 7 | INT4 weight-only | Low-Med | 1.5-2x mem savings | 0.5-2.0 dB | Likely |

**Recommendation:** Start with TorchAO INT8 weight-only (approach 2) as it is trivial to implement and verify. If memory savings are sufficient for batch 2, stop there. If more speed is needed, move to TensorRT INT8 (approach 1/4).

---

## 1. VRAM Budget Analysis

### Current fp16 baseline (estimated)

| Component | fp16 Size |
|-----------|-----------|
| Model weights (116M params) | ~232 MB |
| Activations (1080p, batch 1) | ~400-600 MB |
| Activations (1080p, batch 2) | ~800-1200 MB |
| PyTorch/CUDA overhead | ~300-500 MB |
| torch.compile cache | ~200-400 MB |
| **Total (batch 1)** | **~1.1-1.7 GB** |
| **Total (batch 2)** | **~1.5-2.3 GB** |

### With INT8 weights

| Component | INT8 Size |
|-----------|-----------|
| Model weights (116M params) | ~116 MB |
| Activations (1080p, batch 1, fp16) | ~400-600 MB |
| Activations (1080p, batch 2, fp16) | ~800-1200 MB |
| Activations (1080p, batch 4, fp16) | ~1600-2400 MB |
| PyTorch/CUDA overhead | ~300-500 MB |
| **Total (batch 2)** | **~1.4-2.0 GB** |
| **Total (batch 4)** | **~2.2-3.2 GB** |

### Verdict

Even at fp16, batch 2 at 1080p should fit in 6GB. The main benefit of INT8 quantization is not fitting in VRAM (we already fit) but rather:
- **Speed:** INT8 tensor core ops on RTX 3060 (101.9 TOPS dense INT8 vs 12.7 TFLOPS fp16) -- an 8x theoretical throughput advantage
- **Memory bandwidth:** INT8 weights are half the size of fp16, reducing memory-bound bottleneck
- **Higher batch sizes:** Batch 4 becomes feasible at INT8, which could improve GPU utilization

The RTX 3060 (Ampere, compute capability 8.6) has 3rd-gen tensor cores supporting INT8. It does NOT support FP8 (Ada/Hopper only).

---

## 2. Post-Training Quantization (PTQ)

### 2a. TorchAO INT8 Weight-Only

The simplest approach. Quantize weights to INT8, keep activations in fp16. Dequantization happens on-the-fly during matmul/conv.

**Expected quality impact:** <0.1 dB PSNR drop. Weight-only quantization is very safe for CNNs because conv weights are well-distributed and the activation precision is preserved.

**Expected speedup:** 1.2-1.4x from reduced memory bandwidth. The speedup is modest because compute is still done in fp16 -- only the weight loading is faster.

**torch.compile compatibility:** Full support. TorchAO is designed to work with torch.compile and Inductor.

**Implementation:**

```python
import torch
import torchao
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# Load model normally
model = load_nafnet(checkpoint_path, device="cuda", fp16=True)

# Apply INT8 weight-only quantization
quantize_(model, Int8WeightOnlyConfig())

# torch.compile works on top of quantized model
model = torch.compile(model, mode="reduce-overhead")

# Warmup and run as before
```

**Caveats:**
- TorchAO's `Int8WeightOnlyConfig` targets linear layers by default. For Conv2d, we may need to check if TorchAO handles conv layers or if we need custom quantization.
- The `quantize_` function modifies the model in-place.
- Install: `pip install torchao` (requires PyTorch 2.4+).

### 2b. TorchAO INT8 Dynamic Activation + INT8 Weight (W8A8)

Quantize both weights and activations to INT8. The activation quantization is dynamic (computed per-batch).

**Expected quality impact:** 0.1-0.3 dB PSNR drop. Activation quantization adds more error, but for well-behaved CNN activations this is usually minor. The LayerNorm2d layers (which cast to fp32 internally) should be kept in higher precision.

**Expected speedup:** 1.3-1.5x. Both weight loading and compute benefit from INT8.

**Implementation:**

```python
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

model = load_nafnet(checkpoint_path, device="cuda", fp16=True)
quantize_(model, Int8DynamicActivationInt8WeightConfig())
model = torch.compile(model, mode="reduce-overhead")
```

### 2c. PyTorch Native Static Quantization

The older `torch.ao.quantization` API with FX graph mode. More complex but gives fine-grained control.

**Not recommended** for this use case: FX quantization is primarily optimized for CPU inference (x86 FBGEMM backend). GPU quantization through TorchAO or TensorRT is better for our CUDA target.

### Quality expectations from literature

Research on image denoising quantization (see Sources) shows:

- **INT8 PTQ on DnCNN:** ~0.6 dB PSNR drop with naive PTQ, recoverable to <0.1 dB with calibration ([ETASR paper](https://mail.etasr.com/index.php/ETASR/article/download/15428/6155))
- **INT8 PTQ on image SR models:** 0.1-0.5 dB drop depending on architecture, with CNN architectures more robust than transformers ([2DQuant, NeurIPS 2024](https://arxiv.org/abs/2406.06649))
- **INT4 on SR models:** 1-4 dB drop without specialized techniques, recoverable to 0.5-1.0 dB with 2DQuant-style calibration
- **General rule:** Pure CNN denoisers (like NAFNet) are more quantization-friendly than transformer-based models because their weight and activation distributions are smoother

---

## 3. TensorRT INT8

### Why TensorRT is ideal for NAFNet

NAFNet is a pure CNN with no dynamic control flow, no attention mechanisms, and static input shapes (after padding). This is the best-case scenario for TensorRT:
- All ops (Conv2d, LayerNorm, ReLU-free SimpleGate, AdaptiveAvgPool2d, PixelShuffle) have native TensorRT support
- Static shapes enable aggressive kernel fusion and memory planning
- The existing `swap_layernorm_for_export()` function in `nafnet_arch.py` already handles the custom LayerNorm2d export

### Expected performance

TensorRT benchmarks on ResNet-50 show **60% speedup** going from FP16 to INT8 (507 qps to 812 qps, [NVIDIA docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)). For NAFNet at 1080p:
- FP16 TensorRT alone (no INT8) should give 1.3-1.5x over torch.compile fp16 (better kernel selection, layer fusion)
- INT8 TensorRT should give 1.6-2x over torch.compile fp16
- Combined with batch 2, we could see 2-3x total throughput improvement

### Implementation steps

```python
# Step 1: Export to ONNX
import torch
from lib.nafnet_arch import NAFNet, swap_layernorm_for_export

model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
               enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2])
# Load weights...
model.eval()
model = swap_layernorm_for_export(model)  # TRT-safe LayerNorm
model.half().cuda()

# Fixed input shape (1080p padded to multiple of 16)
dummy = torch.randn(1, 3, 1088, 1920, device="cuda", dtype=torch.float16)
torch.onnx.export(model, dummy, "nafnet_w64.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=17,
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
```

```bash
# Step 2: Convert to TensorRT with INT8 calibration
# Option A: Using trtexec with random calibration (quick test)
trtexec --onnx=nafnet_w64.onnx \
        --int8 --fp16 \
        --minShapes=input:1x3x1088x1920 \
        --optShapes=input:2x3x1088x1920 \
        --maxShapes=input:4x3x1088x1920 \
        --saveEngine=nafnet_w64_int8.engine

# Option B: Using NVIDIA ModelOptimizer for proper calibration
pip3 install --extra-index-url https://pypi.nvidia.com nvidia-modelopt
```

```python
# Step 3: Run inference with TensorRT engine
import tensorrt as trt
import pycuda.driver as cuda

# Load engine, allocate buffers, run inference...
# (or use torch-tensorrt for easier integration)
```

### Alternative: Torch-TensorRT (simpler integration)

```python
import torch_tensorrt

model = load_nafnet(checkpoint_path, device="cuda", fp16=True)
model = swap_layernorm_for_export(model)

trt_model = torch_tensorrt.compile(model,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 3, 1088, 1920],
        opt_shape=[2, 3, 1088, 1920],
        max_shape=[4, 3, 1088, 1920],
        dtype=torch.float16,
    )],
    enabled_precisions={torch.float16, torch.int8},
    calibrator=calibrator,  # needs calibration data
)
```

### Calibration data

INT8 calibration requires ~100-500 representative frames. We can extract these from the same video content we process:

```python
# Extract calibration frames from a sample episode
import av
container = av.open("data/clips/sample.mkv")
frames = []
for i, frame in enumerate(container.decode(video=0)):
    if i % 100 == 0:  # every 100th frame
        arr = frame.to_ndarray(format='rgb24')
        t = torch.from_numpy(arr).permute(2,0,1).half() / 255.0
        frames.append(t)
    if len(frames) >= 200:
        break
calib_data = torch.stack(frames)  # [200, 3, 1080, 1920]
```

### Alignment note

TensorRT achieves optimal INT8 performance when channel dimensions align to 32-element multiples. NAFNet-width64 uses widths of 64, 128, 256, 512 -- all multiples of 32, so this is already optimal.

---

## 4. Mixed-Precision Quantization

### Sensitive layers in NAFNet

Based on general CNN quantization research and NAFNet's architecture, the most sensitive layers are:

1. **`intro` conv (3 -> 64):** First layer, directly processes pixel values. Small channel count (3 input channels) means less redundancy for quantization.
2. **`ending` conv (64 -> 3):** Last layer, directly produces output pixels. Quantization error here maps directly to pixel error.
3. **LayerNorm2d:** Already handles precision internally (casts to fp32). Should stay in fp16 or fp32.
4. **SimpleGate (chunk + multiply):** The element-wise multiply after chunking can amplify quantization error. Keep activations in fp16 here.

### Recommended mixed-precision strategy

- **INT8:** All Conv2d in encoder/decoder NAFBlocks (conv1, conv2, conv3, conv4, conv5), downsampling convs, upsampling convs. These are >95% of compute.
- **FP16:** intro conv, ending conv, LayerNorm2d, SCA (channel attention) convs, skip connections.

This is straightforward to implement in TensorRT by marking specific layers as fp16 in the ONNX graph, or using per-layer precision in Torch-TensorRT.

### Expected impact

Mixed precision typically recovers 50-80% of the PSNR drop from full INT8, at the cost of 10-15% of the INT8 speedup. For NAFNet, expect:
- PSNR drop: 0.1-0.2 dB (vs 0.2-0.6 dB for full INT8)
- Speedup: 1.4-1.7x over fp16 (vs 1.6-2x for full INT8)

---

## 5. Quantization-Aware Training (QAT)

### When to use QAT

QAT is only needed if PTQ quality is unacceptable. For INT8, PTQ on CNNs typically works well enough. QAT becomes important for:
- INT4 quantization (where PTQ drops >1 dB)
- Models with unusual activation distributions
- When every 0.1 dB matters

### Implementation with PyTorch

```python
import torch.ao.quantization as quant

# Insert fake quantize nodes
model.qconfig = quant.get_default_qat_qconfig("x86")  # or custom
model_prepared = quant.prepare_qat(model)

# Fine-tune for a few thousand iterations with quantization simulation
for batch in train_loader:
    loss = criterion(model_prepared(batch), target)
    loss.backward()
    optimizer.step()

# Convert to quantized model
model_quantized = quant.convert(model_prepared)
```

### Cost

- Requires the training pipeline and data (we have this in `training/train_nafnet.py`)
- Typically needs 10-20% of original training iterations (so ~1-5K iters for our model)
- Can be combined with distillation loss from the fp16 teacher
- Adds significant complexity for marginal gain over PTQ at INT8

### Recommendation

Skip QAT unless PTQ INT8 shows >0.3 dB PSNR drop on our test data. For our denoising use case, visual quality matters more than PSNR -- a 0.3 dB drop is unlikely to be visible.

---

## 6. INT4 and Sub-8-bit Quantization

### INT4 weight-only

GPTQ and AWQ are designed for transformer/LLM weight matrices, not CNN conv kernels. However, basic INT4 weight-only quantization can be applied to CNNs:

- TorchAO provides `Int4WeightOnlyConfig()` which uses group-wise quantization
- 4x weight compression vs fp16 (model goes from ~232 MB to ~58 MB)
- Quality impact is significant: expect 0.5-2.0 dB PSNR drop without QAT

Research from NeurIPS 2020 ([Post-training 4-bit quantization of convolutional networks](https://dl.acm.org/doi/10.5555/3454287.3455001)) shows that 4-bit PTQ on CNNs achieves accuracy "just a few percent less than baseline" for classification, but image restoration is more sensitive than classification.

### INT4 for NAFNet: not recommended yet

- The quality bar for denoising is higher than classification (we care about pixel-level fidelity)
- INT8 already provides sufficient memory savings for batch 2-4 on RTX 3060
- If INT4 is needed later, QAT would be required to maintain quality

### FP8

Not applicable: RTX 3060 (Ampere, compute capability 8.6) does not have FP8 tensor core support. FP8 requires Ada Lovelace (cc 8.9) or Hopper (cc 9.0).

---

## 7. Compatibility with torch.compile and CUDA Graphs

### TorchAO + torch.compile

Fully compatible. TorchAO is designed as a torch.compile-native quantization library:
- `quantize_()` inserts quantized ops that Inductor knows how to lower
- `torch.compile(mode="reduce-overhead")` works on quantized models
- CUDA graphs work as long as input shapes are static (which they are in our pipeline, since we pad to fixed dimensions)

### TensorRT + CUDA graphs

TensorRT manages its own CUDA streams and graph recording internally. When using TensorRT, you don't use PyTorch's CUDA graph API -- TensorRT handles optimization at a lower level.

### Torch-TensorRT + torch.compile

Torch-TensorRT integrates as a torch.compile backend:
```python
model = torch.compile(model, backend="torch_tensorrt",
                      options={"enabled_precisions": {torch.float16, torch.int8}})
```
This is the cleanest integration path but requires torch-tensorrt to be installed.

---

## 8. Concrete Implementation Plan

### Phase 1: Quick wins (1-2 hours)

1. Install TorchAO: `pip install torchao`
2. Apply `Int8WeightOnlyConfig()` to NAFNet
3. Verify PSNR on test clips (should be <0.1 dB drop)
4. Benchmark fps at batch 1 and batch 2 on RTX 3060
5. If batch 2 fits and speed is acceptable, done

### Phase 2: TensorRT (half day)

1. Export NAFNet to ONNX using `swap_layernorm_for_export()`
2. Convert to TensorRT engine with INT8 calibration using trtexec
3. Write a TensorRT inference wrapper (replacing PyTorch model forward)
4. Benchmark against torch.compile fp16 baseline
5. If >1.5x speedup, integrate into `pipelines/denoise_nafnet.py`

### Phase 3: Mixed precision tuning (if needed)

1. Profile per-layer sensitivity by quantizing one layer at a time
2. Keep intro/ending conv + LayerNorm in fp16
3. Re-calibrate and benchmark
4. Only if Phase 2 INT8 quality is insufficient

### Phase 4: QAT (only if PTQ fails)

1. Add fake quantize nodes to NAFNet training loop
2. Fine-tune for 2-5K iterations on existing training data
3. Export and convert to TensorRT INT8
4. Compare quality vs PTQ

---

## Sources

### Papers
- [2DQuant: Low-bit Post-Training Quantization for Image Super-Resolution (NeurIPS 2024)](https://arxiv.org/abs/2406.06649)
- [Toward Accurate Post-Training Quantization for Image Super Resolution (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf)
- [Image Denoising Meets Quantization: Exploring the Effects of Post-Training Quantization (IEEE 2024)](https://ieeexplore.ieee.org/document/11137609/)
- [Outlier-Aware Post-Training Quantization for Image Super-Resolution (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Outlier-Aware_Post-Training_Quantization_for_Image_Super-Resolution_ICCV_2025_paper.pdf)
- [Post-training 4-bit quantization of convolutional networks (NeurIPS 2020)](https://dl.acm.org/doi/10.5555/3454287.3455001)
- [A Lightweight Denoising CNN with INT8 Quantization](https://mail.etasr.com/index.php/ETASR/article/download/15428/6155)
- [Convolutional neural network mixed-precision quantization method considering layer sensitivity](https://www.sciopen.com/article/10.11887/j.issn.1001-2486.25010015)
- [Robust ternary quantization for lightweight image denoising](https://link.springer.com/article/10.1007/s11760-025-04992-x)

### Tools and Documentation
- [TorchAO: PyTorch native quantization (GitHub)](https://github.com/pytorch/ao)
- [TorchAO GPU Quantization Tutorial](https://docs.pytorch.org/tutorials/unstable/gpu_quantization_torchao_tutorial.html)
- [TorchAO Quantized Inference Docs](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- [NVIDIA TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [NVIDIA TensorRT Model Optimizer](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)
- [TensorRT Quantization (INT8/FP8/FP4) with Torch-TensorRT](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html)
- [NVIDIA TensorRT for RTX](https://developer.nvidia.com/blog/nvidia-tensorrt-for-rtx-introduces-an-optimized-inference-ai-library-on-windows/)

### Hardware
- [RTX 3060 Specs: 101.9 INT8 TOPS, Compute Capability 8.6](https://gpupoet.com/gpu/learn/card/nvidia-geforce-rtx-3060)
