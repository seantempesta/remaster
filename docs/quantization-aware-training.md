# Quantization-Aware Training (QAT) for INT8 NAFNet Inference

Research into QAT workflows for training a NAFNet denoiser that runs INT8 inference on RTX 3060 (Ampere, SM 8.6, 6GB VRAM). The goal is to exploit the dedicated INT8 tensor cores (101.9 TOPS vs 12.7 TFLOPS fp16) for faster 1080p video denoising.

**Current baseline:** ~2 fps fp16 on RTX 3060, ~1.9 fps TensorRT FP16.

---

## Executive Summary

| Question | Answer |
|----------|--------|
| Can QAT work for NAFNet INT8? | Yes, via NVIDIA ModelOpt or PT2E. TorchAO QAT targets linear layers only — not usable for our Conv2d-heavy model. |
| Realistic INT8 speedup on RTX 3060? | **5-40% over FP16 via TensorRT**, not the 2x theoretical. Many real-world CNN workloads are memory-bandwidth-bound, not compute-bound, limiting gains. |
| torch.compile + INT8? | TorchAO quantize_() only handles nn.Linear, not nn.Conv2d. For CNN INT8, must use TensorRT. |
| PSNR loss fp16 -> INT8? | PTQ: 0.1-0.6 dB. QAT: <0.1 dB. For denoising, 0.3 dB is visually imperceptible. |
| Layer-specific gotchas? | LayerNorm must stay fp16/fp32. SimpleGate multiply amplifies quantization error. SCA attention is low-risk. |
| FP8 alternative? | Not on RTX 3060 — requires compute capability 8.9+ (Ada/Hopper). |
| TorchAO maturity for CNNs? | Immature for CNNs. Optimized for LLM linear layers. Not recommended for our Conv2d workload. |

**Bottom line:** QAT is viable but the expected speedup is modest (5-40% over TensorRT FP16). The effort is high relative to the gain. Recommended only if PTQ INT8 via TensorRT shows unacceptable quality loss (>0.3 dB PSNR drop). The bigger wins come from model architecture shrinking (width32, fewer middle blocks) rather than quantization.

---

## 1. QAT Workflow Options for PyTorch CNNs

There are three viable paths for QAT with Conv2d layers. Each inserts fake-quantize (QDQ) nodes during training so the model learns to be robust to INT8 rounding, then exports to an INT8 inference engine.

### 1a. NVIDIA ModelOpt (Recommended for our stack)

NVIDIA Model Optimizer (`nvidia-modelopt`) is the most practical path. It natively supports `nn.Conv2d` quantization (unlike TorchAO) and exports directly to TensorRT-compatible ONNX with QDQ nodes.

**Workflow:**

```python
import modelopt.torch.quantization as mtq

# 1. Load pretrained NAFNet
model = load_nafnet(checkpoint_path, device="cuda", fp16=False)  # fp32 for QAT
model.train()

# 2. Define calibration forward loop
def calibrate(model, calib_loader):
    with torch.no_grad():
        for batch in calib_loader:
            model(batch.cuda())

# 3. Quantize — inserts QDQ nodes around Conv2d and Linear layers
mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)

# 4. Fine-tune (QAT) — typically 10% of original training iterations
# Use same training loop as train_nafnet.py but with quantized model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for batch in train_loader:
    loss = criterion(model(batch), target)
    loss.backward()
    optimizer.step()

# 5. Export to ONNX with QDQ nodes
torch.onnx.export(model, dummy_input, "nafnet_qat_int8.onnx", opset_version=17)

# 6. Build TensorRT engine — QDQ nodes tell TRT which layers run INT8
# trtexec --onnx=nafnet_qat_int8.onnx --int8 --fp16 --saveEngine=nafnet_qat.engine
```

**Key details:**
- ModelOpt wraps `nn.Conv2d` with `QuantConv2d`, inserting input/weight/output quantizers
- The `INT8_DEFAULT_CFG` uses per-channel symmetric quantization for weights, per-tensor for activations
- Calibration (step 3) determines activation ranges; fine-tuning (step 4) adjusts weights to compensate
- The ONNX export preserves QDQ nodes, which TensorRT reads to decide INT8 vs FP16 per layer
- Install: `pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt`

**Integration with our training pipeline:** ModelOpt QAT can run on Modal (A100/H100) using the existing `training/train_nafnet.py` loop. The quantized model exports to ONNX, then converts to TensorRT engine locally on the RTX 3060.

### 1b. PyTorch PT2E QAT (Alternative)

The PyTorch 2 Export (PT2E) quantization path uses `torch.export` + fake quantize insertion. It supports Conv2d via the `XNNPACKQuantizer` or custom quantizers.

**Workflow:**

```python
import torch
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

# 1. Export model to ATen IR
model = load_nafnet(checkpoint_path, device="cpu", fp32=True)
model.train()
example_input = torch.randn(1, 3, 256, 256)
exported = torch.export.export_for_training(model, (example_input,))

# 2. Configure quantizer
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True))

# 3. Insert fake quantize nodes (handles Conv2d + BN fusion)
prepared = prepare_qat_pt2e(exported, quantizer)

# 4. Fine-tune
# Note: cannot call model.train()/eval() on exported model
# Use torch.ao.quantization.move_exported_model_to_train/eval instead

# 5. Convert to quantized model
converted = convert_pt2e(prepared)
```

**Limitations:**
- PT2E is prototype-status in PyTorch 2.11; API may change
- `XNNPACKQuantizer` targets mobile/CPU; for CUDA GPU, need a custom quantizer or use the `EmbeddingQuantizer` / write one
- Cannot call `model.train()` / `model.eval()` after export — must use special API
- More complex than ModelOpt for our TensorRT deployment target
- Does not directly export QDQ nodes for TensorRT

**Verdict:** Use PT2E only if ModelOpt has compatibility issues. PT2E is more general but less polished for the NVIDIA GPU deployment path.

### 1c. Legacy `torch.ao.quantization` (Not Recommended)

The older FX-graph-mode QAT API (`torch.ao.quantization.prepare_qat`) works with Conv2d but:
- Primarily targets CPU inference (FBGEMM/QNNPACK backends)
- No native CUDA INT8 kernel support — quantized ops run on CPU
- Would need to export to ONNX and then TensorRT anyway
- Being superseded by PT2E

**Verdict:** Skip. If we are going through ONNX -> TensorRT regardless, ModelOpt is simpler.

---

## 2. Realistic INT8 Speedup on RTX 3060

### Theoretical vs Real-World

The RTX 3060 has 101.9 INT8 TOPS vs 12.7 TFLOPS FP16 — an 8x theoretical throughput advantage. However, real-world CNN inference speedup from INT8 is **far less than 8x** for several reasons:

**Why INT8 is often only marginally faster than FP16 for CNNs:**

1. **Memory bandwidth bottleneck.** At 1080p, NAFNet processes large feature maps (e.g., 64x1080x1920 at the first encoder level). Moving these through memory is often the bottleneck, not the arithmetic. INT8 halves weight bandwidth but activations may still flow as FP16 (in mixed-precision), so total bandwidth savings are modest.

2. **Lack of optimized INT8 convolution kernels.** TensorRT's FP16 convolution kernels are heavily optimized. INT8 kernels exist but may not cover all conv configurations optimally. TensorRT will fall back to FP16 for layers where INT8 is slower.

3. **Requantization overhead.** Converting between INT8 and FP16 at layer boundaries (especially around LayerNorm, SimpleGate, skip connections) adds overhead that eats into the INT8 compute savings.

4. **Small batch sizes.** Tensor cores achieve peak throughput at large batch sizes (multiples of 32). At batch 1-2, the tensor cores are underutilized regardless of precision.

### Benchmark Evidence

| Source | Model | GPU | INT8 vs FP16 Speedup |
|--------|-------|-----|---------------------|
| NVIDIA TensorRT Best Practices | ResNet-50 | V100 | 1.6x (507 -> 812 qps) |
| TensorRT GitHub issues (#585, #993, #1310) | YOLO variants | RTX 2080/3060 | 1.0-1.1x (often no improvement) |
| NVIDIA blog (autonomous vehicles) | Modified YOLO-v2 | V100 | ~1.5x with INT8 calibration |
| TensorRT GitHub issue #2905 | UNet | Various | INT8 slower than FP16 (attention layers) |
| NVIDIA blog (semantic segmentation) | FCN | V100 | ~3.4x (but FP32->INT8, not FP16->INT8) |

**Critical observation:** Multiple TensorRT GitHub issues report that INT8 provides **0-10% speedup over FP16** on consumer GPUs for CNN workloads. The impressive benchmarks (1.5-2x) tend to be on datacenter GPUs (V100, A100) with large batch sizes.

### Expected NAFNet INT8 Performance on RTX 3060

| Scenario | Expected FPS | Speedup vs FP16 TRT |
|----------|-------------|---------------------|
| TensorRT FP16 (baseline) | ~1.9 fps | 1.0x |
| TensorRT INT8 PTQ, batch 1 | ~2.0-2.3 fps | 1.05-1.2x |
| TensorRT INT8 QAT, batch 1 | ~2.0-2.3 fps | 1.05-1.2x |
| TensorRT INT8 QAT, batch 2 | ~3.5-4.5 fps | 1.8-2.4x (batch effect) |

The batch 2 scenario is where INT8 really helps — not from faster compute, but from **reduced VRAM** allowing higher batch sizes, which improves GPU utilization.

### Why Model Shrinking Beats Quantization

For context, the planned width32 + reduced middle blocks model is expected to reach 15-30 fps in FP16. That is a 10-15x speedup from architecture changes, dwarfing the 5-20% from INT8 quantization. QAT on a smaller model compounds, but the architecture change is the primary lever.

---

## 3. torch.compile and INT8: The Conv2d Problem

### TorchAO Does Not Support Conv2d Quantization

This is a critical finding. TorchAO's `quantize_()` function and its configs (`Int8WeightOnlyConfig`, `Int8DynamicActivationInt8WeightConfig`) **only target `nn.Linear` layers**. They do not quantize `nn.Conv2d`.

NAFNet is almost entirely Conv2d operations:
- `intro`: Conv2d(3, width, 3, 1, 1)
- `ending`: Conv2d(width, 3, 3, 1, 1)
- NAFBlock: 5x Conv2d per block (conv1-conv5) + SCA Conv2d
- Downsample/Upsample: Conv2d

There are **zero `nn.Linear` layers** in NAFNet. TorchAO quantization would be a complete no-op.

**Implication:** The `torch.compile` + TorchAO INT8 path described in our earlier `quantization-research.md` document **will not work for NAFNet**. The code would run without errors but produce zero speedup because no layers get quantized.

### What Actually Works for INT8 CNN Inference

| Path | Conv2d Support | torch.compile | GPU INT8 |
|------|---------------|---------------|----------|
| TorchAO quantize_() | No (Linear only) | Yes | Yes (Linear) |
| PT2E quantization | Yes | Partial | CPU only (XNNPACK) |
| NVIDIA ModelOpt | Yes | No | Via TensorRT |
| TensorRT (ONNX) | Yes | N/A | Yes |
| Torch-TensorRT | Yes | As backend | Yes |

**Verdict:** For INT8 NAFNet on GPU, TensorRT (via ONNX export) is the only mature path. torch.compile cannot help with INT8 Conv2d inference today.

---

## 4. Quality Impact: FP16 to INT8

### Literature on Image Restoration Quantization

| Paper/Source | Model | Quantization | PSNR Drop |
|-------------|-------|-------------|-----------|
| ETASR 2024 | DnCNN | INT8 PTQ | ~0.6 dB (naive), <0.1 dB (calibrated) |
| 2DQuant (NeurIPS 2024) | Various SR | INT8 PTQ | 0.1-0.5 dB |
| 2DQuant (NeurIPS 2024) | Various SR | INT4 PTQ | 1-4 dB |
| IEEE 2024 | Denoisers | INT8 PTQ | 0.2-0.5 dB |
| NVIDIA QAT blog | VGG/ResNet | INT8 QAT | <0.1 dB (matches FP32) |
| General CNN literature | Classification | INT8 QAT | ~0% accuracy loss |

### Key Takeaways

1. **PTQ INT8 on CNNs is usually fine.** Pure CNN denoisers like NAFNet have smooth weight/activation distributions. Expect 0.1-0.3 dB PSNR drop with proper calibration. This is visually imperceptible for denoising.

2. **QAT recovers nearly all quality.** NVIDIA's benchmarks show QAT can achieve FP32-equivalent accuracy at INT8. For NAFNet, QAT should recover any PTQ quality loss to <0.1 dB.

3. **The quality bar for denoising is forgiving.** Unlike super-resolution (which must preserve fine texture), denoising removes noise — small quantization artifacts are masked by the noise reduction benefit. A 0.3 dB PSNR drop in denoised output is well within measurement noise.

4. **Try PTQ first.** If TensorRT INT8 PTQ with calibration data gives <0.3 dB drop, QAT is unnecessary.

### NAFNet-Specific Quality Risks

The main risk areas are:
- **Intro conv (3->width):** Only 3 input channels, less redundancy for quantization. Keep in FP16.
- **Ending conv (width->3):** Directly produces output pixels. Keep in FP16.
- **Skip connections:** Accumulated error from encoder propagates through skip connections to decoder. Mixed precision (INT8 compute, FP16 skip) mitigates this.
- **SimpleGate multiply:** Element-wise multiply of two INT8 tensors produces INT16 intermediate. If the accumulation clips, errors amplify. TensorRT handles this internally with wider accumulators.

---

## 5. Layer-Specific Quantization Gotchas

### LayerNorm2d

**Risk: HIGH. Must stay in FP16/FP32.**

LayerNorm computes mean and variance across channels. In INT8, the dynamic range is only 256 levels (-128 to 127). Channel-wise statistics lose precision catastrophically at INT8 — this is well-documented for both CNN and transformer architectures.

Our `LayerNorm2d` already casts to FP32 internally for fp16 inference. For INT8, TensorRT should automatically keep LayerNorm in FP16 when using mixed-precision (`--int8 --fp16` flags). If using QAT via ModelOpt, LayerNorm is excluded from quantization by default.

**Action:** Verify TensorRT keeps LayerNorm in FP16 by inspecting the engine layer precisions. If not, mark it explicitly.

### SimpleGate (chunk + element-wise multiply)

**Risk: MEDIUM.**

SimpleGate splits a tensor in half along channels and multiplies: `x1 * x2`. When both inputs are INT8:
- The product requires INT16 to avoid overflow
- If the framework handles this correctly (TensorRT does), quality impact is minimal
- If it clips to INT8, errors amplify quadratically

With QAT, the model learns activation scales that keep products in range. With PTQ, calibration should capture the output range of the multiply.

**Action:** No special handling needed if using TensorRT (it manages intermediate precision). If quality drops, keep SimpleGate outputs in FP16.

### Channel Attention (SCA: AdaptiveAvgPool2d + Conv2d(1x1))

**Risk: LOW.**

The SCA block computes a per-channel scaling factor via global average pooling followed by a 1x1 conv. The pooling reduces spatial dimensions to 1x1, which is safe in INT8 (averaging many values). The 1x1 conv is a standard INT8 operation.

The output is a per-channel multiplier applied to features (`x * sca(x)`). This multiply has the same considerations as SimpleGate but is lower risk because the SCA output is a smooth scalar per channel.

**Action:** Safe to quantize to INT8. No special handling needed.

### Beta/Gamma Scaling Parameters

**Risk: LOW.**

NAFBlock uses learnable `beta` and `gamma` parameters for residual scaling: `inp + x * beta`. These are small tensors (1, C, 1, 1) that act as per-channel multipliers. They should stay in FP16 as they are tiny and don't benefit from INT8 compute.

**Action:** TensorRT will likely keep these in FP16 automatically since element-wise multiply with a constant is not a tensor-core operation.

### Recommended Mixed-Precision Strategy for TensorRT

```
INT8:  All Conv2d in NAFBlocks (conv1-conv5), downsample convs, upsample convs
       These are >95% of compute.

FP16:  intro conv, ending conv, LayerNorm2d, beta/gamma scaling,
       skip connection additions
```

This is the default behavior when passing `--int8 --fp16` to TensorRT — it will choose the fastest precision per layer, preferring INT8 where quality permits and falling back to FP16 elsewhere.

---

## 6. FP8 (E4M3/E5M2): Not an Option on RTX 3060

### Hardware Requirements

FP8 tensor core operations require **compute capability 8.9 or higher**:
- Ada Lovelace (RTX 4090, RTX 4080, etc.) — compute capability 8.9
- Hopper (H100, H200) — compute capability 9.0
- Blackwell (B100, B200) — compute capability 10.0

The RTX 3060 is Ampere with compute capability 8.6. **FP8 is not supported in hardware.**

### Software Emulation

NVIDIA's Transformer Engine can emulate FP8 training on Ampere GPUs for development/debugging, but inference does not benefit — the operations fall back to FP16/FP32 on hardware.

vLLM supports FP8 W8A16 (weight-only FP8 with FP16 activations) on compute capability 7.5+ via the Marlin backend, but this is for Linear layers in LLMs, not Conv2d.

### If We Upgraded GPU

| GPU | FP8 Support | FP16 TFLOPS | FP8 TOPS | Price (2026) |
|-----|-------------|-------------|----------|------------|
| RTX 3060 (current) | No | 12.7 | N/A | N/A |
| RTX 4070 | Yes (Ada) | 29.2 | 233 | ~$400 |
| RTX 4080 | Yes (Ada) | 48.7 | 390 | ~$700 |
| RTX 5070 (Blackwell) | Yes | ~45 | ~360 | ~$550 |

An RTX 4070 would give ~2.3x the FP16 TFLOPS plus FP8 support. However, for our use case (overnight batch processing at ~2 fps), the cost of a GPU upgrade is hard to justify vs. the $2.40/episode cloud cost on H100.

**Verdict:** FP8 is not relevant to our RTX 3060. If upgrading GPU, the FP16 speedup alone (2-3x) would matter more than FP8.

---

## 7. TorchAO Maturity Assessment for CNN Inference

### Current State (April 2026)

TorchAO is the official PyTorch quantization library, replacing the legacy `torch.ao.quantization` module. It is:

**Mature for:**
- LLM inference (INT4/INT8 weight-only for Linear layers)
- torch.compile integration via Inductor
- HuggingFace Transformers/Diffusers integration
- INT8 dynamic quantization for Linear layers
- FP8 training and inference (on supported hardware)

**Not mature for:**
- Conv2d quantization (not supported by `quantize_()`)
- CNN model quantization in general
- GPU INT8 inference for convolutions (no CUDA INT8 conv kernels in Inductor)

### Why TorchAO Skips Conv2d

TorchAO's design follows the LLM trend where most compute is in `nn.Linear` (attention projections, FFN layers). CNNs have fallen out of the quantization spotlight. The `quantize_()` API literally filters for `nn.Linear` modules and skips everything else.

There is a PT2E path (`prepare_qat_pt2e` / `convert_pt2e`) that handles Conv2d, but it:
- Targets CPU backends (XNNPACK, FBGEMM)
- Does not have GPU INT8 kernel support via Inductor
- Is labeled "prototype" in PyTorch 2.11

### Recommendation

**Do not use TorchAO for NAFNet quantization.** Use NVIDIA ModelOpt for QAT + TensorRT for deployment. This is the only mature path for GPU INT8 CNN inference.

---

## 8. Complete QAT + TensorRT Deployment Pipeline

### Step-by-Step Plan

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (Modal A100/H100)               │
│                                                             │
│  1. Load NAFNet checkpoint (fp32)                           │
│  2. ModelOpt mtq.quantize() — inserts QDQ nodes             │
│     - Calibrate with 200-500 training frames                │
│     - Automatically wraps Conv2d → QuantConv2d              │
│     - LayerNorm stays fp32, Conv2d gets INT8 quantizers     │
│  3. Fine-tune 2-5K iterations (10% of original training)    │
│     - Same loss function (Charbonnier + VGG perceptual)     │
│     - LR = 1e-5 (1% of original), cosine schedule          │
│     - Same data pipeline from train_nafnet.py               │
│  4. Export ONNX with QDQ nodes                              │
│  5. Validate PSNR on held-out test set                      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    DEPLOYMENT (Local RTX 3060)              │
│                                                             │
│  6. Download ONNX model from Modal volume                   │
│  7. trtexec: convert ONNX → TensorRT engine                │
│     --onnx=nafnet_qat.onnx --int8 --fp16                   │
│     --minShapes=input:1x3x1088x1920                        │
│     --optShapes=input:1x3x1088x1920                        │
│     --maxShapes=input:2x3x1088x1920                        │
│  8. Run inference with TRT engine in denoise_nafnet.py      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Estimated Effort

| Step | Time | Notes |
|------|------|-------|
| Install ModelOpt on Modal image | 30 min | pip install nvidia-modelopt |
| Write QAT wrapper for train_nafnet.py | 2-3 hours | Insert mtq.quantize(), add ONNX export |
| QAT training run (2-5K iters) | 1-2 hours | A100 on Modal |
| ONNX → TensorRT conversion | 30 min | Local, trtexec |
| Integration + benchmarking | 2-3 hours | Modify denoise_nafnet.py pipeline |
| **Total** | **~1 day** | |

### When to Do This

**Not now.** The current priority is model architecture experiments (width32, reduced middle blocks) which offer 10-15x speedup potential. QAT INT8 offers 5-20% on top of that. Do QAT after the architecture is finalized:

1. Finalize architecture (width32, middle blocks 4, etc.)
2. Complete distillation training with final architecture
3. Benchmark TensorRT FP16 on final model
4. Try TensorRT INT8 PTQ with calibration data
5. Only if PTQ quality is unacceptable (>0.3 dB drop), do QAT
6. QAT fine-tune for 2-5K iterations
7. Export and deploy

---

## 9. Alternative: PTQ INT8 Without QAT

Before investing in QAT, try Post-Training Quantization with TensorRT. NAFNet is a pure CNN with well-behaved distributions — PTQ may be sufficient.

### TensorRT PTQ Workflow

```bash
# 1. Export NAFNet to ONNX (already have this from TensorRT FP16 work)
# 2. Build INT8 engine with calibration
trtexec --onnx=nafnet_w64.onnx \
        --int8 --fp16 \
        --calib=calibration_cache.bin \
        --minShapes=input:1x3x1088x1920 \
        --optShapes=input:1x3x1088x1920 \
        --maxShapes=input:2x3x1088x1920 \
        --saveEngine=nafnet_w64_int8.engine
```

For calibration, TensorRT's `IInt8EntropyCalibrator2` or `IInt8MinMaxCalibrator` can be used with 100-500 representative frames. This requires writing a small Python calibrator class, or using trtexec with a directory of calibration images.

**Expected quality:** 0.1-0.3 dB PSNR drop for a well-calibrated CNN. If this is acceptable, QAT is unnecessary.

### Decision Tree

```
TensorRT INT8 PTQ
  ├── PSNR drop < 0.3 dB → Done. Ship PTQ INT8.
  ├── PSNR drop 0.3-0.6 dB → Try mixed precision (intro/ending in FP16)
  │     ├── Recovered to < 0.3 dB → Done.
  │     └── Still > 0.3 dB → Do QAT
  └── PSNR drop > 0.6 dB → Do QAT (unusual for CNN, investigate calibration)
```

---

## 10. Open Questions and Risks

### ModelOpt Compatibility

- **Windows support:** ModelOpt may not have Windows wheels. QAT training runs on Modal (Linux), so this is likely fine. But if we need to debug locally, it could be an issue.
- **PyTorch version:** ModelOpt requires specific PyTorch versions. Need to verify compatibility with our Modal image (PyTorch 2.7.1+cu124, planned upgrade to 2.11.0+cu126).
- **NAFNet custom ops:** The custom `LayerNormFunction` autograd function may confuse ModelOpt's module detection. Should swap to `LayerNorm2dCompile` (uses standard F.layer_norm) before quantization.

### TensorRT INT8 Engine Size and Build Time

- TensorRT INT8 engine build with calibration takes 10-30 minutes for a 1080p model
- Engine is GPU-architecture-specific (must rebuild if changing GPU)
- Engine file is 50-200 MB depending on optimization level

### Calibration Data Sensitivity

- Calibration frames should be representative of actual inference data (compressed video frames, not clean images)
- Using too few calibration frames (<50) can cause poor activation range estimates
- Dark scenes and bright scenes should both be represented
- The calibration cache can be saved and reused across engine builds

---

## Sources

### NVIDIA ModelOpt
- [PyTorch Quantization — NVIDIA Model Optimizer](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)
- [NVIDIA Model Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer)
- [Achieving FP32 Accuracy for INT8 Inference Using QAT with TensorRT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
- [How QAT Enables Low-Precision Accuracy Recovery](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/)

### TorchAO and PyTorch Quantization
- [TorchAO GitHub](https://github.com/pytorch/ao)
- [TorchAO QAT README](https://github.com/pytorch/ao/blob/main/torchao/quantization/qat/README.md)
- [TorchAO: PyTorch-Native Training-to-Serving Model Optimization (ICML 2025)](https://arxiv.org/abs/2507.16099)
- [PT2E QAT Tutorial](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_qat.html)
- [Quantization-Aware Training for LLMs with PyTorch](https://pytorch.org/blog/quantization-aware-training/)
- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)

### TensorRT INT8 Performance
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [INT8 mode is slower than FP16 (TensorRT Issue #993)](https://github.com/NVIDIA/TensorRT/issues/993)
- [INT8 only 5-10% faster than FP16 (TensorRT Issue #585)](https://github.com/NVIDIA/TensorRT/issues/585)
- [Little speed difference INT8 vs FP16 on RTX 2080 (TensorRT Issue #1310)](https://github.com/NVIDIA/TensorRT/issues/1310)
- [Deploying QAT Models in INT8 with Torch-TensorRT](https://docs.pytorch.org/TensorRT/_notebooks/vgg-qat.html)
- [Sparsity in INT8: Training Workflow for TensorRT](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/)

### Image Restoration Quantization
- [A Lightweight Denoising CNN with INT8 Quantization (ETASR 2024)](https://mail.etasr.com/index.php/ETASR/article/download/15428/6155)
- [2DQuant: Low-bit PTQ for Image Super-Resolution (NeurIPS 2024)](https://arxiv.org/abs/2406.06649)
- [INTEGER QUANTIZATION FOR DEEP LEARNING INFERENCE (NVIDIA Whitepaper)](https://arxiv.org/pdf/2004.09602)

### FP8 Hardware
- [Understanding the NVIDIA FP8 Format (Scaleway)](https://www.scaleway.com/en/docs/gpu/reference-content/understanding-nvidia-fp8/)
- [FP8 Introduction (NVIDIA Blog)](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [RTX 4090 FP8 Compute Discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/4090-doesnt-have-fp8-compute/232256)

### Quantization Layer Sensitivity
- [INT8 Quantization on ViT, LayerNorm (TVM Discuss)](https://discuss.tvm.apache.org/t/int8-quantization-on-vit-especially-layernorm/11898)
- [Towards Superior Quantization Accuracy: Layer-sensitive Approach](https://arxiv.org/html/2503.06518v1)
- [Quantizing Attention and Normalization Layers (apxml)](https://apxml.com/courses/quantized-llm-deployment/chapter-5-addressing-advanced-challenges/quantizing-specific-llm-components)
