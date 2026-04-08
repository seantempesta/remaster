# Fast Architecture Research Brief

## Goal
Find or design a denoising model that runs at **30+ fps at 1080p** on RTX 3060
(33ms/frame budget). Quality target: visually comparable to NAFNet w32_mid4.

## Strategy
1. Work backwards from hardware: what ops are fastest on this GPU + TensorRT/torch.compile?
2. Find pretrained fast models we can fine-tune (cheapest path)
3. If nothing exists, design from scratch and distill from our NAFNet w32_mid4 teacher

## Teacher Available
NAFNet w32_mid4 at 5.5 fps produces excellent results (49.50 dB PSNR vs SCUNet targets).
We can use it as the teacher for distillation — no need to re-run the slow SCUNet.
Training cost: ~$13 for 25K iters on H100. Fast iteration.

## What the Next Agent Should Do

### Phase 1: Benchmark the hardware limits
Run `bench/sweep_architectures.py` on a clean GPU to find what NAFNet configs hit 30 fps.
This tells us the parameter/FLOP budget.

### Phase 2: Research fastest ops for this GPU
Look at what TensorRT 10.16 and torch.compile optimize best:
- **Depthwise separable convolutions** (MobileNet-style) — much fewer FLOPs than standard conv
- **Group convolutions** — middle ground between standard and depthwise
- **Fused Conv+BN+ReLU** — TRT has dedicated kernels
- **Avoid:** channel attention (global pooling at 1080p is expensive), transpose ops, 
  layer normalization (decomposed into many ops in TRT)
- Check TensorRT's layer fusion documentation and supported optimized patterns
- Check `reference-code/TensorRT/` for optimized layer examples

### Phase 3: Search for pretrained fast denoisers
Models to investigate (search GitHub, papers, model zoos):

**Image restoration models optimized for speed:**
- **ECBSR** (Edge-oriented Conv Block for SR) — designed for mobile, very fast
- **RFDN** (Residual Feature Distillation Network) — NTIRE 2020 efficient SR winner
- **IMDN** (Information Multi-Distillation Network) — lightweight SR
- **PAN** (Pixel Attention Network) — simple attention, fast
- **BSRN** (Blueprint Separable Residual Network) — NTIRE 2022 efficient SR winner
- **SAFMN** (Spatially-Adaptive Feature Modulation Network) — very recent, designed for speed
- **PlainUSR** — plain convolution USR, no attention at all
- **MobileSR** variants — MobileNet backbone for super-resolution

**Denoising-specific:**
- **FFDNet** — fast and flexible denoising (plain CNN)
- **DnCNN** — classic deep denoiser (17-layer plain CNN, very fast)
- **IRCNN** — iterative residual CNN denoiser
- Check if any NTIRE denoising challenge winners focused on speed

**Key:** These are mostly super-resolution models but the architecture transfers to
denoising (same input/output resolution, just change the final layer). Look for ones
with pretrained weights on denoising or general restoration tasks.

### Phase 4: Design criteria for custom architecture
If building from scratch, the architecture should:
1. Use ONLY ops that TRT fuses well: Conv3x3, Conv1x1, ReLU/GELU, Add, PixelShuffle
2. Avoid: global pooling, channel attention, LayerNorm, Transpose
3. Use depthwise separable convolutions for large channel counts
4. Minimize downsampling levels (each level = more memory traffic)
5. Keep channels narrow (16-32) to reduce memory bandwidth
6. Residual connection from input to output (model learns residual noise)
7. Consider processing at half resolution (540p) with final PixelShuffle to 1080p

### Phase 5: Train and evaluate
Use the existing training infrastructure:
- `training/train_nafnet.py` — modify to support new architectures
- `training/losses.py` — Charbonnier + DISTS + FFT losses already implemented
- `cloud/modal_train.py` — Modal wrapper for GPU training
- Teacher targets: use NAFNet w32_mid4 output (5.5 fps, much faster than SCUNet)

## Reference: RTX 3060 Laptop Specs
- SM 8.6 (Ampere), 30 SMs
- 192-bit memory bus, ~336 GB/s bandwidth
- FP16 tensor core peak: ~12.7 TFLOPS
- 6 GB GDDR6 VRAM
- Our model uses ~5% of compute, ~5% of bandwidth — heavily memory-bound
- Implication: reducing memory traffic (fewer channels, less data movement) matters
  more than reducing FLOPs

## Key Files
- `bench/sweep_architectures.py` — benchmark different NAFNet configs
- `lib/nafnet_arch.py` — current architecture (modify or create new)
- `training/train_nafnet.py` — training loop (supports --width, --middle-blk-num)
- `docs/architecture-investigation.md` — detailed analysis of current bottleneck
- `CLAUDE.md` — project context (corrected speed numbers)
- `reference-code/TensorRT/` — TRT source for checking optimized ops
- `reference-code/torch-tensorrt/` — torch_tensorrt source
