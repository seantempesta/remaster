# Experiment Log

Results from completed experiments on the Firefly S01E01 test clips.

## 1. Real-ESRGAN Frame-by-Frame Upscaling (480p -> 1080p)

- **Script:** `experiments/realesrgan_upscale.py`
- **Hardware:** RTX 3060 6GB, fp16 with 384px tiling
- **Speed:** ~0.19 fps (5.3s/frame)
- **Result:** Looks great perceptually but scores worse than bicubic on PSNR/SSIM because it hallucinates detail
- **Verdict:** Good for upscaling, not useful for denoising at native resolution

## 2. RAFT Optical Flow + Warp-Fuse + Real-ESRGAN

- **Scripts:** `experiments/raft_flow.py` → `experiments/fuse_only.py` → `experiments/fused_sr.py` (or combined: `experiments/warp_fuse_sr.py`)
- **Flow data:** `data/flow_npy/` (~2GB for 150 frame pairs)
- **Result:** Flow fusion introduced blur and artifacts. Scored worse than plain Real-ESRGAN
- **Problem:** Hand-crafted pixel averaging fundamentally can't distinguish noise from detail. Even with occlusion masking, soft consistency weighting, and motion-aware blending, the result was blurry
- **Verdict:** Warp-fuse is not a viable denoising approach without a learned component

## 3. SCUNet Learned Denoiser (current best)

- **Script:** `pipelines/denoise_batch.py` (production), `experiments/scunet_frame.py` / `scunet_mid.py` (early tests)
- **Hardware:** RTX 3060 6GB, fp16 direct inference
- **Speed:** ~0.52 fps (1.9s/frame), 3.1GB VRAM
- **Model:** `scunet_color_real_psnr` — trained on synthetic real-world degradations including JPEG compression
- **SDPA optimization:** 10% speedup + 260MB less VRAM (see `bench/bench_sdpa.py`)
- **Full episodes:** ~32 hours per 42-min episode at 0.52 fps
- **Verdict:** Best quality so far but too slow for a full library. Motivates distillation (idea #6)

## 4. Video Depth Anything

- **Script:** `experiments/depth_maps.py`
- **Hardware:** RTX 3060 6GB, vits model at 364px input
- **Speed:** ~30s for 720 frames
- **Result:** Produces temporally consistent depth maps. Not yet integrated into any pipeline
- **Output:** `data/depth_output/`, `data/depth_npy/`

## 5. Real-ESRGAN Roundtrip Denoiser (4x up + downscale back)

- **Script:** `experiments/realesrgan_roundtrip.py`
- **Concept:** Upscale to 4K then downscale back to 1080p — model removes artifacts as a side effect
- **Speed:** ~0.19 fps (same as upscaling, since it's the same model)
- **Verdict:** Partially tested, interrupted. Untested at scale

## 6. 1080p Flow-Fusion Pipeline

- **Script:** `experiments/flow_warp_fuse.py`
- **Concept:** RAFT flow at half-res, upscale to full-res, occlusion-aware warp-fuse at 1080p
- **Result:** Same blur problem as #2, just at native resolution
- **Output:** `data/frames_1080p_denoised/`

## 7. NAFNet Distilled Denoiser — Speed Optimization (2026-04-01)

- **Scripts:** `cloud/modal_profile.py` (profiling), `cloud/modal_denoise.py` (production), `pipelines/denoise_nafnet.py` (pipeline)
- **Model:** NAFNet-width64 distilled from SCUNet (56.82 dB val PSNR vs teacher)
- **Starting point:** 0.7 fps on Modal L4, same speed as SCUNet despite being a pure CNN

### Speed optimization journey

| Change | FPS | Cost/ep | GPU |
|--------|-----|---------|-----|
| Baseline (eager, L4) | 0.78 | $17.38 | L4 |
| + torch.compile + channels_last | 2.56 | $5.29 | L4 |
| + A100 (more bandwidth) | 12.43 | $3.79 | A100 |
| + PyTorch 2.7.1 + Inductor opts | 15.0 | $2.37 | A100-80GB |
| + H100 (profiler) | 30.82 | $2.17 | H100 |
| + CUDA graph shape fix (pipeline) | **27.9** | **~$2.40** | H100 |

Key optimizations:
- **PyTorch 2.5.1 → 2.7.1** + cu124 (better Inductor codegen)
- **TORCHINDUCTOR_FREEZING=1** (inlines weights as constants, 15-30% gain)
- **conv_1x1_as_mm=True** (1x1 conv → GEMM, better Tensor Cores)
- **LayerNorm2dCompile** (F.layer_norm wrapper — Inductor can fuse it, unlike custom autograd.Function)
- **CUDA graph shape fix** — pipeline warmed up at 1920x1088 but fed 1920x1080 tensors, causing re-recording every batch. Fixed with pre-allocated padded buffers.
- **PyAV decode** — in-process decode eliminates subprocess pipe bottleneck
- **Pre-allocated pinned memory** — avoids CUDA graph invalidation from alloc/free
- **apt ffmpeg + libx264** — H100/A100 have no NVENC; libx264 encodes at 440 fps on CPU

### What didn't work
- **TensorRT 2.7.0** — DataDependentOutputException bug in weight mapping, falls back to eager (3 fps)
- **INT8 quantization** — blocked by TRT; realistic gain only 1.2-1.4x due to depthwise conv limitations
- **Batch size > 8** — CUDA graphs don't scale; bs=16 is 2.75x slower, bs=32 OOMs on 80GB
- **GroupNorm(1) swap** — NOT equivalent to LayerNorm2d (different normalization dimensions), produced garbage

### Full episode result
- **Firefly S01E02**: 61,463 frames in 36.7 min at 27.9 fps, 0 errors, ~$2.40 on H100
- Output: H.264 High10 + 5.1 audio + commentary + 3 subtitle tracks + chapter markers
- **Quality issues:** Slightly soft, dark shadow areas still noisy — needs training improvements

### Research & docs
- Full research log: `bench/speed-opt/research.md`
- All experiment results: `bench/speed-opt/results.tsv`
- Transfer doc: `bench/speed-opt/TRANSFER.md`
