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
