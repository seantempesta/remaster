# Ideas, Experiments, and Results

## What We've Tried

### 1. Real-ESRGAN Frame-by-Frame Upscaling (480p -> 1080p)
- **Status:** Done, works
- **Result:** Looks great perceptually but scores worse than bicubic on PSNR/SSIM because it hallucinates detail
- **Speed:** ~0.19 fps (4x upscale with tiling on 6GB)
- **Verdict:** Good for upscaling, not useful for denoising at native resolution

### 2. RAFT Optical Flow + Warp-Fuse + Real-ESRGAN
- **Status:** Done, disappointing
- **Result:** Flow fusion introduced blur and artifacts. Scored worse than plain Real-ESRGAN
- **Problem:** Hand-crafted pixel averaging fundamentally can't distinguish noise from detail. Even with occlusion masking, soft consistency weighting, and motion-aware blending, the result was blurry
- **Verdict:** Warp-fuse is not a viable denoising approach without a learned component

### 3. SCUNet Learned Denoiser (current best)
- **Status:** Working, deployed as streaming pipeline
- **Result:** Noticeably cleaner frames, removes compression artifacts well
- **Speed:** ~0.52 fps fp16 direct inference, 3.1GB VRAM
- **Model:** `scunet_color_real_psnr` — trained on synthetic real-world degradations including JPEG compression
- **Optimization:** SDPA attention replacement gives 10% speedup + 260MB less VRAM
- **Verdict:** Best approach so far. Quality is good but speed is the bottleneck for full episodes

### 4. Video Depth Anything
- **Status:** Working (vits model at 364 input size)
- **Result:** Produces temporally consistent depth maps. Looks cool but not yet integrated into any pipeline
- **Speed:** Very fast with small model (~30s for 720 frames)

### 5. Real-ESRGAN Roundtrip Denoiser (4x up + downscale back)
- **Status:** Partially tested, interrupted
- **Concept:** Upscale to 4K then downscale back to 1080p — the model removes artifacts as a side effect
- **Verdict:** Untested at scale

## Ideas Not Yet Tried

### 6. Distilled Student Model
- **Concept:** Use SCUNet as a teacher to generate (noisy, clean) training pairs from actual Firefly footage. Train a tiny fast student model (small U-Net or MobileNet-style) that mimics SCUNet's output
- **Expected speed:** 10-30+ fps — fast enough for real-time or near-real-time processing
- **Training:** Use Modal cloud GPUs. Generate pairs locally, upload, train, download weights
- **Open question:** How small can the student be before quality degrades noticeably?

### 7. Diffusion Model Denoiser
- **Concept:** Fine-tune a one-step consistency/distilled diffusion model for video denoising
- **Candidates:** FlashVSR (one-step diffusion VSR, claimed 6GB), DOVE
- **Challenge:** Training diffusion models needs significant compute. Inference could be fast with distillation
- **Could use depth+flow:** Diffusion models accept conditioning signals naturally (like ControlNet)

### 8. Depth + Flow Conditioned Model
- **Concept:** Train a model that takes frame + depth map + optical flow as input channels, outputs clean frame
- **Why:** Depth tells the model about scene structure (foreground vs background). Flow provides temporal context without needing multiple frames at inference time
- **Approach:** Add extra input channels to a pre-trained denoiser, fine-tune on video data
- **Data:** We already have RAFT flow and Video Depth Anything depth maps computed

### 9. Temporal Consistency Post-Processing
- **Concept:** Run SCUNet per-frame (current approach) but add a lightweight temporal consistency pass afterward using optical flow to reduce flicker between frames
- **Simpler than:** Training a full temporal model
- **Could be:** A fast warp-blend between consecutive SCUNet outputs, or a learned temporal filter

### 10. FlashVSR — One-Step Diffusion (Potentially Real-Time)
- **Status:** Not yet tried
- **Repo:** https://github.com/OpenImagingLab/FlashVSR (also FlashVSR-Pro: https://github.com/LujiaJin/FlashVSR-Pro with low-VRAM tiling)
- **Concept:** One-step diffusion model for video super-resolution. Single forward pass, no iterative denoising. Could potentially run at or near real-time on RTX 3060
- **Why exciting:** Combines the quality of diffusion models with the speed of single-pass inference. Claimed to work on 6GB VRAM
- **Use case:** Could replace SCUNet as the denoiser if quality is comparable, at 10-50x the speed
- **Open questions:** Does it work as a denoiser at native resolution (not just upscaling)? VRAM with 1080p input?

### 11. NVENC Hardware-Accelerated Pipeline
- **Status:** Partially implemented (H.265 NVENC encoding added to episode script)
- **Concept:** Use the RTX 3060's dedicated video encoder for output, freeing CUDA cores entirely for inference
- **Also possible:** NVDEC for hardware-accelerated input decoding

## Key Learnings
- Hand-crafted temporal fusion (warp + average) always introduces blur — needs a learned component
- PSNR/SSIM metrics penalize perceptual enhancement models — visual quality is what matters
- fp16 inference is critical for 6GB VRAM — halves memory with negligible quality loss
- Tiling is 7x slower than direct inference — avoid if the model fits in VRAM
- Dependencies on Windows are fragile — pin PyTorch CUDA version, avoid xformers, patch code for modern APIs
- Streaming pipeline (ffmpeg pipes) avoids disk space issues for full episodes
- SCUNet at 0.52fps means ~32 hours per 42-min episode — too slow for a full library, motivates distillation
