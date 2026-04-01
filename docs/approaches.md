# Approach Comparison

| Approach | Quality | Speed | VRAM | Script | Status |
|----------|---------|-------|------|--------|--------|
| SCUNet | Best | 0.52 fps | 3.1GB | `pipelines/denoise_batch.py` | Production |
| NAFNet (distilled) | TBD | 10-30+ fps target | ~1GB | `pipelines/denoise_nafnet.py` | Training |
| FlashVSR | TBD | Near real-time? | ~6GB | `cloud/modal_flashvsr.py` | Testing (active run) |
| Real-ESRGAN roundtrip | Mediocre | 0.19 fps | ~4GB | `experiments/realesrgan_roundtrip.py` | Done, not viable |
| RAFT flow-fusion | Bad (blurry) | N/A | N/A | `experiments/flow_warp_fuse.py` | Failed |
| Real-ESRGAN upscale | Good (perceptual) | 0.19 fps | ~4GB | `experiments/realesrgan_upscale.py` | Done |
| Video Depth Anything | N/A (depth only) | ~30s/720 frames | ~2GB | `experiments/depth_maps.py` | Done, not integrated |

## Key Trade-offs

- **SCUNet** is the quality leader but at 0.52 fps, a 42-min episode takes ~32 hours
- **NAFNet distillation** aims to match SCUNet quality at 10-30x speed by training a student CNN
- **FlashVSR** is a one-step diffusion model — potential to combine quality and speed, but unproven for denoising at native resolution
- **Flow-based approaches** consistently produce blur — temporal fusion needs learned components
