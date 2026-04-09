# NeRV Denoising - Implementation Status

## What Exists (2026-04-09)

### Completed
- `tools/train_nerv.py` — Standalone HNeRV training script (0.92M params, fits 6GB VRAM)
- `output/nerv/clip_02/` — 21 epochs trained, checkpoints at epochs 4, 9, 19
- `E:/upscale-data/nerv-test/clip_02/` — 240 frames from Firefly S01E01 at 20:00 (faces, interior, 1080p)
- `lib/convnext_autoencoder.py` — ConvNeXt-V2 autoencoder (separate exploration, not used by HNeRV)

### Architecture in train_nerv.py
- HNeRVSimple: ConvNeXt encoder (5 stages, stride 5,3,2,2,2) + PixelShuffle decoder
- 0.92M params (encoder 0.31M + decoder 0.62M)
- AMP FP16 training, fits in 6GB VRAM at 1080p batch_size=1
- ~28s per epoch on RTX 3060
- Metrics: PSNR, HF energy ratio (FFT), late-layer weight norm

### Early Results (21 epochs)
- Train PSNR: 9.2 -> 30.9 dB
- Val PSNR: 8.6 -> 28.1 dB
- HF ratio drops from 3.3 to 0.2 (spectral bias working)
- Model learns structure first, hasn't started memorizing noise yet

## Next Step
Integrate into existing training pipeline (`training/train.py`) instead of standalone script.
Reuse: W&B logging, loss functions, viz composites, checkpoint management.
Add: HNeRV arch to lib/, sequential dataset, FFT noise metrics to W&B.
Resume from epoch 19 checkpoint.
