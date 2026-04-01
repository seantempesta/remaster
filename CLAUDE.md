# Video Enhancement Experiment Platform

## What This Is
An experimentation platform for improving video quality using ML models. The primary use case is reducing compression artifacts in existing video libraries (e.g., BluRay rips) at native resolution.

## Goals
- Denoise/enhance 1080p video content using learned models
- Process full episodes in reasonable time on consumer hardware (RTX 3060 6GB)
- Compare different approaches: per-frame denoisers, temporal fusion, diffusion models
- Eventually train a custom fast model via distillation

## Architecture
- **Local inference** runs on Windows with conda env `upscale` (Python 3.10, PyTorch 2.5.1+cu121)
- **Cloud training** uses [Modal](https://modal.com) for GPU compute — CLI authenticated, account with billing attached
- **Streaming pipeline** reads video -> processes frames -> writes video (no intermediate files)
- All scripts are standalone Python — no framework dependencies beyond PyTorch

## Key Constraints
- 6GB VRAM — must use fp16, tiling, or half-res tricks to fit
- Dependencies get overwritten easily — always install PyTorch CUDA from `--index-url https://download.pytorch.org/whl/cu121` LAST or with `--no-deps`
- Prefer patching code for modern PyTorch over pinning old dependencies
- xformers is unreliable on Windows — use native `F.scaled_dot_product_attention` instead

## Key Scripts
- `denoise_episode.py` — main pipeline: streams full episodes through SCUNet denoiser
- `denoise_scunet.py` / `denoise_scunet_mid.py` — frame-based SCUNet processing
- `denoise_pipeline.py` — optical flow warp-fuse approach (experimental)
- `run_raft.py` — RAFT optical flow computation
- `run_depth.py` — Video Depth Anything depth map generation
- `compare.py` — PSNR/SSIM metrics and side-by-side comparison generation
- `bench_*.py` — benchmarking scripts for optimization

## External Repos (cloned, git-ignored)
- `SCUNet/` — Swin-Conv-UNet denoiser (the current best approach)
- `RAFT/` — optical flow estimation
- `Video-Depth-Anything/` — temporally consistent depth maps
- `BasicVSR_PlusPlus/` — video super resolution (archived, not actively used)
- `mmagic/` — OpenMMLab (archived, not used)

## Data Directory
`data/` contains video clips, extracted frames, model outputs. Git-ignored due to size.
