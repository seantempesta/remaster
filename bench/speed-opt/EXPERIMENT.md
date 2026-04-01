# NAFNet Speed Optimization Experiments

## Goal

Get NAFNet-width64 inference to **>=5 fps at 1080p** so a 42-min Firefly episode (61K frames) costs **<=$3** on Modal.

Current: 0.7 fps on L4 = ~$19/episode. Target: 5+ fps = ~$2.70/episode.

## Method (Karpathy Loop)

Each experiment tests ONE optimization. The profiling script (`cloud/modal_profile.py`) runs on Modal, processes ~50 frames at 1080p, and reports fps / peak_memory / PSNR. Cost: ~$0.10-0.20 per run.

```
LOOP:
  1. Pick ONE optimization
  2. Edit cloud/modal_profile.py or lib/nafnet_arch.py
  3. git commit -m "speed-opt: <description>"
  4. Run: PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_profile.py
  5. Record results in bench/speed-opt/results.tsv
  6. If fps improved AND psnr held (within 1 dB of baseline): KEEP
  7. If not: DISCARD (git reset --hard to last keep commit)
```

## Results File

`bench/speed-opt/results.tsv` — tab-separated, one row per experiment:

```
commit	fps	peak_gb	psnr_db	cost_ep	gpu	batch_size	status	description
```

- **fps**: frames per second (primary metric, higher is better)
- **peak_gb**: peak GPU memory in GB (constraint — must fit in GPU VRAM)
- **psnr_db**: PSNR of output vs SCUNet reference frame (quality gate — must stay within 1 dB of baseline)
- **cost_ep**: estimated cost for 61K frames = `61000 / fps / 3600 * gpu_hourly_rate` (L4=$0.80, A10G=$1.10, A100=$2.78, H100=$3.95)
- **gpu**: GPU type used (L4, A10G, A100, H100)
- **batch_size**: batch size tested
- **status**: `keep` / `discard` / `crash`
- **description**: what this experiment tried

## Experiment Queue (ranked by expected impact)

| # | Optimization | Expected Gain | Notes |
|---|---|---|---|
| 1 | **Baseline** | — | Establish current fps/memory/psnr |
| 2 | **channels_last + cudnn.benchmark** | 1.3-1.6x | 2 lines of code, free perf for CNNs |
| 3 | **torch.compile reduce-overhead** | 1.5-3x | Proper 1080p warmup, not 256x256 |
| 4 | **torch.compile max-autotune** | possibly better than reduce-overhead | Slower warmup, better kernels |
| 5 | **TensorRT fp16** | 3-5x over eager | Nuclear option for static CNNs |
| 6 | **Optimal batch_size** | 1.2-1.5x | Test bs=1,2,3,4,6 at best config |
| 7 | **A100 GPU** | ~3x bandwidth | 2TB/s vs L4's 300 GB/s |
| 8 | **INT8 quantization** | 2x over fp16 | Via TensorRT, check quality |

## Profiling Script

`cloud/modal_profile.py` — self-contained Modal script that:
1. Loads NAFNet-width64 from checkpoint on Modal volume
2. Decodes ~50 frames from clip_mid_1080p.mp4 (uploaded to volume)
3. Runs inference with the specified configuration
4. Measures: fps (excluding warmup), peak VRAM, PSNR vs first input frame
5. Prints a TSV-formatted results line for easy copy-paste

## Key Files

| File | Purpose |
|------|---------|
| `cloud/modal_profile.py` | Modal profiling script (edit per experiment) |
| `bench/speed-opt/results.tsv` | Experiment results log |
| `bench/speed-opt/EXPERIMENT.md` | This file — experiment guide |
| `lib/nafnet_arch.py` | NAFNet architecture (may be edited for optimizations) |
| `checkpoints/nafnet_distill/nafnet_best.pth` | Trained checkpoint (443MB) |
| `data/clip_mid_1080p.mp4` | Reference clip for profiling (81MB, 720 frames) |

## Environment

- **Local**: Windows 11, conda env `upscale`, RTX 3060 6GB — DO NOT run inference locally
- **Modal**: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_profile.py`
- **Volume**: `upscale-data` mounted at `/mnt/data` in containers
- Checkpoint uploaded to `/mnt/data/checkpoints/nafnet_best.pth`
- Clip uploaded to `/mnt/data/input/clip_mid_1080p.mp4`

## Quality Gate

Baseline PSNR (NAFNet output vs compressed input) should be ~56.8 dB (from distillation training). Any optimization that drops PSNR by more than 1 dB is rejected. INT8 gets a 2 dB tolerance since the quality tradeoff is intentional.

## Critical Gotchas

- **torch.compile warmup**: Must warm up at actual 1080p resolution (1920x1088 padded), NOT 256x256. Different shapes trigger separate compilations.
- **NAFNet fp16 LayerNorm**: `lib/nafnet_arch.py` casts to fp32 for normalization. Don't revert this.
- **Modal volume**: Must call `vol.reload()` before reading uploaded files in container.
- **PYTHONUTF8=1**: Required on Windows to avoid UnicodeEncodeError with Modal CLI.
- **channels_last**: Apply with `model.to(memory_format=torch.channels_last)` AND convert input tensors too.
- **TensorRT**: May need to rewrite LayerNorm2d custom autograd for ONNX export compatibility.
