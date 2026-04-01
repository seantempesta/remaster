# NAFNet Speed Optimization Experiments

## Goal

Maximize NAFNet-width64 inference fps at 1080p on Modal while maintaining quality (PSNR within 1 dB of baseline). Lower cost per episode is better — but the primary objective is **maximum throughput**. Don't settle for "good enough."

## How to Run an Experiment

1. Read `bench/speed-opt/results.tsv` — understand what's been tried and what the current best is
2. Read `bench/speed-opt/research.md` — check what previous agents have already researched
3. **Do web research FIRST** before trying anything new. Use WebSearch/WebFetch to look up:
   - Current best practices for the optimization you're attempting (e.g. "torch_tensorrt NAFNet CNN inference optimization 2025")
   - Known issues and compatibility problems (e.g. "torch_tensorrt custom LayerNorm ONNX export")
   - Benchmark data from others doing similar work
   - Do NOT guess from memory — APIs change, versions matter, and wrong assumptions waste expensive GPU time
   - **Write your findings to `bench/speed-opt/research.md`** — append a dated section with what you learned, links, and conclusions. Future agents depend on this.
4. **Think hard** about the best next experiment. Consider:
   - What is the current bottleneck? (memory bandwidth? compute? batch utilization? GPU choice?)
   - What does the data suggest? (e.g. if peak VRAM is 2 GB on a 40 GB GPU, increase batch size!)
   - Don't just increment one variable — make the smartest move given ALL available information
5. Run: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_profile.py [flags]`
6. Verify results from Modal logs: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal app list` then `... -m modal app logs <app-id>`
7. Record results in `bench/speed-opt/results.tsv`
8. If fps improved and quality held: **keep** — commit and build on it
9. If not: **discard** — record it and try something different

## Profiling Script Flags

```
cloud/modal_profile.py [options]:
  --batch-size N        Frames per batch (default: 1)
  --compile             Enable torch.compile
  --compile-mode MODE   reduce-overhead (default) or max-autotune
  --channels-last       Use channels_last memory format
  --cudnn-benchmark     Enable cudnn.benchmark
  --tensorrt            Use TensorRT (requires trt_image)
  --gpu GPU             L4 (default), A10G, or A100
  --num-frames N        Frames to process (default: 50)
```

## Results File

`bench/speed-opt/results.tsv` — tab-separated, one row per experiment:

```
commit	fps	peak_gb	psnr_db	cost_ep	gpu	batch_size	status	description
```

Cost formula: `61000 / fps / 3600 * gpu_hourly_rate` (L4=$0.80, A10G=$1.10, A100=$2.78, H100=$3.95)

## Key Context

- **NAFNet-width64**: 116M params, pure CNN, no attention, no dynamic control flow. torch.compile and TensorRT friendly.
- **Memory-bandwidth bound**: steady per-frame latency with zero variance confirms this. Optimizations that reduce memory traffic (kernel fusion, bigger batches to amortize weight reads) help most.
- **Available GPUs on Modal**: L4 (24GB, 300 GB/s, $0.80/hr), A10G (24GB, 600 GB/s, $1.10/hr), A100 (40GB, 2 TB/s, $2.78/hr), H100 (80GB, 3.35 TB/s, $3.95/hr)
- **VRAM headroom**: compiled model uses ~2 GB peak at bs=1. Most GPUs have massive headroom for larger batches.

## Optimization Levers

| Lever | What it does | When to use |
|-------|-------------|-------------|
| **batch_size** | Process N frames at once. Amortizes weight reads, improves GPU utilization. | When peak VRAM << available VRAM |
| **torch.compile** | Fuses ops, reduces kernel launches, enables CUDA graphs. | Always — 2-3x speedup confirmed |
| **compile mode** | `reduce-overhead` uses CUDA graphs. `max-autotune` adds Triton kernel search. | Try max-autotune if reduce-overhead plateaus |
| **channels_last** | NHWC memory layout, better for cuDNN conv kernels. | Helps ~15% with compile, hurts without |
| **cudnn.benchmark** | Autoselects fastest conv algorithm. | Free, always enable with compile |
| **TensorRT** | Maximum kernel fusion for static CNNs. Eliminates intermediate memory writes. | If compile isn't enough — biggest potential gain |
| **GPU tier** | More bandwidth = more fps for bandwidth-bound models. | When software optimizations plateau |
| **INT8 quantization** | Halves memory bandwidth needs. | Last resort — may affect quality |

## Environment

- **Local**: Windows 11, conda env `upscale`, RTX 3060 6GB — **DO NOT run inference locally**
- **Modal command**: `PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_profile.py`
- **Verify results**: Always check Modal logs directly. Agent-reported results must be verified.
- **Volume**: `upscale-data` mounted at `/mnt/data`. Checkpoint at `/mnt/data/checkpoints/nafnet_best.pth`, clip at `/mnt/data/input/clip_mid_1080p.mp4`.

## Critical Gotchas

- **torch.compile warmup**: Must warm up at 1920x1088 (actual resolution), not 256x256.
- **NAFNet fp16 LayerNorm**: `lib/nafnet_arch.py` casts to fp32 for normalization. Don't revert.
- **Modal volume**: Must call `vol.reload()` before reading uploaded files.
- **PYTHONUTF8=1**: Required on Windows for Modal CLI.
