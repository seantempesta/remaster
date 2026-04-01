# Architecture

## Directory Structure

```
upscale-experiment/
  lib/                    Shared importable code (paths, ffmpeg, metrics, NAFNet arch)
  pipelines/              Production streaming denoisers (batch SCUNet, NAFNet, episode)
  experiments/            One-off experiments and older approaches
  training/               NAFNet distillation training
  cloud/                  Modal remote GPU execution
  bench/                  Benchmarking and quality comparison
  tools/                  Small utilities (clip extraction, probing, MP4 repair)
  docs/                   Documentation
  reference-code/         Git submodules (SCUNet, RAFT, NAFNet, Video-Depth-Anything, etc.)
  bin/                    Standalone binaries (ffmpeg.exe)
  checkpoints/            Trained model weights (git-ignored)
  data/                   Video clips, frames, outputs (git-ignored, ~2GB)
```

## Data Flow

```
Source video (MKV/MP4)
  |
  v
ffmpeg decode (rawvideo pipe) --> GPU model (fp16) --> ffmpeg encode (H.265 pipe)
  |                                                        |
  v                                                        v
Raw RGB frames in memory                            Output video (MKV)
```

The streaming pipeline (`pipelines/denoise_batch.py`) uses threaded reader/writer processes to overlap IO with GPU inference. No intermediate files are written to disk.

## Shared Library (`lib/`)

- **paths.py**: Portable path resolution for SCUNet, RAFT, VDA directories. Works across local Windows, Modal containers, and different project locations via candidate-list search.
- **ffmpeg_utils.py**: FFmpeg/ffprobe discovery and video metadata extraction. Shared by all pipeline and benchmark scripts.
- **metrics.py**: PSNR and SSIM computation. Shared by compare.py and bench_nafnet.py.
- **nafnet_arch.py**: Standalone NAFNet architecture with fp16-safe LayerNorm fix. No basicsr dependency.

## Reference Code Submodules

External repos are cloned as git submodules under `reference-code/`. Scripts use `lib/paths.py` to resolve their locations and add them to `sys.path` at runtime:

```python
from lib.paths import add_scunet_to_path
add_scunet_to_path()  # Adds reference-code/SCUNet to sys.path
from models.network_scunet import SCUNet  # Now works
```

Some submodule files have local patches (xformers → SDPA, thop try/except) applied directly to the submodule working tree.

## Modal Deployment

Modal scripts (`cloud/`) bundle local files into container images:

- `modal_denoise.py`: Bundles `pipelines/denoise_batch.py` + `lib/` + SCUNet into an L4 container with custom ffmpeg (NVENC + libx265)
- `modal_train.py`: Bundles `training/train_nafnet.py` + `lib/nafnet_arch.py` into an A10G container
- `modal_flashvsr.py`: Self-contained — clones repos inside the container image
- `modal_runner.py`: Generic runner — bundles the entire project (minus data/)

Data transfer uses Modal Volumes (`upscale-data`).
