# Architecture

## Directory Structure

```
remaster/
  lib/                    Shared importable code (paths, ffmpeg, metrics, NAFNet arch)
  pipelines/              Production streaming denoisers (batch SCUNet, NAFNet, episode)
  remaster/               Production VapourSynth pipeline scripts (encode, playback)
  experiments/            One-off experiments and older approaches
  training/               NAFNet distillation training
  cloud/                  Modal remote GPU execution
  bench/                  Benchmarking and quality comparison
  tools/                  Small utilities (clip extraction, probing, MP4 repair)
  docs/                   Documentation
  reference-code/         Git submodules (SCUNet, RAFT, NAFNet, Video-Depth-Anything, etc.)
  bin/                    Standalone binaries (ffmpeg.exe)
  checkpoints/            Trained model weights (git-ignored)
  data/                   Training data, validation data, archived outputs (git-ignored, symlink to E:/)
    mixed_pairs/          3,382 training pairs (input/ + target/ PNGs)
    mixed_val/            362 validation pairs (input/ + target/ PNGs)
    archive/              Model comparison outputs (full episodes, demo clips)
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

## VapourSynth Pipeline (Production)

```
Source video (any format)
  |
  v
BestSource/lsmas decode --> VapourSynth resize (YUV->RGBS) --> vs-mlrt TensorRT --> VapourSynth resize (RGBS->YUV) --> vspipe y4m pipe --> ffmpeg NVENC HEVC
                                                                    |
                                                              All C++ -- no Python GIL
```

## Shared Library (`lib/`)

- **paths.py**: Portable path resolution for SCUNet, RAFT, KAIR directories. Works across local Windows, Modal containers, and different project locations via candidate-list search.
- **ffmpeg_utils.py**: FFmpeg discovery and video metadata extraction. Shared by all pipeline and data extraction scripts.
- **metrics.py**: PSNR and SSIM computation. Shared by compare.py and bench_nafnet.py.
- **nafnet_arch.py**: Standalone NAFNet architecture with fp16-safe LayerNorm fix. No basicsr dependency.
- **plainnet_arch.py**: PlainDenoise and UNetDenoise architectures (INT8-native, RepConvBlock).

## Reference Code Submodules

External repos are cloned as git submodules under `reference-code/`. Scripts use `lib/paths.py` to resolve their locations and add them to `sys.path` at runtime:

```python
from lib.paths import add_scunet_to_path
add_scunet_to_path()  # Adds reference-code/SCUNet to sys.path
from models.network_scunet import SCUNet  # Now works
```

Some submodule files have local patches (xformers -> SDPA, thop try/except) applied directly to the submodule working tree.

Additional tools (not submodules):

- **VapourSynth**: C++ frame server for video processing. Provides the filter graph that connects decode, resize, inference, and output.
- **vs-mlrt**: VapourSynth plugin for TensorRT/ONNX inference. Runs NAFNet ONNX models entirely in C++ with no Python GIL overhead.
- **BestSource**: VapourSynth source plugin for frame-accurate decoding. Alternative to lsmas for broad format support.

## Modal Deployment

Modal scripts (`cloud/`) bundle local files into container images:

- `modal_denoise.py`: Bundles `pipelines/denoise_batch.py` + `lib/` + SCUNet into an L4 container with custom ffmpeg (NVENC + libx265)
- `modal_train.py`: Bundles `training/train.py` + `lib/` into an L40S container (unified training, all architectures)
- `modal_flashvsr.py`: Self-contained — clones repos inside the container image
- `modal_runner.py`: Generic runner — bundles the entire project (minus data/)

Data transfer uses Modal Volumes (`upscale-data`).
