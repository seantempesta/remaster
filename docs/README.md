# Documentation

## Guides

- [Setup](setup.md) -- Environment, dependencies, CUDA, TensorRT, VapourSynth, C++ build
- [Training](training.md) -- Data pipeline, teacher training, student distillation, Modal cloud
- [Deployment](deployment.md) -- All encoding pipelines, benchmarks, real-time playback, engine builds

## Model Cards

- [Student Model](model-card-student.md) -- 1.06M param DRUNet, 57 fps, architecture and usage
- [Teacher Model](model-card-teacher.md) -- 32.6M param DRUNet, quality reference, training details

## Research

Active and completed research topics, each in a self-contained folder.

- [research/temporal-context/](research/temporal-context/) -- Recurrent temporal architecture: 9-channel input PRD
- [research/training-data/](research/training-data/) -- Staged data pipeline: extraction, SCUNet GAN targets, degradation mixing
- [research/cpp-pipeline/](research/cpp-pipeline/) -- C++ pipeline PRD (completed: 57 fps achieved)
- [research/raft-alignment/](research/raft-alignment/) -- RAFT optical flow for temporal consistency (explored, deprioritized)
- [research/hypir/](research/hypir/) -- HYPIR diffusion restoration (rejected -- adds AI texture artifacts)
- [research/upscale-a-video/](research/upscale-a-video/) -- Diffusion video SR (rejected -- 100-1000x too slow)
- [research/nerv-denoising/](research/nerv-denoising/) -- NeRV implicit neural video (concluded -- memorizes, doesn't generalize)

## Reference

- [guides/gpu-profiling-guide.md](guides/gpu-profiling-guide.md) -- CUDA profiling with Nsight and torch.profiler
- [guides/modal-graceful-shutdown.md](guides/modal-graceful-shutdown.md) -- Graceful checkpoint saving on Modal spot preemption

## Archive

Outdated docs from the NAFNet era. Kept for historical reference. See [archive/](archive/).
